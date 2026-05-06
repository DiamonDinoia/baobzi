#ifndef BAOBZI_DETAIL_FUNCTION_IMPL_HPP
#define BAOBZI_DETAIL_FUNCTION_IMPL_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <poet/poet.hpp>
#include <polyfit/polyeval.hpp>

// ---------------------------------------------------------------------------
// Compatibility shims for DiamonDinoia/polyfit@main. The pre-rewrite polyfit
// exposed `function_traits`, `value_type_or_identity`, and `has_tuple_size_v`
// at poly_eval / global scope; the new header renames them. We provide
// thin aliases so the rest of this file — which predates the rewrite — keeps
// compiling without a mass-rename.
// ---------------------------------------------------------------------------
namespace poly_eval {
template <class Func>
struct function_traits {
    using arg0_type   = fitInput_t<Func>;
    using result_type = fitOutput_t<Func>;
};
} // namespace poly_eval

template <class T>
inline constexpr bool has_tuple_size_v = poly_eval::detail::hasTupleSize_v<T>;

template <class T>
struct value_type_or_identity {
    using type = poly_eval::detail::value_type_or_t<T>;
};

namespace baobzi {

/// Tolerance interpretation for the tree's adaptive refinement.
enum class TolKind : int {
    RelativeTail = 0,   ///< relative tail-coefficient estimate (1D only)
    AbsoluteTail = 1,   ///< absolute tail-coefficient estimate (1D only)
    RelativeMax  = 2,   ///< sample-based, max-abs relative error
    AbsoluteMax  = 3,   ///< sample-based, max-abs absolute error
    RelativeL2   = 4,   ///< sample-based, L2 relative error
    AbsoluteL2   = 5,   ///< sample-based, L2 absolute error
};

namespace detail {
/// Internal fit-time configuration. Replaces the legacy `baobzi_input_t`.
struct TreeInput {
    int     input_dim             = 0;
    int     output_dim            = 1;
    int     degree                = 8;
    double  tol                   = 0.0;
    double  minimum_leaf_fraction = 0.0;
    int     min_depth             = 0;
    int     max_depth             = 50;
    int     n_samples_per_dim     = 8;
    /// Soft cap on accumulated leaf storage, in MiB. The fit aborts with
    /// `MemoryBudgetExceeded` once `polyfits.size() * sizeof(poly_eval_type)`
    /// + node storage crosses this. Set to 0 to disable. Default 64 MiB —
    /// sized to stay below typical LLCs so the evaluator coexists with the
    /// caller's working set; ambitious 3D fits should raise it deliberately.
    int     max_memory_mib        = 64;
    /// When true, panels that fail tolerance at `max_depth` are accepted as
    /// best-effort leaves instead of throwing `MaxDepthExceeded`. The
    /// resulting `Function` exposes the unconverged panel list via
    /// `non_converged_panels()`. Default false (throw).
    bool    allow_max_depth_leaves = false;
    TolKind tol_kind              = TolKind::RelativeMax;
};
} // namespace detail

/// A panel where adaptive paneling failed to meet `tol` at the configured
/// `max_depth`. Reported either through `MaxDepthExceeded::panels()` (the
/// default throwing path) or through `Function::non_converged_panels()`
/// when `options.allow_max_depth_leaves == true`.
struct NonConvergedPanel {
    std::vector<double> a;     ///< Lower bound of the half-open panel.
    std::vector<double> b;     ///< Upper bound of the half-open panel.
    std::size_t         depth; ///< Tree depth at which convergence failed.
};

/// Thrown when adaptive paneling hits `options.max_depth` without converging.
/// Carries the offending panel as a half-open domain [a, b) and the depth,
/// so callers can identify the singular region — typically a latent
/// singularity in the function that sampling alone cannot resolve.
class MaxDepthExceeded : public std::exception {
  public:
    MaxDepthExceeded() = default;

    template <class VecC, class VecH>
    MaxDepthExceeded(std::size_t depth, const VecC &center_in, const VecH &half_length_in)
        : depth_(depth) {
        auto cit = center_in.begin();
        auto hit = half_length_in.begin();
        for (; cit != center_in.end() && hit != half_length_in.end(); ++cit, ++hit) {
            a_.push_back(*cit - *hit);
            b_.push_back(*cit + *hit);
        }
        std::ostringstream os;
        os << "Baobzi fit error: tree depth exceeded max allowed input depth ("
           << depth_ << ") on panel ";
        format_domain(os);
        os << " — likely a singularity in this interval; "
              "subdivide manually away from the singularity or raise options.max_depth.";
        msg_ = os.str();
    }

    /// Construct from a list of unconverged panels (the aggregate path).
    /// `panels` must be non-empty; the first entry's bounds are mirrored into
    /// the legacy single-panel accessors `a()/b()` for backwards compatibility.
    explicit MaxDepthExceeded(std::vector<NonConvergedPanel> panels)
        : panels_(std::move(panels)) {
        if (!panels_.empty()) {
            depth_ = panels_.front().depth;
            a_     = panels_.front().a;
            b_     = panels_.front().b;
        }
        std::ostringstream os;
        os << "Baobzi fit error: tree depth exceeded max allowed input depth ("
           << depth_ << ") on " << panels_.size() << " panel"
           << (panels_.size() == 1 ? "" : "s") << "; first ";
        format_domain(os);
        os << " — likely a singularity; subdivide manually, raise "
              "options.max_depth, or set options.allow_max_depth_leaves=true "
              "to accept best-effort leaves.";
        msg_ = os.str();
    }

    std::size_t                depth() const noexcept { return depth_; }
    /// Lower bound(s) of the first unconverged panel [a, b). Legacy accessor;
    /// for the full list use `panels()`.
    const std::vector<double> &a()     const noexcept { return a_; }
    /// Upper bound(s) of the first unconverged panel [a, b).
    const std::vector<double> &b()     const noexcept { return b_; }
    /// Full list of unconverged panels gathered before the throw.
    const std::vector<NonConvergedPanel> &panels() const noexcept { return panels_; }
    const char                *what()  const noexcept override { return msg_.c_str(); }

  private:
    std::size_t         depth_ = 0;
    std::vector<double> a_;
    std::vector<double> b_;
    std::vector<NonConvergedPanel> panels_;
    std::string         msg_ =
        "Baobzi fit error: tree depth exceeded max allowed input depth";

    void format_domain(std::ostringstream &os) const {
        if (a_.size() == 1) {
            os << "[" << a_[0] << ", " << b_[0] << ")";
            return;
        }
        os << "[";
        for (std::size_t i = 0; i < a_.size(); ++i)
            os << (i ? " x " : "") << "[" << a_[i] << ", " << b_[i] << ")";
        os << "]";
    }
};

/// Thrown when adaptive paneling pushes accumulated leaf storage past
/// `options.max_memory_mib`. Carries the amount used, the budget, and the
/// offending half-open panel [a, b) — the same shape as `MaxDepthExceeded`
/// — so callers can either raise the budget or excise the singular region.
class MemoryBudgetExceeded : public std::exception {
  public:
    MemoryBudgetExceeded() = default;

    template <class VecC, class VecH>
    MemoryBudgetExceeded(std::size_t used_bytes, std::size_t budget_bytes,
                         const VecC &center_in, const VecH &half_length_in)
        : used_bytes_(used_bytes), budget_bytes_(budget_bytes) {
        auto cit = center_in.begin();
        auto hit = half_length_in.begin();
        for (; cit != center_in.end() && hit != half_length_in.end(); ++cit, ++hit) {
            a_.push_back(*cit - *hit);
            b_.push_back(*cit + *hit);
        }
        std::ostringstream os;
        os << "Baobzi fit error: leaf-storage exceeded budget ("
           << (static_cast<double>(used_bytes_)   / (1024.0 * 1024.0)) << " MiB used vs "
           << (static_cast<double>(budget_bytes_) / (1024.0 * 1024.0)) << " MiB budget) on panel ";
        if (a_.size() == 1)
            os << "[" << a_[0] << ", " << b_[0] << ")";
        else {
            os << "[";
            for (std::size_t i = 0; i < a_.size(); ++i)
                os << (i ? " x " : "") << "[" << a_[i] << ", " << b_[i] << ")";
            os << "]";
        }
        os << " — raise options.max_memory_mib (or set to 0 to disable), "
              "or restrict the fit domain to skip this region.";
        msg_ = os.str();
    }

    std::size_t                used_bytes()   const noexcept { return used_bytes_; }
    std::size_t                budget_bytes() const noexcept { return budget_bytes_; }
    const std::vector<double> &a()            const noexcept { return a_; }
    const std::vector<double> &b()            const noexcept { return b_; }
    const char                *what()         const noexcept override { return msg_.c_str(); }

  private:
    std::size_t         used_bytes_   = 0;
    std::size_t         budget_bytes_ = 0;
    std::vector<double> a_;
    std::vector<double> b_;
    std::string         msg_ =
        "Baobzi fit error: leaf-storage exceeded budget";
};

namespace detail {
/// Wrapper class that treats arrays and scalars as a common type with basic
/// arithmetic operators.
template <typename T, std::size_t N>
class Value {
    using storage_t = std::conditional_t<N == 1, T, std::array<T, N>>;
    storage_t data_{};

  public:
    // Scalar constructor
    template <std::size_t M = N, typename = std::enable_if_t<M == 1>>
    Value(const T &val) : data_(val) {}
    Value(const std::array<T, 1> &arr) { data_ = arr[0]; }

    // Array constructor
    template <std::size_t M = N, typename = std::enable_if_t<M != 1>>
    Value(const std::array<T, N> &arr) : data_(arr) {}
    Value() = default;
    Value(const Value &) = default;
    Value(Value &&) noexcept = default;
    Value &operator=(const Value &) = default;
    Value &operator=(Value &&) noexcept = default;
    ~Value() = default;

    Value(const T *arr) {
        if constexpr (N == 1) {
            data_ = arr[0];
        } else {
            poet::static_for<N>([&](auto I) {
                constexpr std::size_t i = I;
                data_[i] = arr[i];
            });
        }
    }

    inline Value operator+(const Value &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ + rhs.data_));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] + rhs.data_[i];
            return Value(result);
        }
    }

    inline Value operator+(const T &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ + rhs));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] + rhs;
            return Value(result);
        }
    }

    inline Value operator-(const Value &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ - rhs.data_));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] - rhs.data_[i];
            return Value(result);
        }
    }

    inline Value operator-(const T &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ - rhs));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] - rhs;
            return Value(result);
        }
    }

    inline Value operator*(const Value &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ * rhs.data_));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] * rhs.data_[i];
            return Value(result);
        }
    }

    inline Value operator*(const T &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ * rhs));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] * rhs;
            return Value(result);
        }
    }

    inline Value operator/(const Value &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ / rhs.data_));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] / rhs.data_[i];
            return Value(result);
        }
    }

    inline Value operator/(const T &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ / rhs));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] / rhs;
            return Value(result);
        }
    }

    inline T &operator[](std::size_t idx) {
        if constexpr (N == 1) {
            static_cast<void>(idx);
            return data_;
        } else {
            return data_[idx];
        }
    }

    inline const T &operator[](std::size_t idx) const {
        if constexpr (N == 1) {
            static_cast<void>(idx);
            return data_;
        } else {
            return data_[idx];
        }
    }

    inline T *begin() {
        if constexpr (N == 1) {
            return &data_;
        } else {
            return data_.data();
        }
    }

    inline T *end() {
        if constexpr (N == 1) {
            return &data_ + 1;
        } else {
            return data_.data() + N;
        }
    }

    inline const T *begin() const {
        if constexpr (N == 1) {
            return &data_;
        } else {
            return data_.data();
        }
    }

    inline const T *end() const {
        if constexpr (N == 1) {
            return &data_ + 1;
        } else {
            return data_.data() + N;
        }
    }

    inline T prod() const {
        if constexpr (N == 1) {
            return data_;
        } else {
            T result = T{1};
            for (const auto &val : data_)
                result *= val;
            return result;
        }
    }

    inline storage_t get() const { return data_; }

    // Automatic casting to scalar or array
    inline operator T() const {
        static_assert(N == 1, "Can only cast to scalar if N == 1");
        return data_;
    }

    inline operator std::array<T, N>() const {
        static_assert(N != 1, "Can only cast to array if N != 1");
        return data_;
    }

    // Access underlying data
    inline const T &scalar() const {
        static_assert(N == 1, "Not a scalar");
        return data_;
    }
    inline const std::array<T, N> &array() const {
        static_assert(N != 1, "Not an array");
        return data_;
    }

    inline std::array<T, N> as_array() const {
        if constexpr (N == 1) {
            return std::array<T, N>{data_};
        } else {
            return data_;
        }
    }
};

template <typename T>
Value(const T &) -> Value<T, 1>;

template <typename T, std::size_t N>
Value(const std::array<T, N> &) -> Value<T, N>;

using index_t = std::size_t; ///< Type specifying indexing into flattened tree

template <int EXP, typename T>
constexpr T powi(T base) {
    if constexpr (EXP == 0) {
        return T{1};
    } else if constexpr (EXP % 2 == 0) {
        const auto half = powi<EXP / 2>(base);
        return half * half;
    } else {
        return base * powi<EXP - 1>(base);
    }
}

/// Trait to extract number of elements from input/output types.
template <typename T>
constexpr std::size_t get_tuple_size() {
    if constexpr (has_tuple_size_v<T>)
        return std::tuple_size_v<T>;
    else
        return 1;
}

/// Geometric portion of Baobzi nodes.
template <typename T, std::size_t Dim>
struct Box {
    Value<T, Dim> center;
    Value<T, Dim> half_length;

    inline Box(const auto &x, const auto &hl) : center{x}, half_length{hl} {}
};

/// Check whether the leading-coefficient 'tail estimate' for a given fit
/// exceeds the requested tolerance.
template <class Polyfit>
inline bool tail_error_check(TolKind tol_type, double tol, const Polyfit &polyfit) {
    constexpr std::size_t input_dim = get_tuple_size<typename Polyfit::InputType>();
    constexpr std::size_t output_dim = get_tuple_size<typename Polyfit::OutputType>();
    using T = typename value_type_or_identity<typename Polyfit::InputType>::type;
    if constexpr (input_dim != 1)
        throw std::runtime_error(
            "Baobzi fit error: tail_error check is currently only implemented for 1D input; "
            "use a sample-based tol_type (RELATIVE_MAX, ABSOLUTE_MAX, RELATIVE_L2, ABSOLUTE_L2) instead");

    T maxcoeff{0.0};
    T scaling_factor{1.0};

    if constexpr (input_dim == 1) {
        const auto &coeffs = polyfit.coeffs();
        constexpr std::size_t N = Polyfit::NCOEFFS;
        static_assert(output_dim == 1, "tail_error only implemented for single output in 1D");

        for (std::size_t i = 0; i < 2; ++i)
            maxcoeff = std::max(std::abs(coeffs[i]), maxcoeff);
        scaling_factor = std::max(scaling_factor, std::abs(coeffs[N - 1]));
    }

    if (tol_type == TolKind::RelativeL2 || tol_type == TolKind::RelativeMax)
        return maxcoeff / scaling_factor > tol;
    else
        return maxcoeff > tol;
}

/// Sample a polynomial fit on a uniform grid of points and compare to the
/// actual function. Returns true if measured error exceeds tol.
template <class Func, class Polyfit>
inline bool
sample_error_check(int n_sample_1d, TolKind tol_type, double tol, const typename Polyfit::InputType &center_in,
                   const typename Polyfit::InputType &half_length_in, const Func &func, const Polyfit &polyfit) {
    constexpr std::size_t input_dim = get_tuple_size<typename Polyfit::InputType>();
    constexpr std::size_t output_dim = get_tuple_size<typename Polyfit::OutputType>();
    const std::size_t n_sample_1d_sz = static_cast<std::size_t>(n_sample_1d);
    const std::size_t n_samples = powi<static_cast<int>(input_dim)>(n_sample_1d_sz);
    const Value half_length = half_length_in;
    const Value center = center_in;

    double max_abs_err{0.0};
    double max_rel_err{0.0};
    double abs_err_l2{0.0};
    double direct_sum{0.0};
    for (std::size_t linear_index = 0; linear_index < n_samples; ++linear_index) {
        Value<double, input_dim> sample_point;
        std::size_t curr_index = linear_index;

        for (std::size_t dim = 0; dim < input_dim; ++dim) {
            const double dx = 2.0 * half_length[dim] / static_cast<double>(n_sample_1d_sz);
            sample_point[dim] = center[dim] - half_length[dim] + dx / 2.0 +
                                dx * static_cast<double>(curr_index % n_sample_1d_sz);
            curr_index /= n_sample_1d_sz;
        }

        Value<double, output_dim> actual = func(sample_point);
        Value<double, output_dim> approx = polyfit(sample_point);

        for (std::size_t i = 0; i < output_dim; ++i) {
            const double abs_err = std::abs(approx[i] - actual[i]);
            max_abs_err = std::max(max_abs_err, abs_err);
            if (actual[i] != 0.0)
                max_rel_err = std::max(max_rel_err, std::abs(abs_err / actual[i]));
            abs_err_l2 += powi<2>(abs_err);
            direct_sum += powi<2>(actual[i]);
        }
    }

    switch (tol_type) {
    case TolKind::RelativeL2:
        return std::sqrt(abs_err_l2 / direct_sum) > tol;
    case TolKind::AbsoluteL2:
        return std::sqrt(abs_err_l2) / static_cast<double>(n_samples * output_dim) > tol;
    case TolKind::RelativeMax:
        return max_rel_err > tol;
    case TolKind::AbsoluteMax:
        return max_abs_err > tol;
    default:
        throw std::runtime_error("Baobzi fit error: unknown tolerance type for sampling");
    }
}

/// Node in baobzi::PolyTree. If leaf, contains evaluation data, otherwise
/// references children by index.
template <class Func, std::size_t Degree>
class Node {
  public:
    using input_type = std::remove_cvref_t<typename poly_eval::function_traits<Func>::arg0_type>;
    using output_type = typename poly_eval::function_traits<Func>::result_type;
    using value_type = typename value_type_or_identity<input_type>::type;
    using poly_eval_type =
        typename std::conditional<has_tuple_size_v<input_type>, poly_eval::FuncEvalND<Func, Degree, poly_eval::FusionMode::Never>,
                                  poly_eval::FuncEval<Func, Degree, 1, poly_eval::FusionMode::Never>>::type;

    static constexpr std::size_t input_dim = get_tuple_size<input_type>();
    static constexpr std::size_t output_dim = get_tuple_size<output_type>();

    static constexpr std::uint32_t kLeafSentinel = std::numeric_limits<std::uint32_t>::max();

    // Slim 8-B node (Phase 9 / Layer A): no `center`, no `_pad_`, and
    // `poly_eval_id` shrunk to uint32. `center` is recomputed on the fly
    // during descent from the subtree's (lo, hi) carried in registers, and
    // the caller-side leaf-id table already reads `poly_eval_id` as uint32.
    // 8 nodes/cache line for every Dim — descent walks ~5× fewer lines vs
    // the old 40-B 3D node. `first_child_idx == kLeafSentinel` is the leaf
    // sentinel; `poly_eval_id` is meaningful only when leaf.
    std::uint32_t first_child_idx = kLeafSentinel;
    std::uint32_t poly_eval_id    = kLeafSentinel;

    Node() = default;

    inline bool is_leaf() const { return first_child_idx == kLeafSentinel; }

    /// Fit this node to the requested tolerance. On success, stores the
    /// poly_eval_id into polyfits and returns true. Center/half_length are
    /// passed in (the runtime Node no longer carries `center`).
    bool fit(const detail::TreeInput &input, const Func &func,
             const Value<value_type, input_dim> &center,
             const Value<value_type, input_dim> &half_length,
             const std::vector<value_type> &samples,
             std::vector<poly_eval_type> &polyfits) {
        if (!samples.empty())
            throw std::runtime_error("Baobzi fit error: sample points not yet supported");

        const auto n_polyfit_before = polyfits.size();
        const input_type lb = center - half_length;
        const input_type ub = center + half_length;

        auto rollback_and_fail = [&polyfits, n_polyfit_before]() {
            while (polyfits.size() != n_polyfit_before)
                polyfits.pop_back();
            return false;
        };

        auto polyfit = polyfits.emplace_back(func, lb, ub);

        if (input.tol_kind == TolKind::RelativeTail || input.tol_kind == TolKind::AbsoluteTail) {
            if (tail_error_check(input.tol_kind, input.tol, polyfit))
                return rollback_and_fail();
        } else {
            if (sample_error_check(input.n_samples_per_dim, input.tol_kind, input.tol,
                                   center, half_length, func, polyfit))
                return rollback_and_fail();
        }

        poly_eval_id = static_cast<std::uint32_t>(n_polyfit_before);
        return true;
    }

    /// Accept the polynomial as a leaf even though the tolerance check
    /// failed. Used at `max_depth` when `allow_max_depth_leaves` is set.
    void force_fit_as_leaf(const Func &func,
                           const Value<value_type, input_dim> &center,
                           const Value<value_type, input_dim> &half_length,
                           std::vector<poly_eval_type> &polyfits) {
        const input_type lb = center - half_length;
        const input_type ub = center + half_length;
        polyfits.emplace_back(func, lb, ub);
        poly_eval_id = static_cast<std::uint32_t>(polyfits.size() - 1);
    }

    inline std::size_t memory_usage() const { return sizeof(*this); }
};

// Phase 9 / Layer A: lock the slim 8-B node invariant. If this fires,
// either an extra field was added or alignment regressed — the descent
// load-volume win in `get_node_index` depends on 8 nodes/cache line.
namespace detail_node_size_check {
struct ScalarFn {
    double operator()(double) const { return 0.0; }
};
struct Array2Fn {
    std::array<double, 1> operator()(std::array<double, 2>) const { return {0.0}; }
};
struct Array3Fn {
    std::array<double, 1> operator()(std::array<double, 3>) const { return {0.0}; }
};
static_assert(sizeof(Node<ScalarFn, 8>) == 8, "slim Node expected to be 8 B (1D)");
static_assert(sizeof(Node<Array2Fn, 8>) == 8, "slim Node expected to be 8 B (2D)");
static_assert(sizeof(Node<Array3Fn, 8>) == 8, "slim Node expected to be 8 B (3D)");
} // namespace detail_node_size_check

/// Represent a function over some domain as a tree of Chebyshev nodes.
template <std::size_t Degree, class Func>
struct PolyTree {
    using input_type = std::remove_cvref_t<typename poly_eval::function_traits<Func>::arg0_type>;
    using value_type = typename value_type_or_identity<input_type>::type;
    using poly_eval_type =
        typename std::conditional<has_tuple_size_v<input_type>, poly_eval::FuncEvalND<Func, Degree, poly_eval::FusionMode::Never>,
                                  poly_eval::FuncEval<Func, Degree, 1, poly_eval::FusionMode::Never>>::type;
    using output_type = typename poly_eval::function_traits<Func>::result_type;

    static constexpr std::size_t output_dim = get_tuple_size<output_type>();
    static constexpr std::size_t input_dim = get_tuple_size<input_type>();
    static constexpr std::size_t n_child = std::size_t{1} << input_dim;

    using node_t = Node<Func, Degree>;
    using box_t = Box<value_type, input_dim>;
    using dim_array_t = Value<value_type, input_dim>;

    inline PolyTree(const detail::TreeInput &input, const Box<value_type, input_dim> &root_box,
                    std::vector<poly_eval_type> &polyfits, const Func &func)
        : lower_(root_box.center - root_box.half_length),
          upper_(root_box.center + root_box.half_length) {
        std::queue<box_t> q;
        dim_array_t half_width = root_box.half_length * value_type{0.5};
        q.push(root_box);

        index_t curr_child_idx = 1;
        max_depth_ = 0;
        while (!q.empty()) {
            const std::size_t n_next = q.size();
            const std::size_t node_index = nodes_.size();
            // True when this is the last allowed level: any node that fails
            // its tolerance check here cannot be subdivided further.
            const bool at_max_depth =
                max_depth_ == static_cast<std::size_t>(input.max_depth);
            for (std::size_t i = 0; i < n_next; ++i) {
                box_t current_box = q.front();
                q.pop();

                nodes_.emplace_back();

                auto &node = nodes_[i + node_index];
                const bool successful_fit = node.fit(input, func, current_box.center,
                                                     current_box.half_length, {}, polyfits);

                if (successful_fit) {
                    assert(polyfits.size() > 0);
                    assert(node.poly_eval_id == polyfits.size() - 1);
                } else if (at_max_depth) {
                    // Record the failed panel and force-accept the polynomial
                    // as a best-effort leaf. The decision to throw or accept
                    // is made at the Function level after all subtrees finish,
                    // so a multi-subtree fit can report panels from every
                    // failing subtree rather than just the first one to throw.
                    const auto a_arr = (current_box.center - current_box.half_length).as_array();
                    const auto b_arr = (current_box.center + current_box.half_length).as_array();
                    non_converged_panels_.push_back(NonConvergedPanel{
                        std::vector<double>(a_arr.begin(), a_arr.end()),
                        std::vector<double>(b_arr.begin(), b_arr.end()),
                        max_depth_});
                    node.force_fit_as_leaf(func, current_box.center,
                                           current_box.half_length, polyfits);
                } else {
                    node.first_child_idx = static_cast<std::uint32_t>(curr_child_idx);
                    curr_child_idx += n_child;

                    const dim_array_t &node_center = current_box.center;
                    for (index_t child = 0; child < n_child; ++child) {
                        dim_array_t center_offset;

                        // Extract sign of each offset component from the bits of child.
                        for (std::size_t j = 0; j < input_dim; ++j) {
                            const value_type signed_hw[2] = {-half_width[j], half_width[j]};
                            center_offset[j] = node_center[j] + signed_hw[(child >> j) & index_t{1}];
                        }

                        q.push(box_t(center_offset, half_width));
                    }
                }
            }

            // At max_depth, all failing panels were force-accepted as leaves;
            // q stays empty for those, so the BFS terminates cleanly. The
            // throw/accept decision happens at the Function level once all
            // subtrees have finished, in `gather_non_converged_panels()`.

            if (!q.empty())
                ++max_depth_;
            if (input.max_memory_mib > 0 && !q.empty()) {
                const std::size_t budget =
                    static_cast<std::size_t>(input.max_memory_mib) * std::size_t{1024} * std::size_t{1024};
                const std::size_t used = polyfits.size() * sizeof(poly_eval_type)
                                       + nodes_.size() * sizeof(node_t);
                if (used > budget) {
                    const auto &offender = q.front();
                    throw MemoryBudgetExceeded(used, budget,
                                               offender.center.as_array(),
                                               offender.half_length.as_array());
                }
            }

            half_width = half_width * value_type{0.5};
        }

        // L9 (Phase 10b): for shallow subtrees, build a quantize-to-leaf
        // table so eval-time descent collapses to a single load. The
        // budget is 64 KiB per subtree (table is `uint32_t`, 4 B/entry,
        // size `1 << (input_dim * max_depth_)`). For our scenarios:
        //   1D gauss depth 5 → 32 entries (128 B)
        //   1D runge depth 6 → 64 entries (256 B)
        //   2D bump  depth 7 → 16 K   (64 KiB) — exactly at budget
        //   3D gauss depth 3 → 512    (2 KiB)
        // Builds cost O(2^(Dim*depth)) descents at fit-time; eval-time
        // saves the per-level (load, compare, branch) chain entirely.
        constexpr std::size_t kTableMaxEntries = std::size_t{1} << 14; // 64 KiB / 4 B
        if (max_depth_ > 0) {
            const std::size_t total_bits = input_dim * max_depth_;
            if (total_bits <= 14) {
                const std::size_t n = std::size_t{1} << total_bits;
                if (n <= kTableMaxEntries) {
                    leaf_table_.assign(n, std::uint32_t{0});
                    leaf_table_depth_ = max_depth_;
                    for (std::size_t d = 0; d < input_dim; ++d) {
                        const value_type span = upper_[d] - lower_[d];
                        inv_span_bins_[d] = static_cast<value_type>(
                            std::size_t{1} << max_depth_) / span;
                    }
                    const value_type bins = static_cast<value_type>(
                        std::size_t{1} << max_depth_);
                    for (std::size_t i = 0; i < n; ++i) {
                        // Decode i into per-axis quantize indices q[d].
                        std::size_t r = i;
                        const std::size_t bits = max_depth_;
                        const std::size_t mask = (std::size_t{1} << bits) - 1;
                        // Compute cell-center x and descend the tree.
                        if constexpr (input_dim == 1) {
                            const std::size_t q0 = r & mask;
                            const value_type span = upper_[0] - lower_[0];
                            const value_type cell = span / bins;
                            const value_type xc = lower_[0] +
                                (static_cast<value_type>(q0) + value_type{0.5}) * cell;
                            leaf_table_[i] = nodes_[get_node_index(xc)].poly_eval_id;
                        } else {
                            input_type xc;
                            for (std::size_t d = 0; d < input_dim; ++d) {
                                const std::size_t qd = r & mask;
                                r >>= bits;
                                const value_type span = upper_[d] - lower_[d];
                                const value_type cell = span / bins;
                                xc[d] = lower_[d] +
                                    (static_cast<value_type>(qd) + value_type{0.5}) * cell;
                            }
                            leaf_table_[i] = nodes_[get_node_index(xc)].poly_eval_id;
                        }
                    }
                }
            }
        }
    }

    inline const node_t &find_node(const input_type &x) const { return nodes_[get_node_index(x)]; }

    /// Combined leaf-id lookup: table if available, else descent.
    inline std::uint32_t find_leaf_id(const input_type &x) const {
        if (!leaf_table_.empty()) {
            const std::size_t bits = leaf_table_depth_;
            const std::size_t mask = (std::size_t{1} << bits) - 1;
            std::size_t idx = 0;
            if constexpr (input_dim == 1) {
                std::size_t q0 = static_cast<std::size_t>(
                    (x - lower_[0]) * inv_span_bins_[0]);
                if (q0 > mask) q0 = mask;
                idx = q0;
            } else {
                for (std::size_t d = 0; d < input_dim; ++d) {
                    std::size_t qd = static_cast<std::size_t>(
                        (x[d] - lower_[d]) * inv_span_bins_[d]);
                    if (qd > mask) qd = mask;
                    idx |= qd << (bits * d);
                }
            }
            return leaf_table_[idx];
        }
        return nodes_[get_node_index(x)].poly_eval_id;
    }

    inline bool has_leaf_table() const noexcept { return !leaf_table_.empty(); }

    /// Layer A: leaf-id lookup that returns `ood_id` for points outside this
    /// subtree's domain. Caller must have verified `has_leaf_table()`.
    ///
    /// The signed `vcvttsd2si` + unsigned compare collapses the OOD pre-check
    /// (`x[d] < lower_left_[d] || x[d] >= upper_right_[d]`) into the same
    /// quantize that builds the table index: an OOR positive double yields
    /// INT64_MIN (x86-64 indefinite-integer), which as `uint64_t` is 2^63 —
    /// far above any reasonable `mask`. Negative inputs likewise produce
    /// INT64_MIN. One `vcvttsd2si` (lat 5) + one `cmp` per axis replaces the
    /// 2 `vcomisd` of the original OOD pre-check plus the cast safeguard
    /// emitted by the unsigned static_cast in `find_leaf_id`.
    ///
    /// Single-subtree precondition: this subtree's `lower_` matches the
    /// owning Function's `lower_left_`. Multi-subtree callers must keep the
    /// Function-level OOD pre-check + `get_linear_bin` path.
    inline std::uint32_t find_leaf_id_with_ood(const input_type &x,
                                               std::uint32_t ood_id) const {
        const std::size_t bits = leaf_table_depth_;
        const std::size_t mask = (std::size_t{1} << bits) - 1;
        std::size_t idx = 0;
        if constexpr (input_dim == 1) {
            const std::int64_t q0 = static_cast<std::int64_t>(
                (x - lower_[0]) * inv_span_bins_[0]);
            if (static_cast<std::uint64_t>(q0) > mask) return ood_id;
            idx = static_cast<std::size_t>(q0);
        } else {
            for (std::size_t d = 0; d < input_dim; ++d) {
                const std::int64_t qd = static_cast<std::int64_t>(
                    (x[d] - lower_[d]) * inv_span_bins_[d]);
                if (static_cast<std::uint64_t>(qd) > mask) return ood_id;
                idx |= static_cast<std::size_t>(qd) << (bits * d);
            }
        }
        return leaf_table_[idx];
    }

    /// Descent hot loop. Phase 9 / Layers A+E: the node no longer carries
    /// `center`; instead the per-subtree (lo, hi) bounds are carried in
    /// registers and `mid = 0.5 * (lo + hi)` is recomputed each level.
    /// This drops Dim×8 B of per-level center loads — the descent is no
    /// longer load-bound on the center field. For ND, the per-axis
    /// `x[i] > mid[i]` lowers to a packed `vcmpgtpd + vmovmskpd` with the
    /// loop unrolled (input_dim is constexpr), collapsing the serial
    /// scalar-compare chain.
    ///
    /// Bit-exactness vs the pre-Phase-9 evaluator is not guaranteed at
    /// boundary points: this `mid` is computed differently from the
    /// fit-time `box.center` (which is built by chained halving). They
    /// agree algebraically; tests assert relative tolerance, not bit
    /// equality.
    inline std::size_t get_node_index(const input_type &x) const {
        dim_array_t lo = lower_;
        dim_array_t hi = upper_;
        index_t curr_index = 0;
        while (!nodes_[curr_index].is_leaf()) {
            index_t child_idx = 0;
            if constexpr (has_tuple_size_v<input_type>) {
                for (std::size_t i = 0; i < input_dim; ++i) {
                    const value_type mid_i = (lo[i] + hi[i]) * value_type{0.5};
                    const bool upper = (x[i] > mid_i);
                    child_idx |= (static_cast<index_t>(upper) << i);
                    (upper ? lo[i] : hi[i]) = mid_i;
                }
            } else {
                const value_type mid = (lo[0] + hi[0]) * value_type{0.5};
                const bool upper = (x > mid);
                child_idx = static_cast<index_t>(upper);
                (upper ? lo[0] : hi[0]) = mid;
            }
            curr_index = nodes_[curr_index].first_child_idx + child_idx;
        }
        return curr_index;
    }

    inline std::size_t size() const { return nodes_.size(); }
    inline std::size_t max_depth() const { return max_depth_; }

    inline std::size_t memory_usage() const {
        std::size_t total = sizeof(*this);
        for (const auto &node : nodes_)
            total += node.memory_usage();
        return total;
    }

    inline auto &get_nodes() { return nodes_; }
    inline auto &get_nodes() const { return nodes_; }

    inline const std::vector<NonConvergedPanel> &non_converged_panels() const {
        return non_converged_panels_;
    }

  private:
    std::vector<node_t> nodes_;
    // Subtree bounding box, set in the constructor. Carried into descent so
    // the runtime Node doesn't need a `center` field (Phase 9 / Layer A).
    dim_array_t lower_{};
    dim_array_t upper_{};
    std::size_t max_depth_ = 0;
    // L9: quantize→leaf table for shallow subtrees. Empty when not built.
    std::vector<std::uint32_t> leaf_table_;
    std::size_t leaf_table_depth_ = 0;
    // Precomputed `(1.0 / span) * 2^depth` per axis so the per-point
    // quantize is a multiply (vmulsd, lat 3) instead of a divide
    // (vdivsd, lat 14).
    std::array<value_type, input_dim> inv_span_bins_{};
    std::vector<NonConvergedPanel> non_converged_panels_;
};
} // namespace detail

/// Represents a function over some domain as a grid of baobzi::detail::PolyTree
/// objects.
template <std::size_t Degree, class Func>
class Function {
  public:
    using input_type = std::remove_cvref_t<typename poly_eval::function_traits<Func>::arg0_type>;
    using output_type = typename poly_eval::function_traits<Func>::result_type;
    using value_type = typename value_type_or_identity<input_type>::type;
    using poly_eval_type =
        typename std::conditional<has_tuple_size_v<input_type>, poly_eval::FuncEvalND<Func, Degree, poly_eval::FusionMode::Never>,
                                  poly_eval::FuncEval<Func, Degree, 1, poly_eval::FusionMode::Never>>::type;
    static constexpr auto degree = Degree;

    static constexpr std::size_t input_dim = detail::get_tuple_size<input_type>();
    static constexpr std::size_t output_dim = detail::get_tuple_size<output_type>();
    static constexpr std::size_t n_child = std::size_t{1} << input_dim;

    using node_t = detail::Node<Func, Degree>;
    using box_t = detail::Box<value_type, input_dim>;
    using dim_array_t = detail::Value<value_type, input_dim>;

    inline std::size_t memory_usage() const {
        std::size_t mem = sizeof(*this);
        mem += polyfits_.capacity() * sizeof(poly_eval_type);
        for (const auto &subtree : subtrees_)
            mem += subtree.memory_usage();
        return mem;
    }

    inline void print_stats() const {
        std::size_t n_nodes = 0;
        std::size_t n_leaves = 0;
        const std::size_t n_subtrees = subtrees_.size();
        std::size_t max_tree_depth = 0;
        const std::size_t mem = memory_usage();
        for (const auto &subtree : subtrees_) {
            n_nodes += subtree.size();
            max_tree_depth = std::max(max_tree_depth, subtree.max_depth());
            for (const auto &node : subtree.get_nodes())
                n_leaves += static_cast<std::size_t>(node.is_leaf());
        }

        std::cout << "Baobzi function mapping " << input_dim << " to " << output_dim << "\n";
        std::cout << "Tree represented by " << n_nodes << " nodes, of which " << n_leaves << " are leaves\n";
        std::cout << "Nodes are distributed across " << n_subtrees << " subtrees at an initial depth of "
                  << stats_.base_depth << " with a maximum subtree depth of " << max_tree_depth << "\n";
        std::cout << "Total function evaluations required for fit: "
                  << n_nodes * static_cast<std::size_t>(std::pow(static_cast<double>(Degree),
                                                                 static_cast<double>(input_dim))) +
                         stats_.n_evals_root
                  << "\n";
        std::cout << "Total time to create tree: " << stats_.t_elapsed << " milliseconds\n";
        std::cout << "Approximate memory usage of tree: "
                  << static_cast<double>(mem) / (1024.0 * 1024.0) << " MiB\n";
    }

    /// Build a Function object by recursively fitting the domain.
    inline Function(const detail::TreeInput &input, const input_type center, const input_type half_width_in,
                    const Func &func)
        : input_(input),
          box_(dim_array_t{center}, dim_array_t{half_width_in}),
          tol_(input.tol) {
        const auto t_start = std::chrono::steady_clock::now();

        dim_array_t lvec{half_width_in};
        std::queue<box_t> q;
        std::queue<box_t> maybe_q;

        const auto hlmin = *std::min_element(lvec.begin(), lvec.end());
        for (std::size_t i = 0; i < input_dim; ++i)
            n_subtrees_[i] = static_cast<std::size_t>(lvec[i] / hlmin);

        q.push(box_t(center, lvec));

        // Half-width of next children
        dim_array_t half_width = lvec * value_type{0.5};

        // Breadth-first search through the tree, testing each level; we exit
        // as soon as a level is not entirely parent nodes, so we can jump
        // straight to the subtree roots on evaluation.
        while (!q.empty()) {
            const std::size_t n_next = q.size();

            auto add_node_children_to_queue = [](std::queue<box_t> &theq, const dim_array_t &parent_center,
                                                 const dim_array_t &child_hw) {
                for (std::size_t child = 0; child < n_child; ++child) {
                    detail::Value<double, input_dim> offset_center;

                    // Extract sign of each offset component from the bits of child.
                    for (std::size_t j = 0; j < input_dim; ++j) {
                        const value_type signed_hw[2] = {-child_hw[j], child_hw[j]};
                        offset_center[j] = parent_center[j] + signed_hw[(child >> j) & std::size_t{1}];
                    }

                    theq.push(box_t(offset_center, child_hw));
                }
            };

            std::vector<node_t> nodes;
            value_type leaf_fraction = value_type{0.0};
            for (std::size_t i = 0; i < n_next; ++i) {
                box_t current_box = q.front();
                q.pop();

                nodes.emplace_back();
                auto &node = nodes.back();
                std::vector<poly_eval_type> dummy;
                node.fit(input, func, current_box.center, current_box.half_length, {}, dummy);
                if (node.poly_eval_id != 0u)
                    node.poly_eval_id = 0;

                if (!node.is_leaf() || stats_.base_depth < static_cast<std::size_t>(input.min_depth)) {
                    add_node_children_to_queue(q, current_box.center, half_width);
                } else {
                    leaf_fraction += value_type{1.0};
                    add_node_children_to_queue(maybe_q, current_box.center, half_width);
                }
            }
            stats_.n_evals_root += static_cast<std::uint64_t>(
                nodes.size() *
                static_cast<std::size_t>(std::pow(static_cast<double>(Degree), static_cast<double>(input_dim))));

            leaf_fraction /= static_cast<value_type>(nodes.size());
            if (leaf_fraction < input.minimum_leaf_fraction) {
                while (!maybe_q.empty()) {
                    q.push(maybe_q.front());
                    maybe_q.pop();
                }
            }

            half_width = half_width * value_type{0.5};
            const std::size_t expected_full =
                std::size_t{1} << (input_dim * (stats_.base_depth + 1));
            if (expected_full == q.size()) {
                n_subtrees_ = n_subtrees_ * std::size_t{2};
                ++stats_.base_depth;
                if (stats_.base_depth > static_cast<std::size_t>(input.max_depth)) {
                    const auto &offender = q.front();
                    throw MaxDepthExceeded(stats_.base_depth,
                                           offender.center.as_array(),
                                           offender.half_length.as_array());
                }
            } else {
                break;
            }
        }

        dim_array_t bin_size;
        for (std::size_t j = 0; j < input_dim; ++j) {
            bin_size[j] = 2.0 * box_.half_length[j] / static_cast<value_type>(n_subtrees_[j]);
            inv_bin_size_[j] = 0.5 * static_cast<value_type>(n_subtrees_[j]) / box_.half_length[j];
        }
        lower_left_ = box_.center - box_.half_length;
        upper_right_ = box_.center + box_.half_length;

        subtrees_.reserve(n_subtrees_.prod());

        auto input_local = input;
        input_local.max_depth -= static_cast<int>(stats_.base_depth);
        const std::size_t total_bins = n_subtrees_.prod();
        for (std::size_t i_bin = 0; i_bin < total_bins; ++i_bin) {
            const std::array<std::size_t, input_dim> bins = get_bins(i_bin);

            dim_array_t parent_center;
            for (std::size_t i = 0; i < input_dim; ++i)
                parent_center[i] =
                    (static_cast<value_type>(bins[i]) + value_type{0.5}) * bin_size[i] + lower_left_[i];

            box_t subtree_root = {parent_center, bin_size * value_type{0.5}};
            subtrees_.emplace_back(input_local, subtree_root, polyfits_, func);
        }

        // Aggregate any per-subtree non-converged panels. Default behaviour
        // prints them to cerr and throws; opt-in keeps the list on the
        // Function for `non_converged_panels()` introspection.
        for (const auto &subtree : subtrees_)
            for (const auto &p : subtree.non_converged_panels())
                non_converged_panels_.push_back(p);

        if (!non_converged_panels_.empty()) {
            if (!input.allow_max_depth_leaves) {
                std::cerr << "Baobzi fit warning: " << non_converged_panels_.size()
                          << " panel" << (non_converged_panels_.size() == 1 ? "" : "s")
                          << " failed to converge at max_depth=" << input.max_depth << ":\n";
                for (const auto &p : non_converged_panels_) {
                    std::cerr << "  [";
                    for (std::size_t k = 0; k < p.a.size(); ++k)
                        std::cerr << (k ? " x " : "") << "[" << p.a[k] << ", " << p.b[k] << ")";
                    std::cerr << "]\n";
                }
                throw MaxDepthExceeded(non_converged_panels_);
            }
        }

        const auto t_end = std::chrono::steady_clock::now();
        stats_.t_elapsed = static_cast<std::uint32_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
        build_cache();
    }

    inline void build_cache() {}

    /// Convert linear bin index to [dim] bin vector.
    inline std::array<std::size_t, input_dim> get_bins(const std::size_t i_bin) const {
        if constexpr (input_dim == 1) {
            return std::array<std::size_t, input_dim>{i_bin};
        } else if constexpr (input_dim == 2) {
            return std::array<std::size_t, input_dim>{i_bin % n_subtrees_[0], i_bin / n_subtrees_[0]};
        } else if constexpr (input_dim == 3) {
            return std::array<std::size_t, input_dim>{i_bin % n_subtrees_[0],
                                                      (i_bin / n_subtrees_[0]) % n_subtrees_[1],
                                                      i_bin / (n_subtrees_[0] * n_subtrees_[1])};
        } else if constexpr (input_dim == 4) {
            return std::array<std::size_t, input_dim>{
                i_bin % n_subtrees_[0],
                (i_bin / n_subtrees_[0]) % n_subtrees_[1],
                (i_bin / (n_subtrees_[0] * n_subtrees_[1])) % n_subtrees_[2],
                i_bin / (n_subtrees_[0] * n_subtrees_[1] * n_subtrees_[2])};
        } else if constexpr (input_dim == 5) {
            return std::array<std::size_t, input_dim>{
                i_bin % n_subtrees_[0],
                (i_bin / n_subtrees_[0]) % n_subtrees_[1],
                (i_bin / (n_subtrees_[0] * n_subtrees_[1])) % n_subtrees_[2],
                (i_bin / (n_subtrees_[0] * n_subtrees_[1] * n_subtrees_[2])) % n_subtrees_[3],
                i_bin / (n_subtrees_[0] * n_subtrees_[1] * n_subtrees_[2] * n_subtrees_[3])};
        }
    }

    /// Find linear index of bin at a point.
    inline std::size_t get_linear_bin(const input_type &x) const {
        if constexpr (input_dim == 1) {
            const value_type x_bin = [this, &x]() {
                if constexpr (has_tuple_size_v<input_type>)
                    return x[0] - lower_left_[0];
                else
                    return x - lower_left_[0];
            }();
            return static_cast<std::size_t>(x_bin * inv_bin_size_[0]);
        } else {
            std::array<std::size_t, input_dim> bin{};
            for (std::size_t i = 0; i < input_dim; ++i)
                bin[i] = static_cast<std::size_t>((x[i] - lower_left_[i]) * inv_bin_size_[i]);

            if constexpr (input_dim == 2) {
                return bin[0] + n_subtrees_[0] * bin[1];
            } else if constexpr (input_dim == 3) {
                return bin[0] + n_subtrees_[0] * bin[1] + n_subtrees_[0] * n_subtrees_[1] * bin[2];
            } else if constexpr (input_dim == 4) {
                return bin[0] + n_subtrees_[0] * bin[1] + n_subtrees_[0] * n_subtrees_[1] * bin[2] +
                       n_subtrees_[0] * n_subtrees_[1] * n_subtrees_[2] * bin[3];
            } else if constexpr (input_dim == 5) {
                return bin[0] + n_subtrees_[0] * bin[1] + n_subtrees_[0] * n_subtrees_[1] * bin[2] +
                       n_subtrees_[0] * n_subtrees_[1] * n_subtrees_[2] * bin[3] +
                       n_subtrees_[0] * n_subtrees_[1] * n_subtrees_[2] * n_subtrees_[3] * bin[4];
            }
        }
    }

    inline const node_t &find_node(const input_type &x) const { return subtrees_[get_linear_bin(x)].find_node(x); }

    /// Layer L5 (Phase 16): thread-local scratch buffers for the batch
    /// path, hoisted into a single struct so `operator()` can both feed
    /// `eval_batch_tile` and (when L5 is opted-in) shrink_to_fit the
    /// scratch after a streak of small calls. Per-Function-instantiation
    /// TLS — different `Function<...>` types do not share scratch, so
    /// `shrink_to_fit` on one does not stomp on another's hot-batch
    /// capacity.
    ///
    /// Layer L6 (Phase 16, opt-in): when `BAOBZI_FIXED_SCRATCH` is
    /// defined, the four tile-K-bounded buffers (`xp_packed`,
    /// `out_packed`, `perm`, `leaf_ids*`) become `alignas(64)
    /// std::array<T, kMaxTileK>` — zero heap, naturally aligned, no
    /// fragmentation. `counts`/`offsets` are sized by `n_leaves` (not
    /// tile_K) and stay heap-backed; they are tiny (~tens of KiB) and
    /// fit L1d. The compile-time `kMaxTileK` (default 65 536, override
    /// via `BAOBZI_FIXED_SCRATCH_MAX_TILE_K`) caps the per-template
    /// per-thread footprint. With `input_dim=3, output_dim=1, kMaxTileK
    /// = 65 536`: ~2.4 MiB.
    struct Scratch {
#ifdef BAOBZI_FIXED_SCRATCH
#  ifndef BAOBZI_FIXED_SCRATCH_MAX_TILE_K
#    define BAOBZI_FIXED_SCRATCH_MAX_TILE_K 65536u
#  endif
        static constexpr std::size_t kMaxTileK = BAOBZI_FIXED_SCRATCH_MAX_TILE_K;
        static constexpr bool fixed = true;
        alignas(64) std::array<std::uint16_t, kMaxTileK>             leaf_ids16{};
        alignas(64) std::array<std::uint32_t, kMaxTileK>             leaf_ids32{};
        alignas(64) std::array<std::uint32_t, kMaxTileK>             perm{};
        alignas(64) std::array<value_type,    input_dim * kMaxTileK> xp_packed{};
        alignas(64) std::array<value_type,    output_dim * kMaxTileK> out_packed{};
#else
        static constexpr std::size_t kMaxTileK = std::numeric_limits<std::size_t>::max();
        static constexpr bool fixed = false;
        std::vector<std::uint16_t> leaf_ids16;
        std::vector<std::uint32_t> leaf_ids32;
        std::vector<std::uint32_t> perm;
        std::vector<value_type>    xp_packed;
        std::vector<value_type>    out_packed;
#endif
        std::vector<std::uint32_t> counts;   // sized by n_leaves
        std::vector<std::uint32_t> offsets;  // sized by n_leaves
        void shrink_to_fit() {
            counts.shrink_to_fit();
            offsets.shrink_to_fit();
            if constexpr (!fixed) {
                // Vector path only — std::array storage cannot shrink.
#ifndef BAOBZI_FIXED_SCRATCH
                leaf_ids16.shrink_to_fit();
                leaf_ids32.shrink_to_fit();
                perm.shrink_to_fit();
                xp_packed.shrink_to_fit();
                out_packed.shrink_to_fit();
#endif
            }
        }
    };
    static Scratch &scratch() {
        thread_local Scratch s;
        return s;
    }

    /// Layer L5 (Phase 16): consecutive-small-call streak count after
    /// which `BAOBZI_RELEASE_SCRATCH_AFTER` triggers shrink_to_fit. 0
    /// (env unset / 0 / non-numeric) disables the feature — the cold
    /// path. Read once per process via static-init, then optionally
    /// overridden by a test hook (`set_release_after_for_testing`).
    /// Hot path is one load + branch on `cached + override`.
    static std::size_t release_scratch_after() {
        const std::size_t override_v = release_after_test_override();
        if (override_v != static_cast<std::size_t>(-1)) return override_v;
        static const std::size_t cached = []() -> std::size_t {
            const char *e = std::getenv("BAOBZI_RELEASE_SCRATCH_AFTER");
            if (!e || !*e) return 0;
            char *end = nullptr;
            const unsigned long long x = std::strtoull(e, &end, 10);
            return (end == e) ? 0 : static_cast<std::size_t>(x);
        }();
        return cached;
    }

    /// Test-only override for `release_scratch_after()`. -1 (sentinel)
    /// means "no override; use env cache". Setting any other value
    /// bypasses the env. Production code never touches this — it's
    /// intended for the L5 unit test, which can't reasonably spawn a
    /// fresh thread to bust the env cache (large fixed-mode TLS makes
    /// nested-thread stack overflow possible during fit()).
    static std::size_t &release_after_test_override() {
        static std::size_t v = static_cast<std::size_t>(-1);
        return v;
    }
    static void set_release_after_for_testing(std::size_t v) {
        release_after_test_override() = v;
    }

    /// Test-only probe: capacity (in elements, not bytes) of the
    /// thread_local `xp_packed` buffer for this Function instantiation.
    /// Used by the L5 unit test to assert shrink_to_fit fired. In L6
    /// fixed-scratch mode the storage is a `std::array` whose size is
    /// fixed at compile time — return it directly so the test can still
    /// query (the L5 streak still runs in fixed mode but shrink_to_fit
    /// is a no-op for the array buffers).
    static std::size_t test_xp_packed_capacity() {
        if constexpr (Scratch::fixed) {
            return scratch().xp_packed.size();
        } else {
            return scratch().xp_packed.capacity();
        }
    }

    /// Layer L4 (Phase 16): per-call tile size for the batch path. Read
    /// once from the `BAOBZI_BATCH_TILE` env var, default 65536. 0
    /// disables tiling (revert to the pre-L4 monolithic batch path —
    /// useful for diff bisection).
    static std::size_t batch_tile_size() {
        static const std::size_t v = []() -> std::size_t {
            const char *e = std::getenv("BAOBZI_BATCH_TILE");
            if (!e || !*e) return 65536;
            char *end = nullptr;
            const unsigned long long x = std::strtoull(e, &end, 10);
            if (end == e) return 65536;
            return static_cast<std::size_t>(x);
        }();
        return v;
    }

    /// Batch evaluation: n_trg points written into res.
    ///
    /// Fast path groups points by owning leaf via counting sort, then invokes
    /// polyfit's SIMD batch kernel once per leaf. Scalar per-point traversal
    /// is kept for tiny batches where the sort cannot amortize.
    ///
    /// Thread-safe: a single Function may be called concurrently from
    /// multiple threads provided each call's `xp` and `res` slices do not
    /// overlap with another thread's. Per-call scratch is `thread_local`;
    /// the Function's internal state (nodes, polyfits) is immutable after
    /// construction. Pinned by `tests/test_threadsafe.cpp`.
    inline void operator()(const value_type *xp, value_type *res, std::size_t n_trg) const {
        if (n_trg == 0) return;
        if (n_trg == 1) {
            const detail::Value<value_type, input_dim> xi(xp);
            const detail::Value<value_type, output_dim> tmp = (*this)(xi);
            std::copy(tmp.begin(), tmp.end(), res);
            maybe_release_scratch(n_trg);
            return;
        }

        // Below this point the counting-sort overhead likely exceeds the
        // SIMD gain — fall through to scalar per-point.
        constexpr std::size_t kSortThreshold = 32;
        if (n_trg < kSortThreshold) {
            for (std::size_t i_trg = 0; i_trg < n_trg; ++i_trg) {
                const detail::Value<value_type, input_dim> xi(xp + input_dim * i_trg);
                const detail::Value<value_type, output_dim> tmp = (*this)(xi);
                std::copy(tmp.begin(), tmp.end(), res + i_trg * output_dim);
            }
            maybe_release_scratch(n_trg);
            return;
        }

        // Layer L4 (Phase 16): tile the batch path so the thread_local
        // scratch resizes to `tile_K` rather than `n_trg`. Default 64 K
        // covers the per-tile working set within L1d/L2 for low-leaf-
        // count Functions (1D / 3D in the perf-driver). For high-leaf-
        // count Functions (2D bump has ~7700 leaves) a hard 64 K tile
        // gives only ~8 points per leaf per tile and the polyfit batch
        // kernel's per-call setup overhead dominates — observed -19 %
        // regression on 2D bump at K=64 K. The adaptive floor
        // `n_leaves * kMinPtsPerLeaf` keeps each tile populated with
        // enough points to amortise that setup, while the env override
        // (`BAOBZI_BATCH_TILE`) is the floor — letting the user trade
        // throughput for footprint on a per-Function basis. 0 disables
        // tiling entirely (revert to the pre-L4 monolithic batch).
        constexpr std::size_t kMinPtsPerLeaf = 32;
        const std::size_t env_K = batch_tile_size();
        // L6: when fixed-scratch is compiled in, the per-tile arrays
        // are sized for `Scratch::kMaxTileK`. Cap tile_K at that limit
        // and force tiling for large n_trg even when the user disabled
        // it via env_K=0 — otherwise eval_batch_tile would index past
        // the fixed buffer.
        const std::size_t base_K = (env_K == 0)
            ? (Scratch::fixed ? Scratch::kMaxTileK : 0)
            : std::max(env_K, polyfits_.size() * kMinPtsPerLeaf);
        const std::size_t tile_K = std::min(base_K, Scratch::kMaxTileK);
        if (tile_K != 0 && n_trg > tile_K) {
            for (std::size_t tile_off = 0; tile_off < n_trg; tile_off += tile_K) {
                const std::size_t tile_n = std::min(tile_K, n_trg - tile_off);
                eval_batch_tile(xp + input_dim * tile_off,
                                res + output_dim * tile_off, tile_n);
            }
            maybe_release_scratch(n_trg);
            return;
        }
        eval_batch_tile(xp, res, n_trg);
        maybe_release_scratch(n_trg);
    }

    /// Layer L5 (Phase 16): when `BAOBZI_RELEASE_SCRATCH_AFTER` is set
    /// to N > 0, shrink_to_fit the thread_local scratch after N
    /// consecutive `operator()` calls below `kSmallCallThreshold`.
    /// "Small" means the scratch was likely oversized for the call; a
    /// streak of small calls suggests the hot batch is over and the
    /// caller wants the capacity back. Big calls reset the streak. Off
    /// by default — `release_scratch_after()` returns 0 → entire branch
    /// folds away on first miss.
    inline void maybe_release_scratch(std::size_t n_trg) const {
        const std::size_t after_n = release_scratch_after();
        if (after_n == 0) return;
        constexpr std::size_t kSmallCallThreshold = 4096;
        thread_local std::size_t small_streak = 0;
        if (n_trg < kSmallCallThreshold) {
            if (++small_streak >= after_n) {
                small_streak = 0;
                scratch().shrink_to_fit();
            }
        } else {
            small_streak = 0;
        }
    }

    inline void eval_batch_tile(const value_type *xp, value_type *res,
                                std::size_t n_trg) const {
        constexpr std::size_t kSortThreshold = 32;
        if (n_trg < kSortThreshold) {
            for (std::size_t i_trg = 0; i_trg < n_trg; ++i_trg) {
                const detail::Value<value_type, input_dim> xi(xp + input_dim * i_trg);
                const detail::Value<value_type, output_dim> tmp = (*this)(xi);
                std::copy(tmp.begin(), tmp.end(), res + i_trg * output_dim);
            }
            return;
        }

        const std::uint32_t n_leaves = static_cast<std::uint32_t>(polyfits_.size());
        const std::uint32_t ood_id = n_leaves; // sentinel bucket for out-of-domain

        // Thread-local scratch reused across calls. Vectors amortize their
        // capacity so steady-state eval does no heap allocation.
        // Layer L1 (Phase 14): `leaf_ids` is narrowed to u16 when
        // `n_leaves <= 65535` (true for every perf-driver scenario and the
        // overwhelming majority of practical Functions). Halves L1d traffic
        // on the scatter's `leaf_ids[i]` load chain and saves 2 MiB of
        // resident scratch at N=1e6. Reads zero-extend on x86 so the
        // consuming code paths are unchanged. The u32 vector is kept for
        // the rare > 65 K leaf fallback so that path keeps the original
        // type.
        // Layer L5 (Phase 16): scratch is hoisted into the per-Function
        // `Scratch` struct accessed via `scratch()` so the L5 shrink path
        // can reach the same TLS instance.
        Scratch &sc = scratch();
        auto &leaf_ids16 = sc.leaf_ids16; // std::vector or std::array
        auto &leaf_ids32 = sc.leaf_ids32;
        auto &counts     = sc.counts;     // always vector (sized by n_leaves)
        auto &offsets    = sc.offsets;    // always vector
        auto &perm       = sc.perm;
        auto &xp_packed  = sc.xp_packed;
        auto &out_packed = sc.out_packed;

        const bool leaf_ids_fit_u16 =
            n_leaves <= std::numeric_limits<std::uint16_t>::max();

        // L6: when Scratch storage is fixed-array, the buffers are
        // already sized at compile time for kMaxTileK. operator() caps
        // tile_K at kMaxTileK so n_trg here is always within bounds —
        // the resize calls become no-ops on arrays. counts/offsets are
        // sized by n_leaves and stay heap-backed in either mode.
        if constexpr (!Scratch::fixed) {
            if (leaf_ids_fit_u16) leaf_ids16.resize(n_trg);
            else                  leaf_ids32.resize(n_trg);
            perm.resize(n_trg);
            xp_packed.resize(input_dim * n_trg);
            out_packed.resize(output_dim * n_trg);
        } else {
            assert(n_trg <= Scratch::kMaxTileK);
        }
        counts.assign(n_leaves + 1, 0);

        // Traversal: leaf id per point + population histogram.
        // Layer A: when there is a single subtree with a leaf table (the
        // common case for compact domains — all four perf-driver scenarios
        // hit this path), the subtree's quantize already covers the entire
        // Function domain, so OOD detection collapses into the same
        // unsigned-wrap test that produces the table index. The Function-
        // level OOD pre-check (2·Dim `vcomisd`) and `get_linear_bin` (which
        // returns 0 for n_subtrees==1 anyway) drop out entirely.
        auto find_loop = [&](auto &leaf_ids_vec) {
            using LeafIdT = std::remove_reference_t<decltype(leaf_ids_vec[0])>;
            if (subtrees_.size() == 1 && subtrees_.front().has_leaf_table()) {
                const auto &st = subtrees_.front();
                for (std::size_t i = 0; i < n_trg; ++i) {
                    const detail::Value<value_type, input_dim> xi(xp + input_dim * i);
                    const std::uint32_t id = st.find_leaf_id_with_ood(xi, ood_id);
                    leaf_ids_vec[i] = static_cast<LeafIdT>(id);
                    ++counts[id];
                }
            } else {
                for (std::size_t i = 0; i < n_trg; ++i) {
                    const detail::Value<value_type, input_dim> xi(xp + input_dim * i);
                    bool in_domain = true;
                    poet::static_for<input_dim>([&](auto D) {
                        constexpr std::size_t d = D;
                        if (xi[d] < lower_left_[d] || xi[d] >= upper_right_[d])
                            in_domain = false;
                    });
                    const std::uint32_t id = in_domain
                        ? subtrees_[get_linear_bin(xi)].find_leaf_id(xi)
                        : ood_id;
                    leaf_ids_vec[i] = static_cast<LeafIdT>(id);
                    ++counts[id];
                }
            }
        };
        if (leaf_ids_fit_u16) find_loop(leaf_ids16);
        else                  find_loop(leaf_ids32);

        // Prefix sum — offsets[k] is the packed-buffer start for leaf k.
        offsets.resize(n_leaves + 1);
        {
            std::uint32_t run = 0;
            for (std::uint32_t k = 0; k <= n_leaves; ++k) {
                offsets[k] = run;
                run += counts[k];
            }
        }

        // Scatter inputs to packed layout. offsets[] is consumed as a cursor;
        // rebuilt from counts afterwards for the per-leaf dispatch.
        auto scatter_loop = [&](const auto &leaf_ids_vec) {
            for (std::size_t i = 0; i < n_trg; ++i) {
                const std::uint32_t id = leaf_ids_vec[i];
                const std::uint32_t dst = offsets[id]++;
                perm[dst] = static_cast<std::uint32_t>(i);
                if constexpr (input_dim == 1) {
                    xp_packed[dst] = xp[i];
                } else {
                    const value_type *src = xp + input_dim * i;
                    value_type *dstp = xp_packed.data() + input_dim * dst;
                    poet::static_for<input_dim>([&](auto D) {
                        constexpr std::size_t d = D;
                        dstp[d] = src[d];
                    });
                }
            }
        };
        if (leaf_ids_fit_u16) scatter_loop(leaf_ids16);
        else                  scatter_loop(leaf_ids32);
        {
            std::uint32_t run = 0;
            for (std::uint32_t k = 0; k <= n_leaves; ++k) {
                offsets[k] = run;
                run += counts[k];
            }
        }

        // Per-leaf SIMD batch eval. Speculative prefetch of the next
        // non-empty leaf's coefficient store hides cacheline-fill latency
        // when leaves are small relative to the working set.
        for (std::uint32_t id = 0; id < n_leaves; ++id) {
            const std::uint32_t cnt = counts[id];
            if (cnt == 0) continue;
            const std::uint32_t off = offsets[id];
#if defined(__GNUC__) || defined(__clang__)
            std::uint32_t next_id = id + 1;
            while (next_id < n_leaves && counts[next_id] == 0) ++next_id;
            if (next_id < n_leaves) {
                if constexpr (has_tuple_size_v<input_type>) {
                    // ND: polyfit doesn't expose a coefficient pointer; the
                    // evaluator object's first cacheline contains domain
                    // params and (on default layouts) the start of coeffsFlat.
                    __builtin_prefetch(&polyfits_[next_id]);
                } else {
                    __builtin_prefetch(polyfits_[next_id].coeffs().data());
                }
            }
#endif
            if constexpr (has_tuple_size_v<input_type>) {
                using CI = typename poly_eval_type::CanonicalInput;
                using CO = typename poly_eval_type::CanonicalOutput;
                const CI *pts = reinterpret_cast<const CI *>(
                    xp_packed.data() + input_dim * off);
                CO *outs = reinterpret_cast<CO *>(
                    out_packed.data() + output_dim * off);
                polyfits_[id](pts, outs, static_cast<std::size_t>(cnt));
            } else {
                polyfits_[id](xp_packed.data() + off,
                              out_packed.data() + off,
                              static_cast<std::size_t>(cnt));
            }
        }

        // Fill OOD slots with NaN.
        const std::uint32_t ood_cnt = counts[ood_id];
        if (ood_cnt) {
            const std::uint32_t off = offsets[ood_id];
            constexpr value_type nan_v = std::numeric_limits<value_type>::quiet_NaN();
            for (std::uint32_t k = 0; k < ood_cnt; ++k)
                for (std::size_t j = 0; j < output_dim; ++j)
                    out_packed[output_dim * (off + k) + j] = nan_v;
        }

        // Permute outputs back to caller order.
        // Prefetch-for-write hides RFO latency on the random `res[perm[dst]]`
        // store: with a random permutation each cacheline costs ~50-200 c
        // RFO, which dominates 1D throughput post-Layer-A.
        constexpr std::size_t LOOKAHEAD = 32;
        for (std::size_t dst = 0; dst < n_trg; ++dst) {
#if defined(__GNUC__) || defined(__clang__)
            if (dst + LOOKAHEAD < n_trg) {
                const std::uint32_t s = perm[dst + LOOKAHEAD];
                __builtin_prefetch(res + output_dim * s, /*rw=*/1, /*locality=*/0);
            }
#endif
            const std::uint32_t src = perm[dst];
            const value_type *srcp = out_packed.data() + output_dim * dst;
            value_type *dstp = res + output_dim * src;
            poet::static_for<output_dim>([&](auto J) {
                constexpr std::size_t j = J;
                dstp[j] = srcp[j];
            });
        }
    }

    /// Legacy int overload — forwarded to the std::size_t version above.
    /// Same thread-safety contract as the std::size_t overload.
    inline void operator()(const value_type *xp, value_type *res, int n_trg) const {
        (*this)(xp, res, static_cast<std::size_t>(n_trg));
    }

    /// Point evaluation.
    inline output_type operator()(const input_type &x) const {
        if constexpr (input_dim == 1) {
            if (x < lower_left_[0] || x >= upper_right_[0])
                return output_type{NAN};
        } else {
            for (std::size_t i = 0; i < input_dim; ++i)
                if (x[i] < lower_left_[i] || x[i] >= upper_right_[i])
                    return output_type{NAN};
        }

        return polyfits_[find_node(x).poly_eval_id](x);
    }

    /// Panels where adaptive paneling failed at `max_depth`. Always empty
    /// unless `options.allow_max_depth_leaves == true` was set; the default
    /// path throws `MaxDepthExceeded` (which carries the same list) instead.
    inline const std::vector<NonConvergedPanel> &non_converged_panels() const {
        return non_converged_panels_;
    }

    inline std::pair<dim_array_t, dim_array_t> get_bounds() const {
        return std::make_pair(lower_left_, upper_right_);
    }

  private:
    detail::TreeInput input_;
    box_t box_;
    value_type tol_;
    dim_array_t lower_left_{};
    dim_array_t upper_right_{};

    std::vector<detail::PolyTree<Degree, Func>> subtrees_;
    detail::Value<std::size_t, input_dim> n_subtrees_{};
    dim_array_t inv_bin_size_{};

    std::vector<poly_eval_type> polyfits_;

    std::vector<NonConvergedPanel> non_converged_panels_;

    /// Structure containing info about self creation.
    struct {
        std::size_t base_depth = 0;
        std::uint64_t n_evals_root = 0;
        std::uint32_t t_elapsed = 0;
    } stats_;
};

} // namespace baobzi

#endif // BAOBZI_DETAIL_FUNCTION_IMPL_HPP
