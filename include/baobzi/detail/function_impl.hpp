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

#include <baobzi/detail/compiler_macros.hpp>

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
/// Internal fit-time configuration; mirror of `baobzi::options` plus the
/// shape parameters resolved at the public API boundary.
struct TreeInput {
    int     input_dim              = 0;
    int     output_dim             = 1;
    int     degree                 = 8;
    double  tol                    = 0.0;
    int     max_depth              = 50;
    int     max_memory_mib         = 64;
    bool    allow_max_depth_leaves = false;
    TolKind tol_kind               = TolKind::RelativeMax;
};

/// Sample-grid resolution per axis used by sample-based tolerance kinds
/// (`RelativeMax`, `AbsoluteMax`, `RelativeL2`, `AbsoluteL2`). 8 samples
/// per axis is dense enough to expose ringing from a degree-8 fit while
/// keeping ND fit cost bounded (8^Dim per panel).
inline constexpr int kFitSamplesPerDim = 8;
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

    Value operator+(const Value &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ + rhs.data_));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] + rhs.data_[i];
            return Value(result);
        }
    }

    Value operator+(const T &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ + rhs));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] + rhs;
            return Value(result);
        }
    }

    Value operator-(const Value &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ - rhs.data_));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] - rhs.data_[i];
            return Value(result);
        }
    }

    Value operator-(const T &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ - rhs));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] - rhs;
            return Value(result);
        }
    }

    Value operator*(const Value &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ * rhs.data_));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] * rhs.data_[i];
            return Value(result);
        }
    }

    Value operator*(const T &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ * rhs));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] * rhs;
            return Value(result);
        }
    }

    Value operator/(const Value &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ / rhs.data_));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] / rhs.data_[i];
            return Value(result);
        }
    }

    Value operator/(const T &rhs) const {
        if constexpr (N == 1) {
            return Value(static_cast<T>(data_ / rhs));
        } else {
            std::array<T, N> result{};
            for (std::size_t i = 0; i < N; ++i)
                result[i] = data_[i] / rhs;
            return Value(result);
        }
    }

    constexpr T &operator[](std::size_t idx) {
        if constexpr (N == 1) {
            static_cast<void>(idx);
            return data_;
        } else {
            return data_[idx];
        }
    }

    [[nodiscard]] constexpr const T &operator[](std::size_t idx) const {
        if constexpr (N == 1) {
            static_cast<void>(idx);
            return data_;
        } else {
            return data_[idx];
        }
    }

    constexpr T *begin() {
        if constexpr (N == 1) {
            return &data_;
        } else {
            return data_.data();
        }
    }

    constexpr T *end() {
        if constexpr (N == 1) {
            return &data_ + 1;
        } else {
            return data_.data() + N;
        }
    }

    [[nodiscard]] constexpr const T *begin() const {
        if constexpr (N == 1) {
            return &data_;
        } else {
            return data_.data();
        }
    }

    [[nodiscard]] constexpr const T *end() const {
        if constexpr (N == 1) {
            return &data_ + 1;
        } else {
            return data_.data() + N;
        }
    }

    T prod() const {
        if constexpr (N == 1) {
            return data_;
        } else {
            T result = T{1};
            for (const auto &val : data_)
                result *= val;
            return result;
        }
    }

    storage_t get() const { return data_; }

    // Automatic casting to scalar or array
    operator T() const {
        static_assert(N == 1, "Can only cast to scalar if N == 1");
        return data_;
    }

    operator std::array<T, N>() const {
        static_assert(N != 1, "Can only cast to array if N != 1");
        return data_;
    }

    // Access underlying data
    const T &scalar() const {
        static_assert(N == 1, "Not a scalar");
        return data_;
    }
    const std::array<T, N> &array() const {
        static_assert(N != 1, "Not an array");
        return data_;
    }

    std::array<T, N> as_array() const {
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
    if constexpr (poly_eval::detail::hasTupleSize_v<T>)
        return std::tuple_size_v<T>;
    else
        return 1;
}

/// Geometric portion of Baobzi nodes.
template <typename T, std::size_t Dim>
struct Box {
    Value<T, Dim> center;
    Value<T, Dim> half_length;

    Box(const auto &x, const auto &hl) : center{x}, half_length{hl} {}
};

/// Check whether the leading-coefficient 'tail estimate' for a given fit
/// exceeds the requested tolerance.
template <class Polyfit>
inline bool tail_error_check(TolKind tol_type, double tol, const Polyfit &polyfit) {
    constexpr std::size_t input_dim = get_tuple_size<typename Polyfit::InputType>();
    constexpr std::size_t output_dim = get_tuple_size<typename Polyfit::OutputType>();
    using T = poly_eval::detail::value_type_or_t<typename Polyfit::InputType>;
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
    using input_type = std::remove_cvref_t<typename poly_eval::fitInput_t<Func>>;
    using output_type = typename poly_eval::fitOutput_t<Func>;
    using value_type = poly_eval::detail::value_type_or_t<input_type>;
    using poly_eval_type =
        typename std::conditional<poly_eval::detail::hasTupleSize_v<input_type>, poly_eval::FuncEvalND<Func, Degree, poly_eval::FusionMode::Never>,
                                  poly_eval::FuncEval<Func, Degree, 1, poly_eval::FusionMode::Never>>::type;

    static constexpr std::size_t input_dim = get_tuple_size<input_type>();
    static constexpr std::size_t output_dim = get_tuple_size<output_type>();

    static constexpr std::uint32_t kLeafSentinel = std::numeric_limits<std::uint32_t>::max();

    // 8-byte node: `center` is recomputed during descent from the
    // subtree's (lo, hi), and `poly_eval_id` indexes the per-Function
    // polyfits table. `first_child_idx == kLeafSentinel` marks a leaf;
    // `poly_eval_id` is meaningful only when leaf. 8 nodes per cache
    // line in every Dim — descent loads few cachelines.
    std::uint32_t first_child_idx = kLeafSentinel;
    std::uint32_t poly_eval_id    = kLeafSentinel;

    Node() = default;

    bool is_leaf() const { return first_child_idx == kLeafSentinel; }

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
            if (sample_error_check(detail::kFitSamplesPerDim, input.tol_kind, input.tol,
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

    std::size_t memory_usage() const { return sizeof(*this); }
};

// Lock the 8-B node invariant. If this fires, an extra field was added
// or alignment regressed — descent load-volume depends on 8 nodes per
// cache line.
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
    using input_type = std::remove_cvref_t<typename poly_eval::fitInput_t<Func>>;
    using value_type = poly_eval::detail::value_type_or_t<input_type>;
    using poly_eval_type =
        typename std::conditional<poly_eval::detail::hasTupleSize_v<input_type>, poly_eval::FuncEvalND<Func, Degree, poly_eval::FusionMode::Never>,
                                  poly_eval::FuncEval<Func, Degree, 1, poly_eval::FusionMode::Never>>::type;
    using output_type = typename poly_eval::fitOutput_t<Func>;

    static constexpr std::size_t output_dim = get_tuple_size<output_type>();
    static constexpr std::size_t input_dim = get_tuple_size<input_type>();
    static constexpr std::size_t n_child = std::size_t{1} << input_dim;

    using node_t = Node<Func, Degree>;
    using box_t = Box<value_type, input_dim>;
    using dim_array_t = Value<value_type, input_dim>;

    PolyTree(const detail::TreeInput &input, const Box<value_type, input_dim> &root_box,
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

        // For shallow subtrees, build a quantize-to-leaf table so
        // eval-time descent collapses to a single load. Table is
        // uint32, 4 B per entry, size 1 << (input_dim * max_depth_).
        // Capped at 16 K entries (64 KiB) per subtree to stay in L1d
        // for the common compact-domain shapes.
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

    const node_t &find_node(const input_type &x) const { return nodes_[get_node_index(x)]; }

    /// Combined leaf-id lookup: table if available, else descent.
    std::uint32_t find_leaf_id(const input_type &x) const {
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

    [[nodiscard]] constexpr bool has_leaf_table() const noexcept { return !leaf_table_.empty(); }

    /// Leaf-id lookup that returns `ood_id` for points outside this
    /// subtree's domain. Caller must have verified `has_leaf_table()`.
    ///
    /// The signed `vcvttsd2si` + unsigned compare folds the OOD test
    /// into the table-index quantize: an out-of-range double yields
    /// INT64_MIN (x86-64 indefinite-integer), which compares above
    /// `mask` as uint64. Saves one cmp per axis vs. an explicit
    /// in-domain pre-check.
    ///
    /// Precondition: single-subtree Function (this subtree's `lower_`
    /// matches the Function's `lower_left_`). Multi-subtree callers
    /// keep the Function-level OOD check + `get_linear_bin` path.
    [[nodiscard]] BAOBZI_ALWAYS_INLINE std::uint32_t
    find_leaf_id_with_ood(const input_type &x, std::uint32_t ood_id) const {
        const std::size_t bits = leaf_table_depth_;
        const std::size_t mask = (std::size_t{1} << bits) - 1;
        std::size_t idx = 0;
        if constexpr (input_dim == 1) {
            const std::int64_t q0 = static_cast<std::int64_t>(
                (x - lower_[0]) * inv_span_bins_[0]);
            if (static_cast<std::uint64_t>(q0) > mask) [[unlikely]] return ood_id;
            idx = static_cast<std::size_t>(q0);
        } else {
            for (std::size_t d = 0; d < input_dim; ++d) {
                const std::int64_t qd = static_cast<std::int64_t>(
                    (x[d] - lower_[d]) * inv_span_bins_[d]);
                if (static_cast<std::uint64_t>(qd) > mask) [[unlikely]] return ood_id;
                idx |= static_cast<std::size_t>(qd) << (bits * d);
            }
        }
        return leaf_table_[idx];
    }

    /// Descent hot loop. The per-subtree (lo, hi) bounds live in
    /// registers and `mid = 0.5 * (lo + hi)` is recomputed each level
    /// — the node carries no `center`, so descent is not load-bound.
    /// `input_dim` is constexpr so the ND per-axis compare unrolls.
    ///
    /// `mid` is computed differently from the fit-time
    /// `box.center` (chained halving), so bit-exactness at boundary
    /// points is not guaranteed; tests assert relative tolerance.
    std::size_t get_node_index(const input_type &x) const {
        dim_array_t lo = lower_;
        dim_array_t hi = upper_;
        index_t curr_index = 0;
        while (!nodes_[curr_index].is_leaf()) {
            index_t child_idx = 0;
            if constexpr (poly_eval::detail::hasTupleSize_v<input_type>) {
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

    [[nodiscard]] constexpr std::size_t size() const { return nodes_.size(); }
    [[nodiscard]] constexpr std::size_t max_depth() const { return max_depth_; }

    std::size_t memory_usage() const {
        std::size_t total = sizeof(*this);
        for (const auto &node : nodes_)
            total += node.memory_usage();
        return total;
    }

    auto &get_nodes() { return nodes_; }
    auto &get_nodes() const { return nodes_; }

    const std::vector<NonConvergedPanel> &non_converged_panels() const {
        return non_converged_panels_;
    }

  private:
    std::vector<node_t> nodes_;
    // Subtree bounding box, carried into descent so the Node carries
    // no per-axis `center` field.
    dim_array_t lower_{};
    dim_array_t upper_{};
    std::size_t max_depth_ = 0;
    // Quantize→leaf table for shallow subtrees; empty when not built.
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
    using input_type = std::remove_cvref_t<typename poly_eval::fitInput_t<Func>>;
    using output_type = typename poly_eval::fitOutput_t<Func>;
    using value_type = poly_eval::detail::value_type_or_t<input_type>;
    using poly_eval_type =
        typename std::conditional<poly_eval::detail::hasTupleSize_v<input_type>, poly_eval::FuncEvalND<Func, Degree, poly_eval::FusionMode::Never>,
                                  poly_eval::FuncEval<Func, Degree, 1, poly_eval::FusionMode::Never>>::type;
    static constexpr auto degree = Degree;

    static constexpr std::size_t input_dim = detail::get_tuple_size<input_type>();
    static constexpr std::size_t output_dim = detail::get_tuple_size<output_type>();
    static constexpr std::size_t n_child = std::size_t{1} << input_dim;

    using node_t = detail::Node<Func, Degree>;
    using box_t = detail::Box<value_type, input_dim>;
    using dim_array_t = detail::Value<value_type, input_dim>;

    [[nodiscard]] std::size_t memory_usage() const {
        std::size_t mem = sizeof(*this);
        mem += polyfits_.capacity() * sizeof(poly_eval_type);
        for (const auto &subtree : subtrees_)
            mem += subtree.memory_usage();
        return mem;
    }

    void print_stats() const {
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
    Function(const detail::TreeInput &input, const input_type center, const input_type half_width_in,
                    const Func &func)
        : input_(input),
          box_(dim_array_t{center}, dim_array_t{half_width_in}),
          tol_(input.tol) {
        const auto t_start = std::chrono::steady_clock::now();

        dim_array_t lvec{half_width_in};
        std::queue<box_t> q;

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
            for (std::size_t i = 0; i < n_next; ++i) {
                box_t current_box = q.front();
                q.pop();

                nodes.emplace_back();
                auto &node = nodes.back();
                std::vector<poly_eval_type> dummy;
                node.fit(input, func, current_box.center, current_box.half_length, {}, dummy);
                if (node.poly_eval_id != 0u)
                    node.poly_eval_id = 0;

                if (!node.is_leaf())
                    add_node_children_to_queue(q, current_box.center, half_width);
            }
            stats_.n_evals_root += static_cast<std::uint64_t>(
                nodes.size() *
                static_cast<std::size_t>(std::pow(static_cast<double>(Degree), static_cast<double>(input_dim))));

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

    void build_cache() {}

    /// Convert linear bin index to [dim] bin vector.
    std::array<std::size_t, input_dim> get_bins(const std::size_t i_bin) const {
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
    [[nodiscard]] BAOBZI_ALWAYS_INLINE std::size_t get_linear_bin(const input_type &x) const {
        if constexpr (input_dim == 1) {
            const value_type x_bin = [this, &x]() {
                if constexpr (poly_eval::detail::hasTupleSize_v<input_type>)
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

    [[nodiscard]] const node_t &find_node(const input_type &x) const { return subtrees_[get_linear_bin(x)].find_node(x); }

    /// Per-thread, per-Function-instantiation scratch for the batch
    /// path. Vectors grow on first use and reuse capacity thereafter,
    /// so steady-state eval does no heap allocation.
    struct Scratch {
        std::vector<std::uint16_t> leaf_ids16;
        std::vector<std::uint32_t> leaf_ids32;
        std::vector<std::uint32_t> perm;
        std::vector<value_type>    xp_packed;
        std::vector<value_type>    out_packed;
        std::vector<std::uint32_t> counts;   // sized by n_leaves
        std::vector<std::uint32_t> offsets;  // sized by n_leaves
    };
    static Scratch &scratch() {
        thread_local Scratch s;
        return s;
    }

    /// Default per-call tile cap for the batch path. The adaptive floor
    /// (`n_leaves * kMinPtsPerLeaf`) raises this when low-leaf-count
    /// Functions would otherwise starve the polyfit batch kernel.
    static constexpr std::size_t kDefaultTileK = 65536;

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
    BAOBZI_FLATTEN void operator()(const value_type *xp, value_type *res, std::size_t n_trg) const {
        if (n_trg == 0) [[unlikely]] return;
        if (n_trg == 1) [[unlikely]] {
            const detail::Value<value_type, input_dim> xi(xp);
            const detail::Value<value_type, output_dim> tmp = (*this)(xi);
            std::copy(tmp.begin(), tmp.end(), res);
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
            return;
        }

        // Tile the batch path so the per-tile working set fits L1d/L2.
        // The adaptive floor `n_leaves * kMinPtsPerLeaf` keeps each tile
        // populated enough to amortise the polyfit batch kernel's
        // per-call setup — high-leaf-count Functions (e.g. 2D bump,
        // ~7700 leaves) regress sharply at a hard 64 K tile.
        constexpr std::size_t kMinPtsPerLeaf = 32;
        const std::size_t tile_K =
            std::max(kDefaultTileK, polyfits_.size() * kMinPtsPerLeaf);
        if (n_trg > tile_K) {
            for (std::size_t tile_off = 0; tile_off < n_trg; tile_off += tile_K) {
                const std::size_t tile_n = std::min(tile_K, n_trg - tile_off);
                eval_batch_tile(xp + input_dim * tile_off,
                                res + output_dim * tile_off, tile_n);
            }
            return;
        }
        eval_batch_tile(xp, res, n_trg);
    }

    /// Per-tile counting-sort + per-leaf SIMD batch eval. The caller
    /// (`operator()`) ensures `n_trg >= kSortThreshold` and
    /// `n_trg <= tile_K`, so this path always runs the sort.
    void eval_batch_tile(const value_type *xp, value_type *res,
                         std::size_t n_trg) const {
        const std::uint32_t n_leaves = static_cast<std::uint32_t>(polyfits_.size());
        const std::uint32_t ood_id = n_leaves; // sentinel bucket for out-of-domain

        // Thread-local scratch reused across calls. `leaf_ids` is
        // narrowed to u16 when `n_leaves <= 65535` (the common case);
        // this halves L1d traffic on the scatter's load chain and saves
        // ~2 MiB of resident scratch at N=1e6. Reads zero-extend on
        // x86, so consumer code is unchanged. The u32 vector remains
        // for the rare > 65 K leaf fallback.
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

        if (leaf_ids_fit_u16) leaf_ids16.resize(n_trg);
        else                  leaf_ids32.resize(n_trg);
        perm.resize(n_trg);
        xp_packed.resize(input_dim * n_trg);
        out_packed.resize(output_dim * n_trg);
        counts.assign(n_leaves + 1, 0);

        // Traversal: leaf id per point + population histogram. With a
        // single subtree owning a leaf table (the common compact-domain
        // case), the subtree's quantize already covers the whole
        // Function domain — OOD detection collapses into the same
        // unsigned-wrap test that produces the table index, dropping
        // the Function-level OOD pre-check and get_linear_bin entirely.
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
                if constexpr (poly_eval::detail::hasTupleSize_v<input_type>) {
                    // ND: polyfit doesn't expose a coefficient pointer; the
                    // evaluator object's first cacheline contains domain
                    // params and (on default layouts) the start of coeffsFlat.
                    __builtin_prefetch(&polyfits_[next_id]);
                } else {
                    __builtin_prefetch(polyfits_[next_id].coeffs().data());
                }
            }
#endif
            if constexpr (poly_eval::detail::hasTupleSize_v<input_type>) {
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

    /// Point evaluation.
    [[nodiscard]] output_type operator()(const input_type &x) const {
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
    [[nodiscard]] const std::vector<NonConvergedPanel> &non_converged_panels() const {
        return non_converged_panels_;
    }

    [[nodiscard]] std::pair<dim_array_t, dim_array_t> get_bounds() const {
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
