#ifndef BAOBZI_DETAIL_FUNCTION_IMPL_HPP
#define BAOBZI_DETAIL_FUNCTION_IMPL_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <vector>

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
    TolKind tol_kind              = TolKind::RelativeMax;
};
} // namespace detail

class MaxDepthExceeded : public std::exception {
  public:
    const char *what() const noexcept override {
        return "Baobzi fit error: tree depth exceeded max allowed input depth";
    }
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
            std::copy(arr, arr + N, data_.begin());
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
        typename std::conditional<has_tuple_size_v<input_type>, poly_eval::FuncEvalND<Func, Degree>,
                                  poly_eval::FuncEval<Func, Degree, 1, poly_eval::FusionMode::Auto>>::type;

    static constexpr std::size_t input_dim = get_tuple_size<input_type>();
    static constexpr std::size_t output_dim = get_tuple_size<output_type>();

    Value<value_type, input_dim> center;
    std::uint64_t poly_eval_id = std::numeric_limits<std::uint64_t>::max();
    std::uint32_t first_child_idx = std::numeric_limits<std::uint32_t>::max();

    Node(const Box<value_type, input_dim> &box) : center{box.center} {}

    inline bool is_leaf() const {
        return poly_eval_id != std::numeric_limits<std::uint64_t>::max();
    }

    /// Fit this node to the requested tolerance. On success, stores the
    /// poly_eval_id into polyfits and returns true.
    bool fit(const detail::TreeInput &input, const Func &func, const Value<value_type, input_dim> &half_length,
             const std::vector<value_type> &samples, std::vector<poly_eval_type> &polyfits) {
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

        poly_eval_id = n_polyfit_before;
        return true;
    }

    inline std::size_t memory_usage() const { return sizeof(*this); }
};

/// Represent a function over some domain as a tree of Chebyshev nodes.
template <std::size_t Degree, class Func>
struct PolyTree {
    using input_type = std::remove_cvref_t<typename poly_eval::function_traits<Func>::arg0_type>;
    using value_type = typename value_type_or_identity<input_type>::type;
    using poly_eval_type =
        typename std::conditional<has_tuple_size_v<input_type>, poly_eval::FuncEvalND<Func, Degree>,
                                  poly_eval::FuncEval<Func, Degree, 1, poly_eval::FusionMode::Auto>>::type;
    using output_type = typename poly_eval::function_traits<Func>::result_type;

    static constexpr std::size_t output_dim = get_tuple_size<output_type>();
    static constexpr std::size_t input_dim = get_tuple_size<input_type>();
    static constexpr std::size_t n_child = std::size_t{1} << input_dim;

    using node_t = Node<Func, Degree>;
    using box_t = Box<value_type, input_dim>;
    using dim_array_t = Value<value_type, input_dim>;

    inline PolyTree(const detail::TreeInput &input, const Box<value_type, input_dim> &root_box,
                    std::vector<poly_eval_type> &polyfits, const Func &func) {
        std::queue<box_t> q;
        dim_array_t half_width = root_box.half_length * value_type{0.5};
        q.push(root_box);

        index_t curr_child_idx = 1;
        max_depth_ = 0;
        while (!q.empty()) {
            const std::size_t n_next = q.size();
            const std::size_t node_index = nodes_.size();
            for (std::size_t i = 0; i < n_next; ++i) {
                box_t current_box = q.front();
                q.pop();

                nodes_.emplace_back(current_box);

                auto &node = nodes_[i + node_index];
                const bool successful_fit = node.fit(input, func, current_box.half_length, {}, polyfits);

                if (successful_fit) {
                    assert(polyfits.size() > 0);
                    assert(node.poly_eval_id == polyfits.size() - 1);
                } else {
                    node.first_child_idx = static_cast<std::uint32_t>(curr_child_idx);
                    curr_child_idx += n_child;

                    const dim_array_t &node_center = node.center;
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

            if (!q.empty())
                ++max_depth_;
            if (max_depth_ > static_cast<std::size_t>(input.max_depth))
                throw MaxDepthExceeded();

            half_width = half_width * value_type{0.5};
        }
    }

    inline const node_t &find_node(const input_type &x) const { return nodes_[get_node_index(x)]; }

    inline std::size_t get_node_index(const input_type &x) const {
        index_t curr_index = 0;
        while (!nodes_[curr_index].is_leaf()) {
            index_t child_idx = 0;

            if constexpr (has_tuple_size_v<input_type>)
                for (std::size_t i = 0; i < input_dim; ++i)
                    child_idx = child_idx |
                                (static_cast<index_t>(x[i] > nodes_[curr_index].center[i]) << i);
            else
                child_idx = static_cast<index_t>(x > nodes_[curr_index].center[0]);

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

  private:
    std::vector<node_t> nodes_;
    std::size_t max_depth_ = 0;
};
} // namespace detail

/// Represents a function over some domain as a grid of baobzi::detail::PolyTree
/// objects.
template <std::size_t Degree, class Func, bool SplitMultiEval>
class Function {
  public:
    using input_type = std::remove_cvref_t<typename poly_eval::function_traits<Func>::arg0_type>;
    using output_type = typename poly_eval::function_traits<Func>::result_type;
    using value_type = typename value_type_or_identity<input_type>::type;
    using poly_eval_type =
        typename std::conditional<has_tuple_size_v<input_type>, poly_eval::FuncEvalND<Func, Degree>,
                                  poly_eval::FuncEval<Func, Degree, 1, poly_eval::FusionMode::Auto>>::type;
    static constexpr auto degree = Degree;

    static constexpr std::size_t input_dim = detail::get_tuple_size<input_type>();
    static constexpr std::size_t output_dim = detail::get_tuple_size<output_type>();
    static constexpr std::size_t n_child = std::size_t{1} << input_dim;

    using node_t = detail::Node<Func, Degree>;
    using box_t = detail::Box<value_type, input_dim>;
    using dim_array_t = detail::Value<value_type, input_dim>;

    inline std::size_t memory_usage() const {
        std::size_t mem = sizeof(*this);
        mem += subtree_node_offsets_.capacity() * sizeof(typename decltype(subtree_node_offsets_)::value_type);
        mem += node_pointers_.capacity() * sizeof(node_t *);
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

                nodes.emplace_back(node_t(current_box));
                auto &node = nodes.back();
                std::vector<poly_eval_type> dummy;
                node.fit(input, func, current_box.half_length, {}, dummy);
                if (node.poly_eval_id != 0u)
                    node.poly_eval_id = 0;

                if (!node.is_leaf() || stats_.base_depth < static_cast<std::size_t>(input.min_depth)) {
                    add_node_children_to_queue(q, node.center, half_width);
                } else {
                    leaf_fraction += value_type{1.0};
                    add_node_children_to_queue(maybe_q, node.center, half_width);
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
                if (stats_.base_depth > static_cast<std::size_t>(input.max_depth))
                    throw MaxDepthExceeded();
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

        const auto t_end = std::chrono::steady_clock::now();
        stats_.t_elapsed = static_cast<std::uint32_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count());
        build_cache();
    }

    inline void build_cache() {
        subtree_node_offsets_.resize(n_subtrees_.prod());
        subtree_node_offsets_[0] = 0;
        for (std::size_t i = 1; i < subtree_node_offsets_.size(); ++i)
            subtree_node_offsets_[i] = subtree_node_offsets_[i - 1] + subtrees_[i - 1].size();

        const auto n_nodes_tot = std::accumulate(
            subtrees_.begin(), subtrees_.end(), std::size_t{0},
            [](std::size_t prior, const auto &subtree) { return prior + subtree.size(); });

        node_pointers_.resize(n_nodes_tot);

        std::size_t i = 0;
        for (auto &subtree : subtrees_)
            for (auto &node : subtree.get_nodes())
                node_pointers_[i++] = &node;
    }

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

    inline std::size_t get_global_node_index(const input_type &x) const {
        const std::size_t i_sub = get_linear_bin(x);
        return subtree_node_offsets_[i_sub] + subtrees_[i_sub].get_node_index(x);
    }

    /// Batch evaluation: n_trg points written into res.
    inline void operator()(const value_type *xp, value_type *res, std::size_t n_trg) const {
        if constexpr (SplitMultiEval) {
            if (n_trg > 1) {
                std::vector<std::pair<node_t *, input_type>> node_map(n_trg);
                for (std::size_t i = 0; i < n_trg; ++i) {
                    const detail::Value<value_type, input_dim> xi(xp + input_dim * i);
                    node_t *node_ptr = [this, xi]() -> node_t * {
                        for (std::size_t dim = 0; dim < input_dim; ++dim)
                            if (xi[dim] < lower_left_[dim] || xi[dim] >= upper_right_[dim])
                                return nullptr;

                        return node_pointers_[get_global_node_index(xi)];
                    }();

                    node_map[i] = std::make_pair(node_ptr, xi);
                }

                for (std::size_t i_trg = 0; i_trg < n_trg; ++i_trg) {
                    const detail::Value<value_type, output_dim> tmp =
                        node_map[i_trg].first == nullptr
                            ? output_type{NAN}
                            : polyfits_[node_map[i_trg].first->poly_eval_id](node_map[i_trg].second);
                    std::copy(tmp.begin(), tmp.end(), res + i_trg * output_dim);
                }
                return;
            }
        }
        for (std::size_t i_trg = 0; i_trg < n_trg; ++i_trg) {
            const detail::Value<value_type, input_dim> xi(xp + input_dim * i_trg);
            const detail::Value<value_type, output_dim> tmp = (*this)(xi);
            std::copy(tmp.begin(), tmp.end(), res + i_trg * output_dim);
        }
    }

    /// Legacy int overload — forwarded to the std::size_t version above.
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
    std::vector<std::size_t> subtree_node_offsets_;
    std::vector<node_t *> node_pointers_;
    dim_array_t inv_bin_size_{};

    std::vector<poly_eval_type> polyfits_;

    /// Structure containing info about self creation.
    struct {
        std::size_t base_depth = 0;
        std::uint64_t n_evals_root = 0;
        std::uint32_t t_elapsed = 0;
    } stats_;
};

} // namespace baobzi

#endif // BAOBZI_DETAIL_FUNCTION_IMPL_HPP
