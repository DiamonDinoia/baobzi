#ifndef BAOBZI_DETAIL_NODE_HPP
#define BAOBZI_DETAIL_NODE_HPP

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <polyfit/polyeval.hpp>

#include <baobzi/detail/numerics.hpp>
#include <baobzi/detail/tol_kind.hpp>
#include <baobzi/detail/value.hpp>

namespace baobzi::detail {

/// 8-byte tree node: child link + leaf-id slot. The `center` is recomputed
/// during descent from the subtree's `(lo, hi)` bounds, so the node carries
/// no per-axis geometry. `first_child_idx == kLeafSentinel` marks a leaf;
/// `poly_eval_id` indexes the per-Function polyfits table and is meaningful
/// only on leaves. 8 nodes per cache line in every dim — descent loads few
/// cachelines.
template <class Func, std::size_t Degree>
class Node {
  public:
    using input_type = std::remove_cvref_t<poly_eval::fitInput_t<Func>>;
    using output_type = poly_eval::fitOutput_t<Func>;
    using value_type = poly_eval::detail::value_type_or_t<input_type>;
    using poly_eval_type = std::conditional_t<poly_eval::detail::hasTupleSize_v<input_type>, poly_eval::FuncEvalND<Func, Degree, poly_eval::FusionMode::Never>,
                                  poly_eval::FuncEval<Func, Degree, 1, poly_eval::FusionMode::Never>>;

    static constexpr std::size_t input_dim = value_dim_v<input_type>;
    static constexpr std::size_t output_dim = value_dim_v<output_type>;

    static constexpr std::uint32_t kLeafSentinel = std::numeric_limits<std::uint32_t>::max();

    std::uint32_t first_child_idx = kLeafSentinel;
    std::uint32_t poly_eval_id    = kLeafSentinel;

    Node() = default;

    [[nodiscard]] bool is_leaf() const { return first_child_idx == kLeafSentinel; }

    /// Fit this node to the requested tolerance. On success, stores the
    /// poly_eval_id into polyfits and returns true. Center/half_length are
    /// passed in (the runtime Node no longer carries `center`).
    bool fit(const TreeInput &input, const Func &func,
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
            if (tail_error_below_tolerance(input.tol_kind, input.tol, polyfit))
                return rollback_and_fail();
        } else {
            if (sample_error_below_tolerance(kFitSamplesPerDim, input.tol_kind, input.tol,
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

    [[nodiscard]] std::size_t memory_usage() const { return sizeof(*this); }
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

} // namespace baobzi::detail

#endif // BAOBZI_DETAIL_NODE_HPP
