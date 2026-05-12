#ifndef BAOBZI_DETAIL_POLYTREE_HPP
#define BAOBZI_DETAIL_POLYTREE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <queue>
#include <type_traits>
#include <vector>

#include <poet/poet.hpp>
#include <xsimd/xsimd.hpp>
#include <polyfit/polyfit.hpp>

#include <baobzi/detail/compiler_macros.hpp>
#include <baobzi/detail/errors.hpp>
#include <baobzi/detail/node.hpp>
#include <baobzi/detail/numerics.hpp>
#include <baobzi/detail/tol_kind.hpp>
#include <baobzi/detail/value.hpp>

namespace baobzi::detail {

/// One BFS-built subtree: a flat node array plus an optional
/// quantize→leaf table for shallow trees. The eval-time hot path
/// (`get_node_index`, `find_leaf_id_with_ood`) lives here.
template <std::size_t Degree, class Func>
struct PolyTree {
    using input_type = std::remove_cvref_t<poly_eval::fitInput_t<Func>>;
    using value_type = poly_eval::detail::value_type_or_t<input_type>;
    using poly_eval_type = std::conditional_t<
        poly_eval::detail::hasTupleSize_v<input_type>,
        poly_eval::FuncEvalND<Func, Degree, poly_eval::FusionMode::Never, poly_eval::ScalarKernel::Hybrid>,
        poly_eval::FuncEval<Func, Degree, 1, poly_eval::FusionMode::Never, poly_eval::ScalarKernel::Hybrid>>;
    using output_type = poly_eval::fitOutput_t<Func>;

    static constexpr std::size_t output_dim = value_dim_v<output_type>;
    static constexpr std::size_t input_dim = value_dim_v<input_type>;
    static constexpr std::size_t n_child = std::size_t{1} << input_dim;

    using node_t = Node<Func, Degree>;
    using box_t = Box<value_type, input_dim>;
    using dim_array_t = Value<value_type, input_dim>;

    PolyTree(const TreeInput &input, const Box<value_type, input_dim> &root_box,
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
                    assert(node.poly_eval_id() == polyfits.size() - 1);
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
                    node.set_first_child_idx(static_cast<std::uint32_t>(curr_child_idx));
                    curr_child_idx += n_child;

                    const dim_array_t &node_center = current_box.center;
                    for (index_t child = 0; child < n_child; ++child) {
                        dim_array_t center_offset;

                        // Extract sign of each offset component from the bits of child.
                        for (std::size_t j = 0; j < input_dim; ++j) {
                            const std::array<value_type, 2> signed_hw{-half_width[j], half_width[j]};
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
        // Capped at 64 K entries (256 KiB) per subtree — L2-resident
        // on modern x86 cores (SPR has 2 MiB L2/core, Zen4 1 MiB) and
        // worth the L1d eviction tradeoff because the descent path
        // it replaces costs one `vucomisd + ja` per level (IPC ~1.9,
        // branch-miss 6-10%) vs one `vcvttsd2usi + load` for the
        // table (measured ~3x worse at depth 15-16 on bench_pack_scatter).
        constexpr std::size_t kTableMaxEntries = std::size_t{1} << 16; // 256 KiB / 4 B
        if (max_depth_ > 0) {
            const std::size_t total_bits = input_dim * max_depth_;
            if (total_bits <= 16) {
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
                            leaf_table_[i] = nodes_[get_node_index(xc)].poly_eval_id();
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
                            leaf_table_[i] = nodes_[get_node_index(xc)].poly_eval_id();
                        }
                    }
                }
            }
        }
    }

    auto find_node(const input_type &x) const -> const node_t & { return nodes_[get_node_index(x)]; }

    /// Combined leaf-id lookup: table if available, else descent.
    [[nodiscard]] auto find_leaf_id(const input_type &x) const -> std::uint32_t {
        if (!leaf_table_.empty()) {
            const std::size_t bits = leaf_table_depth_;
            const std::size_t mask = (std::size_t{1} << bits) - 1;
            std::size_t idx = 0;
            poet::static_for<input_dim>([&](auto D) -> void {
                constexpr std::size_t d = D;
                const value_type xd = [&]() -> value_type {
                    if constexpr (poly_eval::detail::hasTupleSize_v<input_type>) return x[d];
                    else return x;
                }();
                auto q = static_cast<std::size_t>((xd - lower_[d]) * inv_span_bins_[d]);
                if (q > mask) q = mask;
                idx |= q << (bits * d);
            });
            return leaf_table_[idx];
        }
        return nodes_[get_node_index(x)].poly_eval_id();
    }

    [[nodiscard]] constexpr auto has_leaf_table() const noexcept -> bool { return !leaf_table_.empty(); }

    /// Leaf-id lookup that returns `ood_id` for points outside this
    /// subtree's domain. Caller must have verified `has_leaf_table()`.
    ///
    /// The signed `vcvttsd2si` + unsigned compare folds the OOD test
    /// into the table-index quantize: an out-of-range double yields
    /// INT64_MIN (x86-64 indefinite-integer), which compares above
    /// `mask` as uint64. Saves one cmp per axis vs. an explicit
    /// in-domain pre-check. The early `return ood_id;` inside the
    /// per-axis loop keeps the OOD path off the hot fall-through —
    /// flag-and-post-check formulations regressed 1D batch by 4–7 %
    /// (paired-median, n=24).
    /// 1D batch leaf-id traversal — counterpart to `find_leaf_id_with_ood`
    /// but amortised across `xsimd::batch<value_type>::size` points per
    /// iteration. Used by `Function::eval_batch_tile` when this subtree
    /// owns the whole domain (so OOD detection collapses into the same
    /// unsigned-wrap test as the table index — see the per-point variant
    /// above).
    ///
    /// Pipeline per SIMD chunk:
    ///   1. SIMD compute: `q = (x - lo) * inv_span_bins` for `simd_size`
    ///      lanes (one `load_unaligned`, one sub, one mul). Truncating
    ///      conversion to int produces INT64_MIN on x86 for OOD doubles,
    ///      which compares above `mask` as uint64 — same single-cmp
    ///      OOD trick used by the scalar variant.
    ///   2. Scalar lane sweep: per-lane table lookup `leaf_table_[q]`
    ///      and histogram bump `++counts[id]`. Vectorised gather/scatter
    ///      lose to bank-conflict serialization on shared counters
    ///      (FINUFFT spread.hpp:454-457 documents the same finding).
    ///
    /// The trailing `n % simd_size` points are dispatched through the
    /// scalar quantize for clarity; the loop is short enough that the
    /// branch-predictor handles it without measurable cost.
    template <class LeafIdT, class CountT>
    BAOBZI_ALWAYS_INLINE auto
    find_leaf_ids_batch(const value_type *xp, LeafIdT *leaf_ids,
                        CountT *counts, std::uint32_t ood_id,
                        std::size_t n) const -> void
        requires (input_dim == 1)
    {
        using batch_t                 = xsimd::batch<value_type>;
        constexpr std::size_t lanes   = batch_t::size;
        constexpr std::size_t aligned = batch_t::arch_type::alignment();

        const std::size_t mask  = (std::size_t{1} << leaf_table_depth_) - 1;
        const auto        lo_v  = batch_t::broadcast(lower_[0]);
        const auto        inv_v = batch_t::broadcast(inv_span_bins_[0]);

        auto place_one = [&](std::size_t dst, std::int64_t qi) {
            const auto id = (static_cast<std::uint64_t>(qi) > mask)
                                ? ood_id
                                : leaf_table_[static_cast<std::size_t>(qi)];
            leaf_ids[dst] = static_cast<LeafIdT>(id);
            ++counts[id];
        };

        alignas(aligned) std::array<value_type, lanes> q_arr{};
        const std::size_t n_simd = (n / lanes) * lanes;
        for (std::size_t i = 0; i < n_simd; i += lanes) {
            const auto x_v = batch_t::load_unaligned(xp + i);
            ((x_v - lo_v) * inv_v).store_aligned(q_arr.data());
            for (std::size_t j = 0; j < lanes; ++j)
                place_one(i + j, static_cast<std::int64_t>(q_arr[j]));
        }
        for (std::size_t i = n_simd; i < n; ++i)
            place_one(i, static_cast<std::int64_t>(
                            (xp[i] - lower_[0]) * inv_span_bins_[0]));
    }

    [[nodiscard]] BAOBZI_ALWAYS_INLINE auto
    find_leaf_id_with_ood(const input_type &x, std::uint32_t ood_id) const -> std::uint32_t {
        const std::size_t bits = leaf_table_depth_;
        const std::size_t mask = (std::size_t{1} << bits) - 1;
        std::size_t idx = 0;
        if constexpr (input_dim == 1) {
            const auto q0 = static_cast<std::int64_t>(
                (x - lower_[0]) * inv_span_bins_[0]);
            if (static_cast<std::uint64_t>(q0) > mask) [[unlikely]] return ood_id;
            idx = static_cast<std::size_t>(q0);
        } else {
            for (std::size_t d = 0; d < input_dim; ++d) {
                const auto qd = static_cast<std::int64_t>(
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
    [[nodiscard]] auto get_node_index(const input_type &x) const -> std::size_t {
        dim_array_t lo = lower_;
        dim_array_t hi = upper_;
        index_t curr_index = 0;
        while (!nodes_[curr_index].is_leaf()) {
            index_t child_idx = 0;
            poet::static_for<input_dim>([&](auto D) -> void {
                constexpr std::size_t d = D;
                const value_type mid_d = (lo[d] + hi[d]) * value_type{0.5};
                const value_type xd = [&]() -> value_type {
                    if constexpr (poly_eval::detail::hasTupleSize_v<input_type>) return x[d];
                    else return x;
                }();
                const bool upper = (xd > mid_d);
                child_idx |= (static_cast<index_t>(upper) << d);
                (upper ? lo[d] : hi[d]) = mid_d;
            });
            curr_index = nodes_[curr_index].first_child_idx() + child_idx;
        }
        return curr_index;
    }

    [[nodiscard]] constexpr auto size() const -> std::size_t { return nodes_.size(); }
    [[nodiscard]] constexpr auto max_depth() const -> std::size_t { return max_depth_; }
    [[nodiscard]] auto lower() const noexcept -> const dim_array_t & { return lower_; }
    [[nodiscard]] auto upper() const noexcept -> const dim_array_t & { return upper_; }

    [[nodiscard]] auto memory_usage() const -> std::size_t {
        std::size_t total = sizeof(*this);
        for (const auto &node : nodes_)
            total += node.memory_usage();
        return total;
    }

    auto &get_nodes() { return nodes_; }
    auto &get_nodes() const { return nodes_; }

    auto non_converged_panels() const -> const std::vector<NonConvergedPanel> & {
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

} // namespace baobzi::detail

#endif // BAOBZI_DETAIL_POLYTREE_HPP
