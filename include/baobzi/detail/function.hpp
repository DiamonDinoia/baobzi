#ifndef BAOBZI_DETAIL_FUNCTION_HPP
#define BAOBZI_DETAIL_FUNCTION_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <type_traits>
#include <utility>
#include <vector>

#include <poet/poet.hpp>
#include <polyfit/polyfit.hpp>

#include <baobzi/detail/compiler_macros.hpp>
#include <baobzi/detail/errors.hpp>
#include <baobzi/detail/node.hpp>
#include <baobzi/detail/numerics.hpp>
#include <baobzi/detail/polytree.hpp>
#include <baobzi/detail/tol_kind.hpp>
#include <baobzi/detail/value.hpp>

namespace baobzi {

struct sorted_t;  // Tag type — defined in <baobzi/baobzi.hpp>; passed by value, never dereferenced.

/// Adaptive piecewise-Chebyshev approximation of a user function. The fit
/// is materialized at construction (via `baobzi::fit`) into a flat array of
/// subtrees over a uniform top-level grid, each subtree built BFS to the
/// requested tolerance. After construction the object is immutable and its
/// evaluators are thread-safe — see header docstring on `baobzi/baobzi.hpp`.
template <std::size_t Degree, class Func>
class Function {
  public:
    using input_type = std::remove_cvref_t<poly_eval::fitInput_t<Func>>;
    using output_type = poly_eval::fitOutput_t<Func>;
    using value_type = poly_eval::detail::value_type_or_t<input_type>;
    using poly_eval_type = std::conditional_t<
        poly_eval::detail::hasTupleSize_v<input_type>,
        poly_eval::FuncEvalND<Func, Degree, poly_eval::FusionMode::Never, poly_eval::ScalarKernel::Hybrid>,
        poly_eval::FuncEval<Func, Degree, 1, poly_eval::FusionMode::Never, poly_eval::ScalarKernel::Hybrid>>;
    static constexpr auto degree = Degree;

    static constexpr std::size_t input_dim = detail::value_dim_v<input_type>;
    static constexpr std::size_t output_dim = detail::value_dim_v<output_type>;
    static constexpr std::size_t n_child = std::size_t{1} << input_dim;

    using node_t = detail::Node<Func, Degree>;
    using box_t = detail::Box<value_type, input_dim>;
    using dim_array_t = detail::Value<value_type, input_dim>;

    /// Approximate resident bytes — including subtree node arrays and the
    /// shared polyfit coefficient store. Useful for budget validation in
    /// callers that build many Functions.
    [[nodiscard]] auto memory_usage() const -> std::size_t {
        std::size_t mem = sizeof(*this);
        mem += polyfits_.capacity() * sizeof(poly_eval_type);
        for (const auto &subtree : subtrees_)
            mem += subtree.memory_usage();
        return mem;
    }

    /// Print a one-screen summary of the fit (node/leaf counts, depth, fit-
    /// time eval count, wall time, memory).
    auto print_stats() const -> void {
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
                  << n_nodes * detail::powi<static_cast<int>(input_dim)>(Degree) +
                         stats_.n_evals_root
                  << "\n";
        std::cout << "Total time to create tree: " << stats_.t_elapsed << " milliseconds\n";
        std::cout << "Approximate memory usage of tree: "
                  << static_cast<double>(mem) / (1024.0 * 1024.0) << " MiB\n";
    }

    /// Build a Function object by recursively fitting the domain.
    /// @throws MaxDepthExceeded   if any subtree fails to converge at
    ///                            `input.max_depth` (and
    ///                            `allow_max_depth_leaves == false`).
    /// @throws MemoryBudgetExceeded  if accumulated leaf storage crosses
    ///                            `input.max_memory_mib` MiB.
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
                        const std::array<value_type, 2> signed_hw{-child_hw[j], child_hw[j]};
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
                if (node.poly_eval_id() != 0u)
                    node.set_poly_eval_id(0);

                if (!node.is_leaf())
                    add_node_children_to_queue(q, current_box.center, half_width);
            }
            stats_.n_evals_root += static_cast<std::uint64_t>(
                nodes.size() *
                detail::powi<static_cast<int>(input_dim)>(Degree));

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

#ifndef NDEBUG
        // The Function box must equal the union of subtree boxes — both
        // layers carry the same domain (Function for the OOD pre-check,
        // subtrees as the descent invariant). If this fires, the bin
        // decomposition has drifted from the Function's stored bounds.
        {
            dim_array_t lo_min = subtrees_.front().lower();
            dim_array_t hi_max = subtrees_.front().upper();
            for (const auto &st : subtrees_) {
                for (std::size_t d = 0; d < input_dim; ++d) {
                    lo_min[d] = std::min(lo_min[d], st.lower()[d]);
                    hi_max[d] = std::max(hi_max[d], st.upper()[d]);
                }
            }
            for (std::size_t d = 0; d < input_dim; ++d) {
                const value_type span = upper_right_[d] - lower_left_[d];
                const value_type tol = std::max<value_type>(
                    1, std::abs(lower_left_[d]) + std::abs(upper_right_[d])) *
                    std::numeric_limits<value_type>::epsilon() * 16;
                assert(std::abs(lo_min[d] - lower_left_[d]) <= tol);
                assert(std::abs(hi_max[d] - upper_right_[d]) <= tol);
                (void)span;
            }
        }
#endif

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
    }

    /// Convert linear bin index to [dim] bin vector.
    [[nodiscard]] auto get_bins(std::size_t i_bin) const -> std::array<std::size_t, input_dim> {
        std::array<std::size_t, input_dim> out{};
        poet::static_for<input_dim - 1>([&](auto D) -> void {
            constexpr std::size_t d = D;
            out[d] = i_bin % n_subtrees_[d];
            i_bin /= n_subtrees_[d];
        });
        out[input_dim - 1] = i_bin;
        return out;
    }

    /// Find linear index of bin at a point.
    [[nodiscard]] BAOBZI_ALWAYS_INLINE auto get_linear_bin(const input_type &x) const -> std::size_t {
        auto axis_bin = [&](auto I) -> std::size_t {
            constexpr std::size_t i = I;
            const value_type xi = [&]() -> value_type {
                if constexpr (poly_eval::detail::hasTupleSize_v<input_type>) return x[i];
                else return x;
            }();
            return static_cast<std::size_t>((xi - lower_left_[i]) * inv_bin_size_[i]);
        };
        if constexpr (input_dim == 1) {
            return axis_bin(std::integral_constant<std::ptrdiff_t, 0>{});
        } else {
            std::size_t result = axis_bin(std::integral_constant<std::ptrdiff_t, static_cast<std::ptrdiff_t>(input_dim) - 1>{});
            // Horner-form: result accumulates from highest axis down.
            poet::static_for<static_cast<std::ptrdiff_t>(input_dim) - 1>([&](auto K) -> void {
                constexpr std::ptrdiff_t r = static_cast<std::ptrdiff_t>(input_dim) - 2 - K;
                result = result * n_subtrees_[r]
                       + axis_bin(std::integral_constant<std::ptrdiff_t, r>{});
            });
            return result;
        }
    }

    [[nodiscard]] auto find_node(const input_type &x) const -> const node_t & { return subtrees_[get_linear_bin(x)].find_node(x); }

    /// Default per-call tile cap for the batch path. The adaptive floor
    /// (`n_leaves * kMinPtsPerLeaf`) raises this when low-leaf-count
    /// Functions would otherwise starve the polyfit batch kernel.
    static constexpr std::size_t kDefaultTileK = 65536;

    /// Batch evaluation: `n_trg` points written into `res`.
    ///
    /// Pipeline (unsorted, the general path):
    ///
    ///   1. **Leaf-id traversal.** For each input point, look up the owning
    ///      leaf index (the `polyfits_` slot that holds its Horner
    ///      coefficients). When the Function has a single subtree with a
    ///      precomputed leaf-table the lookup is a quantize + u32 load and
    ///      folds OOD detection into the same unsigned wrap test. Otherwise,
    ///      we descend the tree per point. Out-of-domain points are tagged
    ///      with the sentinel id `n_leaves` and counted in their own bucket.
    ///   2. **Counting sort + in-place exclusive scan.** Histogram leaf
    ///      populations into `counts[0..n_leaves]`, then exclusive-scan
    ///      `counts` in place. After scatter (next stage) consumes `counts`
    ///      as a cursor, `counts[k]` equals the one-past-end of leaf k's
    ///      slice — enough to recover `(off, cnt)` for the per-leaf dispatch
    ///      by walking ids with a running `prev_end` (Reinecke's trick).
    ///   3. **Scatter to packed layout.** Walk points in input order; for
    ///      each, append its coordinates to `xp_packed` at its leaf's cursor
    ///      (`counts[id]++`) and record the inverse mapping in
    ///      `perm[dst] = i`. Result: points sharing a leaf land contiguously
    ///      and in lock-step between `xp_packed` and the soon-to-be-filled
    ///      `out_packed`.
    ///   4. **Per-leaf SIMD batch eval.** For each non-empty leaf, hand its
    ///      contiguous slice of `xp_packed` to polyfit's SIMD batch kernel
    ///      once and write into the same slice of `out_packed`. This is the
    ///      whole point of the sort: one fixed coefficient set, one SIMD
    ///      Horner stream, no per-point branch on which leaf to evaluate.
    ///      The OOD bucket is filled with NaN instead of evaluated.
    ///   5. **Permute back to caller order.** For each `dst`, copy
    ///      `out_packed[dst]` into `res[perm[dst] * output_dim]`. The store
    ///      address is random in `res`, so we prefetch ahead by `LOOKAHEAD`
    ///      to hide the RFO latency that otherwise dominates 1D throughput.
    ///
    /// Tiny batches (`n_trg < kSortThreshold`) skip stages 2–5 and just
    /// loop point-at-a-time — the counting sort can't amortize its setup
    /// at that size. Large batches are tiled (`kDefaultTileK`, lifted by an
    /// adaptive floor for high-leaf-count Functions) so the packed buffers
    /// fit in L1d/L2.
    ///
    /// For 1D, callers who can promise sortedness should prefer the
    /// `(xp, res, n, baobzi::Sorted)` overload — it skips stages 2, 3, 5
    /// entirely and runs ~3–4× faster.
    ///
    /// Thread-safe: a single Function may be called concurrently from
    /// multiple threads provided each call's `xp` and `res` slices do not
    /// overlap with another thread's. Scratch is allocated and freed
    /// inside each batch call (stack-local owning buffers); the
    /// Function's internal state (nodes, polyfits) is immutable after
    /// construction. Pinned by `tests/test_threadsafe.cpp`.
    /// @param xp     `n_trg * input_dim` packed input coordinates.
    /// @param res    `n_trg * output_dim` output buffer.
    /// @param n_trg  number of points to evaluate.
    BAOBZI_FLATTEN auto operator()(const value_type *xp, value_type *res, std::size_t n_trg) const -> void {
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
                const detail::Value<value_type, input_dim> xi(xp + (input_dim * i_trg));
                const detail::Value<value_type, output_dim> tmp = (*this)(xi);
                std::copy(tmp.begin(), tmp.end(), res + (i_trg * output_dim));
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

        // Scratch is allocated once here (sized to one tile) and reused
        // across tiles. Only `counts` needs zeroing between tiles; the
        // other buffers are write-before-read on every tile.
        const auto n_leaves = static_cast<std::uint32_t>(polyfits_.size());
        BatchScratch s(n_leaves, std::min(n_trg, tile_K));

        if (n_trg > tile_K) {
            for (std::size_t tile_off = 0; tile_off < n_trg; tile_off += tile_K) {
                const std::size_t tile_n = std::min(tile_K, n_trg - tile_off);
                eval_batch_tile(xp + (input_dim * tile_off),
                                res + (output_dim * tile_off), tile_n, s);
            }
            return;
        }
        eval_batch_tile(xp, res, n_trg, s);
    }

    /// Sorted-input batch evaluation (1D).
    ///
    /// The caller promises `xp[i] <= xp[i+1]`. Under that promise the
    /// leaf-id sequence is monotone non-decreasing (1D leaves tile the
    /// domain in coordinate order), so points sharing a leaf are
    /// already contiguous in the input. That collapses the unsorted
    /// pipeline's five stages to two:
    ///
    ///   * find the leaf at `i`, scan forward until the leaf changes,
    ///   * dispatch the run `[i, j)` directly to polyfit's SIMD batch
    ///     kernel writing straight into `res + i`.
    ///
    /// No `leaf_ids` write, no counts/prefix-sum, no scatter into
    /// `xp_packed`, no permute back from `out_packed`, no `thread_local`
    /// scratch — the input and output buffers themselves are the packed
    /// layout, and the run-length scan amortizes the per-leaf eval as
    /// well as the counting sort did.
    ///
    /// OOD points form a contiguous prefix and/or suffix (since the
    /// input is sorted) and are NaN-filled by two short guards around
    /// the main loop.
    ///
    /// On a paired bench (1D, presorted, `-O3 -march=native`) this path
    /// is ~3–4× faster than calling `operator()(xp, res, n)` on the
    /// same buffer: ~1.5 ns/eval vs ~5.5 ns/eval at N=1e6 across both
    /// the leaf-table fast path and the descent fallback. ins/eval drops
    /// from ~36 to ~14 — exactly the work removed by skipping stages
    /// 2, 3, 5 of the unsorted pipeline.
    ///
    /// `res` must hold `n * output_dim` elements. Restricted to
    /// `input_dim == 1`: 2D/3D leaf-id sequences are not monotone under
    /// single-axis sorting so the same trick does not apply.
    BAOBZI_FLATTEN auto operator()(const value_type *xp, value_type *res,
                                   std::size_t n, sorted_t) const -> void
        requires (input_dim == 1)
    {
        if (n == 0) [[unlikely]] return;

        constexpr value_type nan_v = std::numeric_limits<value_type>::quiet_NaN();
        auto write_nan = [&](std::size_t i) -> void {
            for (std::size_t j = 0; j < output_dim; ++j)
                res[i * output_dim + j] = nan_v;
        };

        const auto n_leaves = static_cast<std::uint32_t>(polyfits_.size());
        const std::uint32_t ood_id = n_leaves;
        const bool fast = subtrees_.size() == 1 && subtrees_.front().has_leaf_table();

        auto leaf_id_at = [&](std::size_t i) -> std::uint32_t {
            const value_type x = xp[i];
            if (fast)
                return subtrees_.front().find_leaf_id_with_ood(x, ood_id);
            if (x < lower_left_[0] || x >= upper_right_[0])
                return ood_id;
            return subtrees_[get_linear_bin(x)].find_leaf_id(x);
        };

        std::size_t i = 0;
        // OOD prefix: sorted input means anything below lower_left_[0]
        // is contiguous at the front.
        while (i < n && xp[i] < lower_left_[0]) { write_nan(i); ++i; }

        while (i < n) {
            if (xp[i] >= upper_right_[0]) [[unlikely]] {
                // OOD suffix begins here (and continues to the end).
                do { write_nan(i); ++i; } while (i < n);
                break;
            }
            const std::uint32_t id = leaf_id_at(i);
            if (id == ood_id) [[unlikely]] {
                // Fast-path quantize wrap can flag points slightly above
                // the upper bound that survived the explicit prefix
                // guards (e.g. NaN). Fall back to a per-point NaN here.
                write_nan(i); ++i;
                continue;
            }
            std::size_t j = i + 1;
            while (j < n && leaf_id_at(j) == id) ++j;
            polyfits_[id](xp + i, res + i, j - i);
            i = j;
        }
    }

    /// RAII scratch buffers for `eval_batch_tile`, owned by `operator()`
    /// and reused across tiles. `leaf_ids` is a raw byte view
    /// reinterpreted as `u16[]` or `u32[]` depending on
    /// `leaf_ids_fit_u16` (u16 halves L1d traffic on the scatter when
    /// leaf count permits). `counts` doubles as the histogram, the
    /// exclusive-scan output, and the scatter cursor — after scatter,
    /// `counts[k]` is the one-past-end of leaf k's packed slice
    /// (Reinecke's trick), removing the need for a separate `offsets`
    /// array.
    class BatchScratch {
      public:
        BatchScratch(std::uint32_t n_leaves, std::size_t n_cap)
            : leaf_ids_fit_u16_(n_leaves <= std::numeric_limits<std::uint16_t>::max()),
              leaf_ids_(std::make_unique_for_overwrite<std::byte[]>(
                  (leaf_ids_fit_u16_ ? 2 : 4) * n_cap)),
              perm_inv_(std::make_unique_for_overwrite<std::uint32_t[]>(n_cap)),
              xp_packed_(std::make_unique_for_overwrite<value_type[]>(input_dim * n_cap)),
              out_packed_(std::make_unique_for_overwrite<value_type[]>(output_dim * n_cap)),
              counts_(std::make_unique_for_overwrite<std::uint32_t[]>(n_leaves + 1)) {}

        [[nodiscard]] auto leaf_ids() const -> std::byte *     { return leaf_ids_.get(); }
        [[nodiscard]] auto perm_inv() const -> std::uint32_t * { return perm_inv_.get(); }
        [[nodiscard]] auto xp_packed() const -> value_type *   { return xp_packed_.get(); }
        [[nodiscard]] auto out_packed() const -> value_type *  { return out_packed_.get(); }
        [[nodiscard]] auto counts() const -> std::uint32_t *   { return counts_.get(); }
        [[nodiscard]] auto leaf_ids_fit_u16() const -> bool    { return leaf_ids_fit_u16_; }

      private:
        bool leaf_ids_fit_u16_;
        std::unique_ptr<std::byte[]>     leaf_ids_;
        std::unique_ptr<std::uint32_t[]> perm_inv_;
        std::unique_ptr<value_type[]>    xp_packed_;
        std::unique_ptr<value_type[]>    out_packed_;
        std::unique_ptr<std::uint32_t[]> counts_;
    };

    /// Per-tile body of the unsorted batch pipeline (stages 1–5, see the
    /// `operator()(xp, res, n)` doc above). The caller ensures
    /// `n_trg >= kSortThreshold` and `n_trg <= tile_K`, and owns the
    /// scratch buffers (sized to one tile, reused across tiles).
    auto eval_batch_tile(const value_type *xp, value_type *res,
                         std::size_t n_trg, const BatchScratch &s) const -> void {
        const auto n_leaves = static_cast<std::uint32_t>(polyfits_.size());
        const std::uint32_t ood_id = n_leaves; // sentinel bucket for out-of-domain

        auto *perm_inv   = s.perm_inv();
        auto *xp_packed  = s.xp_packed();
        auto *out_packed = s.out_packed();
        auto *counts     = s.counts();

        // `counts` is the only scratch buffer that needs reset between
        // tiles — the others are write-before-read.
        std::memset(counts, 0, (n_leaves + 1) * sizeof(std::uint32_t));

        // Traversal: leaf id per point + population histogram. With a
        // single subtree owning a leaf table (the common compact-domain
        // case), the subtree's quantize already covers the whole
        // Function domain — OOD detection collapses into the same
        // unsigned-wrap test that produces the table index, dropping
        // the Function-level OOD pre-check and get_linear_bin entirely.
        auto find_loop = [&](auto *leaf_ids_vec) -> void {
            using LeafIdT = std::remove_reference_t<decltype(leaf_ids_vec[0])>;
            if (subtrees_.size() == 1 && subtrees_.front().has_leaf_table()) {
                const auto &st = subtrees_.front();
                if constexpr (input_dim == 1) {
                    // SIMD-amortised quantize for the 1D table fast path.
                    // ND keeps the per-point loop: a SIMD batch of ND points
                    // would need stride-`input_dim` loads (or a transpose),
                    // and the scalar path is already fast there.
                    st.find_leaf_ids_batch(xp, leaf_ids_vec, counts,
                                           ood_id, n_trg);
                } else for (std::size_t i = 0; i < n_trg; ++i) {
                    const detail::Value<value_type, input_dim> xi(xp + (input_dim * i));
                    const std::uint32_t id = st.find_leaf_id_with_ood(xi, ood_id);
                    leaf_ids_vec[i] = static_cast<LeafIdT>(id);
                    ++counts[id];
                }
            } else {
                for (std::size_t i = 0; i < n_trg; ++i) {
                    const detail::Value<value_type, input_dim> xi(xp + (input_dim * i));
                    bool in_domain = true;
                    poet::static_for<input_dim>([&](auto D) -> void {
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
        if (s.leaf_ids_fit_u16()) find_loop(reinterpret_cast<std::uint16_t *>(s.leaf_ids()));
        else                      find_loop(reinterpret_cast<std::uint32_t *>(s.leaf_ids()));

        // Exclusive scan in place — counts[k] becomes the start of leaf k's
        // packed slice. After the scatter below consumes it as a cursor,
        // counts[k] will equal the one-past-end of that slice (Reinecke).
        std::exclusive_scan(counts, counts + n_leaves + 1,
                            counts, std::uint32_t{0});

        // Scatter inputs to packed layout. counts[] is consumed as a cursor;
        // afterwards counts[id] equals the one-past-end of leaf id's slice.
        auto scatter_loop = [&](const auto *leaf_ids_vec) -> void {
            for (std::size_t i = 0; i < n_trg; ++i) {
                const std::uint32_t id = leaf_ids_vec[i];
                const std::uint32_t dst = counts[id]++;
                perm_inv[i] = dst;
                if constexpr (input_dim == 1) {
                    xp_packed[dst] = xp[i];
                } else {
                    const value_type *src = xp + (input_dim * i);
                    value_type *dstp = xp_packed + (input_dim * dst);
                    poet::static_for<input_dim>([&](auto D) -> void {
                        constexpr std::size_t d = D;
                        dstp[d] = src[d];
                    });
                }
            }
        };
        if (s.leaf_ids_fit_u16()) scatter_loop(reinterpret_cast<const std::uint16_t *>(s.leaf_ids()));
        else                      scatter_loop(reinterpret_cast<const std::uint32_t *>(s.leaf_ids()));

        // Per-leaf SIMD batch eval. Walk ids with a running prev_end:
        // counts[id] is the slice end; cnt = end - prev_end; off = prev_end.
        // Speculative prefetch of the next non-empty leaf's coefficient
        // store hides cacheline-fill latency when leaves are small.
        std::uint32_t prev_end = 0;
        for (std::uint32_t id = 0; id < n_leaves; ++id) {
            const std::uint32_t end = counts[id];
            const std::uint32_t cnt = end - prev_end;
            const std::uint32_t off = prev_end;
            prev_end = end;
            if (cnt == 0) continue;
#if defined(__GNUC__) || defined(__clang__)
            std::uint32_t next_id = id + 1;
            while (next_id < n_leaves && counts[next_id] == end) ++next_id;
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
                    xp_packed + (input_dim * off));
                CO *outs = reinterpret_cast<CO *>(
                    out_packed + (output_dim * off));
                polyfits_[id](pts, outs, static_cast<std::size_t>(cnt));
            } else {
                polyfits_[id](xp_packed + off,
                              out_packed + off,
                              static_cast<std::size_t>(cnt));
            }
        }

        // Fill OOD slots with NaN.
        const std::uint32_t ood_off = prev_end;
        const std::uint32_t ood_cnt = counts[ood_id] - prev_end;
        if (ood_cnt) {
            constexpr value_type nan_v = std::numeric_limits<value_type>::quiet_NaN();
            for (std::uint32_t k = 0; k < ood_cnt; ++k)
                for (std::size_t j = 0; j < output_dim; ++j)
                    out_packed[output_dim * (ood_off + k) + j] = nan_v;
        }

        // Permute outputs back to caller order.
        // Inverse-permutation form: sequential write to `res`, random read
        // from `out_packed`. Sequential stores coalesce (no RFO); random
        // loads are easily prefetched.
        [[maybe_unused]] constexpr std::size_t LOOKAHEAD = 32;
        for (std::size_t i = 0; i < n_trg; ++i) {
#if defined(__GNUC__) || defined(__clang__)
            if (i + LOOKAHEAD < n_trg) {
                const std::uint32_t pf = perm_inv[i + LOOKAHEAD];
                __builtin_prefetch(out_packed + (output_dim * pf),
                                   /*rw=*/0, /*locality=*/0);
            }
#endif
            const std::uint32_t src = perm_inv[i];
            const value_type *srcp = out_packed + (output_dim * src);
            value_type *dstp = res + (output_dim * i);
            poet::static_for<output_dim>([&](auto J) -> void {
                constexpr std::size_t j = J;
                dstp[j] = srcp[j];
            });
        }
    }

    /// Point evaluation. Returns NaN for out-of-domain inputs.
    ///
    /// `BAOBZI_ALWAYS_INLINE` is load-bearing: without it clang keeps
    /// this out-of-line at -O3 and every scalar evaluator pays a per-
    /// point `callq` + xmm spill (verified via objdump).
    ///
    /// ND residual: `polyfit::FuncEvalND::evalCanonical` is not
    /// always-inlined upstream, so leaf eval keeps one `callq` per ND
    /// point. `BAOBZI_FLATTEN` here doesn't fix it — the durable fix is
    /// `PF_ALWAYS_INLINE` on `evalCanonical`.
    [[nodiscard]] BAOBZI_ALWAYS_INLINE auto
    operator()(const input_type &x) const -> output_type {
        bool ood = false;
        poet::static_for<input_dim>([&](auto D) -> void {
            constexpr std::size_t d = D;
            const value_type xd = [&]() -> value_type {
                if constexpr (poly_eval::detail::hasTupleSize_v<input_type>) return x[d];
                else return x;
            }();
            if (xd < lower_left_[d] || xd >= upper_right_[d]) ood = true;
        });
        if (ood) [[unlikely]]
            return output_type{std::numeric_limits<value_type>::quiet_NaN()};
        return polyfits_[subtrees_[get_linear_bin(x)].find_leaf_id(x)](x);
    }

    /// Panels where adaptive paneling failed at `max_depth`. Always empty
    /// unless `options.allow_max_depth_leaves == true` was set; the default
    /// path throws `MaxDepthExceeded` (which carries the same list) instead.
    [[nodiscard]] auto non_converged_panels() const -> const std::vector<NonConvergedPanel> & {
        return non_converged_panels_;
    }

    [[nodiscard]] auto get_bounds() const -> std::pair<dim_array_t, dim_array_t> {
        return std::make_pair(lower_left_, upper_right_);
    }

    /// True iff every subtree built a leaf-id lookup table. When false,
    /// at least one subtree falls through to the descent path inside
    /// `find_leaf_id` — measurably slower per eval. Exposed so tests can
    /// pin the leaf-table threshold behavior.
    [[nodiscard]] auto all_subtrees_have_leaf_table() const noexcept -> bool {
        for (const auto &st : subtrees_)
            if (!st.has_leaf_table()) return false;
        return !subtrees_.empty();
    }

    /// Compile-time-N batch point evaluation (1D scalar inputs/outputs).
    ///
    /// Provided so consumers with a small fixed-size pack of evaluations
    /// can express intent at the call site without an ad-hoc loop. Three
    /// regimes by `N`:
    ///
    ///   * `N <= 16` (small): poet::static_for fully unrolls the
    ///     scalar fan-out. operator() is BAOBZI_ALWAYS_INLINE so the
    ///     N FMA chains run on independent registers — best ILP at
    ///     small N.
    ///   * `16 < N < kBatchPathFloor` (medium): plain for-loop. The
    ///     compile-time N still lets the compiler unroll partially,
    ///     and each operator() is inlined, so this matches the
    ///     hand-rolled scalar_loop baseline. Avoids both poet's
    ///     fully-unrolled code bloat at large N and the batch path's
    ///     fixed setup overhead (counts/perm scratch).
    ///   * `N >= kBatchPathFloor` (large): delegate to the SIMD-batched
    ///     `operator()(xp, ys, n)`. The batch-path leaf dispatch
    ///     amortises its fixed overhead only at high N.
    ///
    /// kBatchPathFloor = 1024 picked from the bench_pack_scatter
    /// crossover sweep (1d_runge deg=8, SPR, taskset -c 2). The
    /// 32→1024 bump is commit `3939d75 perf(eval_pack): raise
    /// batch-path floor from 32 to 1024`.
    template <std::size_t N>
    [[nodiscard]] BAOBZI_FLATTEN auto eval_pack(const std::array<value_type, N> &xs) const
        -> std::array<value_type, N>
        requires (input_dim == 1 && output_dim == 1)
    {
        constexpr std::size_t kBatchPathFloor = 1024;
        std::array<value_type, N> ys{};
        if constexpr (N == 0) {
            return ys;
        } else if constexpr (N >= kBatchPathFloor) {
            (*this)(xs.data(), ys.data(), N);
        } else if constexpr (N <= 16) {
            poet::static_for<N>([&](auto I) -> void {
                constexpr std::size_t i = I;
                ys[i] = (*this)(xs[i]);
            });
        } else {
            for (std::size_t i = 0; i < N; ++i) ys[i] = (*this)(xs[i]);
        }
        return ys;
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

    /// Construction-time stats reported via `print_stats()`.
    struct {
        std::size_t base_depth = 0;
        std::uint64_t n_evals_root = 0;
        std::uint32_t t_elapsed = 0;
    } stats_;
};

} // namespace baobzi

#endif // BAOBZI_DETAIL_FUNCTION_HPP
