#ifndef BAOBZI_BAOBZI_HPP
#define BAOBZI_BAOBZI_HPP

/// \file baobzi/baobzi.hpp
/// \brief Public C++ API for the baobzi piecewise-Chebyshev function
///        approximator. Built on polyfit's leaf evaluators; baobzi adds the
///        adaptive tree (paneling) layer on top.
///
/// Thread safety. Once `baobzi::fit(...)` returns, the resulting Function
/// is immutable through its `operator()` overloads, which are safe to call
/// concurrently from multiple threads provided each call writes to a
/// disjoint output slice. The batch path allocates and frees its scratch
/// on each call via the caller-supplied allocator (default
/// `std::allocator<value_type>`); no state is carried between calls.
/// Callers that want pooled reuse should pass a stateful allocator —
/// `std::pmr::polymorphic_allocator` over a `monotonic_buffer_resource`
/// is the idiomatic choice. See `tests/test_threadsafe.cpp`.
///
/// Memory cost. Tightening `tol`, raising `max_depth`, raising
/// `max_memory_mib`, or enabling `allow_max_depth_leaves` all increase the
/// leaf count `L`. The Function holds one persistent allocation: the tree
/// itself, `O(L)`, shared by all threads and reported by
/// `Function::print_stats()` / `memory_usage()`. Each batch call
/// allocates a stack-local scratch of roughly
/// `tile_K * (4 + (input_dim + output_dim) * sizeof(value_type))` plus
/// `4 * (L + 1)` bytes (with `tile_K = max(65536, 32 * L)`) and frees it
/// on return.

#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include <polyfit/polyfit.hpp>

#include <baobzi/detail/eval_policy.hpp>

#include <baobzi/detail/function_impl.hpp>

namespace baobzi {

/// Runtime fit knobs. Every field here only affects fit-time
/// construction; there is no eval-time benefit to promoting any of them
/// to a template parameter, so they stay plain data.
struct options {
    /// How `tol` is interpreted by the convergence check. The default
    /// (`RelativeMax`) compares max-abs error on a sample grid against
    /// `tol * max(|f|)`; switch to `Absolute*` when `f` can be zero or
    /// when relative accuracy isn't meaningful.
    TolKind tol_kind = TolKind::RelativeMax;
    /// Tree-depth ceiling for the adaptive paneler. Hitting it without
    /// converging throws `MaxDepthExceeded` (or, with
    /// `allow_max_depth_leaves`, accepts the panel as best-effort).
    /// 50 is far above what any non-singular function needs; lower it
    /// to fail fast when a near-singularity is suspected.
    int     max_depth = 50;
    /// Hard cap on accumulated leaf storage during the fit, in MiB.
    /// `MemoryBudgetExceeded` is thrown when crossed; set to `0` to
    /// disable. The default (4 MiB) is intentionally strict: it keeps
    /// the evaluator inside L2 on most cores and forces callers to
    /// opt in explicitly before letting an adaptive paneler grow into
    /// memory regions that should raise eyebrows. Raise for ambitious
    /// 3D+ fits — the exception carries the offending panel so the
    /// caller can locate the singular region — or set to `0` to
    /// disable the check entirely.
    int     max_memory_mib = 4;
    /// When true, panels that fail tolerance at `max_depth` are kept
    /// as best-effort leaves. Inspect them via
    /// `Function::non_converged_panels()`. Default false (throw, with
    /// the panel list attached on the exception).
    bool    allow_max_depth_leaves = false;
    /// Force BFS to refine every panel to at least this depth before
    /// the per-panel tolerance test exits. Useful for driving the
    /// leaf-table fast path (`PolyTree::quantize_one`): a uniformly-
    /// refined tree of depth D in input_dim K builds a 2^(K*D)-entry
    /// quantize table the eval-time lookup collapses to one SIMD
    /// quantize + one u32 load. Table size grows as
    /// 2^(K*D) * 4 B and is capped at 64 K entries (256 KiB) per
    /// subtree, so values that push past `K*D > 16` will still build
    /// a uniform tree but not the table. Default 0 (no forcing —
    /// tol-based refinement only).
    int     min_uniform_depth      = 0;
};

/// Any callable that accepts `Domain` and returns a value.
template <class F, class Domain>
concept Fittable = requires(F f, Domain x) { f(x); };

namespace detail {

// Default leaf degree — matches AVX-512 double lane count; 4 / 12 / 16 may
// beat it on other widths. Revisit with the timing harness.
inline constexpr std::size_t kDefaultDegree = 8;

inline auto make_input(int input_dim, int output_dim, int degree,
                       double tol, const options &opts) -> TreeInput {
    TreeInput in{};
    in.input_dim              = input_dim;
    in.output_dim             = output_dim;
    in.degree                 = degree;
    in.tol                    = tol;
    in.tol_kind               = opts.tol_kind;
    in.max_depth              = opts.max_depth;
    in.max_memory_mib         = opts.max_memory_mib;
    in.allow_max_depth_leaves = opts.allow_max_depth_leaves;
    in.min_uniform_depth      = opts.min_uniform_depth;
    return in;
}

template <class Domain>
inline auto midpoint(const Domain &a, const Domain &b) -> Domain {
    if constexpr (std::is_arithmetic_v<Domain>) {
        return static_cast<Domain>(0.5 * (a + b));
    } else {
        Domain out{};
        for (std::size_t i = 0; i < a.size(); ++i)
            out[i] = 0.5 * (a[i] + b[i]);
        return out;
    }
}

template <class Domain>
inline auto half_length(const Domain &a, const Domain &b) -> Domain {
    if constexpr (std::is_arithmetic_v<Domain>) {
        return static_cast<Domain>(0.5 * (b - a));
    } else {
        Domain out{};
        for (std::size_t i = 0; i < a.size(); ++i)
            out[i] = 0.5 * (b[i] - a[i]);
        return out;
    }
}

template <class Domain>
constexpr auto domain_dim() -> int {
    if constexpr (std::is_arithmetic_v<Domain>)
        return 1;
    else
        return static_cast<int>(std::tuple_size_v<Domain>);
}

} // namespace detail

/// Canonical "fit f on [a, b] to tolerance" one-liner. Panel count falls out
/// of the adaptive tree; the leaf degree is a hyperparameter with a
/// reasonable default. `tol` is positional (no default) so every call site
/// makes its target accuracy explicit.
template <std::size_t Degree = detail::kDefaultDegree,
          EvalPolicy Policy = EvalPolicy::Balanced, class Func, class Domain>
    requires Fittable<Func, Domain>
[[nodiscard]] auto fit(Func f, Domain a, Domain b, double tol, options opts = {}) {
    if (!(tol > 0.0))
        throw std::invalid_argument(
            "baobzi::fit: tolerance must be > 0");

    using result_t = std::invoke_result_t<Func &, Domain>;
    constexpr int in_dim  = detail::domain_dim<Domain>();
    constexpr int out_dim = detail::domain_dim<result_t>();

    auto input = detail::make_input(in_dim, out_dim, static_cast<int>(Degree),
                                    tol, opts);
    return Function<Degree, Func, Policy>(input, detail::midpoint(a, b),
                                          detail::half_length(a, b),
                                          std::move(f));
}

} // namespace baobzi

#endif // BAOBZI_BAOBZI_HPP
