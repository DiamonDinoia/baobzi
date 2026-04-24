#ifndef BAOBZI_BAOBZI_HPP
#define BAOBZI_BAOBZI_HPP

/// \file baobzi/baobzi.hpp
/// \brief Public C++ API for the baobzi piecewise-Chebyshev function
///        approximator. Mirrors polyfit's `poly_eval::fit` factories; baobzi
///        adds the adaptive tree (paneling) layer on top.

#include <array>
#include <concepts>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include <polyfit/polyeval.hpp>
#include <poet/poet.hpp>

#include <baobzi/detail/function_impl.hpp>

namespace baobzi {

/// Tag type selecting whether the batch-eval hot path should split the
/// "find-node" and "evaluate-polyfit" passes (original baobzi behaviour) or
/// just loop point-at-a-time. Promoted from a runtime flag to a type-level
/// choice so the branch is eliminated in the instantiation the user picked.
template <bool B> using SplitMultiEval = std::bool_constant<B>;
inline constexpr SplitMultiEval<true>  SplitMultiEvalOn{};
inline constexpr SplitMultiEval<false> SplitMultiEvalOff{};

/// Runtime fit knobs. Every field here only affects fit-time construction;
/// there is no eval-time benefit to promoting any of them to a template
/// parameter, so they stay plain data.
struct options {
    TolKind tol_kind              = TolKind::RelativeMax;
    double  minimum_leaf_fraction = 0.0;
    int     min_depth             = 0;
    int     max_depth             = 50;
    int     n_samples_per_dim     = 8;
};

/// Any callable that accepts `Domain` and returns a value.
template <class F, class Domain>
concept Fittable = requires(F f, Domain x) { f(x); };

namespace detail {

// Supported runtime degrees for the runtime-degree and tolerance-driven
// factories. Adjust by editing here.
inline constexpr int kMinRuntimeDegree = 4;
inline constexpr int kMaxRuntimeDegree = 16;

// Default leaf degree — matches AVX-512 double lane count; 4 / 12 / 16 may
// beat it on other widths. Revisit with the timing harness.
inline constexpr std::size_t kDefaultDegree = 8;

inline TreeInput make_input(int input_dim, int output_dim, int degree,
                            double tol, const options &opts) {
    TreeInput in{};
    in.input_dim             = input_dim;
    in.output_dim            = output_dim;
    in.degree                = degree;
    in.tol                   = tol;
    in.minimum_leaf_fraction = opts.minimum_leaf_fraction;
    in.min_depth             = opts.min_depth;
    in.max_depth             = opts.max_depth;
    in.tol_kind              = opts.tol_kind;
    in.n_samples_per_dim     = opts.n_samples_per_dim;
    return in;
}

template <class Domain>
inline Domain midpoint(const Domain &a, const Domain &b) {
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
inline Domain half_length(const Domain &a, const Domain &b) {
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
constexpr int domain_dim() {
    if constexpr (std::is_arithmetic_v<Domain>)
        return 1;
    else
        return static_cast<int>(std::tuple_size_v<Domain>);
}

// --- Runtime-degree variant plumbing -------------------------------------
template <bool S, class Func, int Lo, class Seq> struct RuntimeVariantImpl;
template <bool S, class Func, int Lo, int... Is>
struct RuntimeVariantImpl<S, Func, Lo, std::integer_sequence<int, Is...>> {
    using type = std::variant<
        ::baobzi::Function<static_cast<std::size_t>(Lo + Is), Func, S>...>;
};
template <bool S, class Func, int Lo, int Hi>
using RuntimeVariant =
    typename RuntimeVariantImpl<S, Func, Lo,
        std::make_integer_sequence<int, Hi - Lo + 1>>::type;

} // namespace detail

/// Type-erased-over-degree evaluator returned by the runtime-degree `fit`
/// overload. Wraps a `std::variant` of `Function<Degree, Func, S>` for each
/// supported `Degree`, so the concrete evaluator type is still reachable via
/// `variant()` but casual callers only need `operator()`.
template <class Variant>
class RuntimeFunction {
  public:
    explicit RuntimeFunction(Variant v) : v_(std::move(v)) {}

    template <class X>
    auto operator()(const X &x) const {
        return std::visit([&](auto const &fn) { return fn(x); }, v_);
    }

    template <class T>
    void operator()(const T *xp, T *res, std::size_t n) const {
        std::visit([&](auto const &fn) { fn(xp, res, n); }, v_);
    }
    template <class T>
    void operator()(const T *xp, T *res, int n) const {
        (*this)(xp, res, static_cast<std::size_t>(n));
    }

    const Variant &variant() const { return v_; }

  private:
    Variant v_;
};

// ---------------------------------------------------------------------------
// (1) Tolerance-first, default degree — the canonical "fit f to eps on
//     [a, b]" one-liner. Panel count falls out of the adaptive tree; the
//     leaf degree is a hyperparameter with a reasonable default.
// ---------------------------------------------------------------------------
template <std::size_t Degree = detail::kDefaultDegree, class Func, class Domain,
          class Tag = SplitMultiEval<true>>
    requires Fittable<Func, Domain>
inline auto fit(Func f, double tol, Domain a, Domain b, options opts = {},
                Tag = {}) {
    if (!(tol > 0.0))
        throw std::invalid_argument(
            "baobzi::fit: tolerance must be > 0");

    using result_t = std::invoke_result_t<Func &, Domain>;
    constexpr int in_dim  = detail::domain_dim<Domain>();
    constexpr int out_dim = detail::domain_dim<result_t>();
    constexpr bool S      = Tag::value;

    auto input = detail::make_input(in_dim, out_dim, static_cast<int>(Degree),
                                    tol, opts);
    return Function<Degree, Func, S>(input, detail::midpoint(a, b),
                                     detail::half_length(a, b), std::move(f));
}

// ---------------------------------------------------------------------------
// (2) Runtime-integer degree with tolerance. POET dispatches `n` to a
//     compile-time specialisation; the result is a `RuntimeFunction`
//     wrapping a `std::variant` of every candidate `Function<Degree, Func, S>`.
// ---------------------------------------------------------------------------
template <class Func, class Domain, class Tag = SplitMultiEval<true>>
    requires Fittable<Func, Domain>
inline auto fit(Func f, int n, double tol, Domain a, Domain b,
                options opts = {}, Tag = {}) {
    if (n < detail::kMinRuntimeDegree || n > detail::kMaxRuntimeDegree)
        throw std::invalid_argument(
            "baobzi::fit: runtime degree out of supported range");
    if (!(tol > 0.0))
        throw std::invalid_argument(
            "baobzi::fit: tolerance must be > 0");

    constexpr bool S = Tag::value;
    using result_t = std::invoke_result_t<Func &, Domain>;
    constexpr int in_dim  = detail::domain_dim<Domain>();
    constexpr int out_dim = detail::domain_dim<result_t>();
    auto input = detail::make_input(in_dim, out_dim, n, tol, opts);

    using Variant = detail::RuntimeVariant<S, Func,
                                           detail::kMinRuntimeDegree,
                                           detail::kMaxRuntimeDegree>;
    std::optional<Variant> holder;
    poet::dispatch(
        [&]<int Degree>() {
            using FT = ::baobzi::Function<static_cast<std::size_t>(Degree), Func, S>;
            holder.emplace(Variant(std::in_place_type<FT>, input,
                                   detail::midpoint(a, b),
                                   detail::half_length(a, b), f));
        },
        poet::DispatchParam<
            poet::make_range<detail::kMinRuntimeDegree,
                             detail::kMaxRuntimeDegree>>{n});
    return RuntimeFunction<Variant>(std::move(*holder));
}

// ---------------------------------------------------------------------------
// (3) Fixed-tolerance, compile-time-degree overload — kept for
//     microbenchmarks and callers that want a deterministic panel count.
//     Uses the internal default tolerance (1e-10).
// ---------------------------------------------------------------------------
template <std::size_t Degree, class Func, class Domain,
          class Tag = SplitMultiEval<true>>
    requires Fittable<Func, Domain>
inline auto fit(Func f, Domain a, Domain b, options opts = {}, Tag = {}) {
    return fit<Degree>(std::move(f), /*tol=*/1e-10, a, b, opts, Tag{});
}

} // namespace baobzi

#endif // BAOBZI_BAOBZI_HPP
