#ifndef BAOBZI_DETAIL_C_BINDING_HPP
#define BAOBZI_DETAIL_C_BINDING_HPP

/// \file baobzi/detail/c_binding.hpp
/// \brief Internal glue between the extern "C" surface (baobzi.h) and the
///        C++ template library. Defines the type-erased evaluator hierarchy
///        (`IEval<T>` + `EvalImpl<T,Deg,IN,OUT,Policy>`), the C-callback ->
///        std::function wrappers, the baobzi_options_t -> baobzi::options
///        mapping, and the (degree, output_dim) dispatch used by the
///        per-(value_type, input_dim) factory translation units.
///
/// The mechanism (runtime tag + abstract base + explicit instantiation of a
/// fixed shape set) follows the pre-rewrite C binding, but the *surface* it
/// wraps is the new `baobzi::fit` / `baobzi::options` API — not the legacy
/// `baobzi_input_t`.

#include <array>
#include <cstddef>
#include <functional>
#include <type_traits>
#include <utility>

#include <baobzi.h>

#include <baobzi/baobzi.hpp>
#include <baobzi/detail/eval_policy.hpp>
#include <baobzi/detail/function.hpp>
#include <baobzi/detail/tol_kind.hpp>

namespace baobzi::capi {

// Supported shape set (mirrored in baobzi.h and the fit error messages):
//   degree     in {7, 11, 15}   — NCOEFFS = degree+1 in {8,12,16}, lane-aligned
//   input_dim  in {1, 2, 3}
//   output_dim in {1, 2, 3}
// The dispatch switch in `make_eval_impl` and the per-(dtype,dim) factory TUs
// are the single source of truth for which (degree, in, out) are instantiated.

/// Set the calling thread's last-error string (defined in src/capi/baobzi.cpp).
void set_last_error(const char *msg) noexcept;

/// Map the C options struct (or the defaults, when `opts == nullptr`) onto
/// the C++ fit knobs.
inline auto to_options(const baobzi_options_t *opts) -> baobzi::options {
    baobzi::options out{};
    if (opts == nullptr)
        return out;
    out.tol_kind               = static_cast<baobzi::TolKind>(opts->tol_kind);
    out.max_depth              = opts->max_depth;
    out.max_memory_mib         = opts->max_memory_mib;
    out.allow_max_depth_leaves = opts->allow_max_depth_leaves != 0;
    out.min_uniform_depth      = opts->min_uniform_depth;
    return out;
}

/// C callback pointer type for value type `T`.
template <class T>
using c_func_t = void (*)(const T *, T *, void *);

/// The std::function spelling baobzi::fit needs for shape (IN, OUT):
///   * scalar 1D->1D uses `T(T)` to keep polyfit's FuncEval scalar fast path;
///   * everything else is array-spelled `array<T,OUT>(array<T,IN>)`, which
///     routes through FuncEvalND. (1D vector-valued fits therefore spell the
///     input as `array<T,1>`, the workaround the library static_assert
///     directs callers to.)
template <class T, std::size_t IN, std::size_t OUT>
using fit_func_t =
    std::conditional_t<(IN == 1 && OUT == 1), std::function<T(T)>,
                       std::function<std::array<T, OUT>(std::array<T, IN>)>>;

/// Wrap the C callback into the std::function fit expects.
template <class T, std::size_t IN, std::size_t OUT>
auto wrap_callback(c_func_t<T> f, void *data) -> fit_func_t<T, IN, OUT> {
    if constexpr (IN == 1 && OUT == 1) {
        return [f, data](T x) -> T {
            T y;
            f(&x, &y, data);
            return y;
        };
    } else {
        return [f, data](std::array<T, IN> x) -> std::array<T, OUT> {
            std::array<T, OUT> y;
            f(x.data(), y.data(), data);
            return y;
        };
    }
}

/// Runtime-polymorphic evaluator interface for value type `T`. One concrete
/// `EvalImpl` per supported (Degree, IN, OUT, Policy) implements it; the C
/// shims hold an `IEval<T>*` behind the opaque handle.
template <class T>
struct IEval {
    virtual ~IEval() = default;
    /// Single point: `x` has input_dim coords, `y` has output_dim slots.
    virtual void eval(const T *x, T *y) const = 0;
    /// AoS batch: `x` is n*input_dim, `res` is n*output_dim.
    virtual void eval_multi(const T *x, T *res, std::size_t n) const = 0;
    /// Sorted 1D batch; sets last_error and no-ops for input_dim != 1.
    virtual void eval_sorted(const T *x, T *res, std::size_t n) const = 0;
    /// SoA batch; sets last_error and no-ops for output_dim == 1.
    virtual void eval_multi_soa(const T *x, T *const *soa,
                                std::size_t n) const = 0;
    [[nodiscard]] virtual auto memory_usage() const -> std::size_t = 0;
    virtual void print_stats() const = 0;
};

/// Concrete evaluator owning a fitted `baobzi::Function` of one fixed shape.
template <class T, int Deg, std::size_t IN, std::size_t OUT, EvalPolicy Policy>
struct EvalImpl final : IEval<T> {
    static constexpr bool scalar = (IN == 1 && OUT == 1);
    using func_t   = fit_func_t<T, IN, OUT>;
    using domain_t = std::conditional_t<scalar, T, std::array<T, IN>>;
    using fn_t = baobzi::Function<static_cast<std::size_t>(Deg), func_t, Policy>;

    explicit EvalImpl(fn_t fn) : fn_(std::move(fn)) {}

    /// Fit and wrap. May throw baobzi fit exceptions — the caller (the
    /// extern "C" shim) catches them and converts to NULL + last_error.
    static auto create(c_func_t<T> f, void *data, const T *a, const T *b,
                       double tol, const baobzi::options &opts) -> IEval<T> * {
        domain_t lo{};
        domain_t hi{};
        if constexpr (scalar) {
            lo = a[0];
            hi = b[0];
        } else {
            for (std::size_t i = 0; i < IN; ++i) {
                lo[i] = a[i];
                hi[i] = b[i];
            }
        }
        return new EvalImpl(baobzi::fit<static_cast<std::size_t>(Deg), Policy>(
            wrap_callback<T, IN, OUT>(f, data), lo, hi, tol, opts));
    }

    void eval(const T *x, T *y) const override {
        typename fn_t::input_type xi{};
        if constexpr (scalar) {
            xi = x[0];
        } else {
            for (std::size_t i = 0; i < IN; ++i) xi[i] = x[i];
        }
        const auto out = fn_(xi);
        if constexpr (scalar) {
            y[0] = out;
        } else {
            for (std::size_t j = 0; j < OUT; ++j) y[j] = out[j];
        }
    }

    void eval_multi(const T *x, T *res, std::size_t n) const override {
        fn_(x, res, n);
    }

    void eval_sorted(const T *x, T *res, std::size_t n) const override {
        if constexpr (IN == 1) {
            fn_.sorted(x, res, n);
        } else {
            (void)x;
            (void)res;
            (void)n;
            set_last_error(
                "baobzi_eval_sorted: only input_dim == 1 is supported");
        }
    }

    void eval_multi_soa(const T *x, T *const *soa,
                        std::size_t n) const override {
        if constexpr (OUT > 1) {
            std::array<T *, OUT> bufs{};
            for (std::size_t j = 0; j < OUT; ++j) bufs[j] = soa[j];
            fn_(x, bufs, n);
        } else {
            (void)x;
            (void)soa;
            (void)n;
            set_last_error(
                "baobzi_eval_multi_soa: only output_dim > 1 is supported");
        }
    }

    [[nodiscard]] auto memory_usage() const -> std::size_t override {
        return fn_.memory_usage();
    }

    void print_stats() const override { fn_.print_stats(); }

  private:
    fn_t fn_;
};

/// (degree, output_dim) dispatch for a fixed (value_type, input_dim). Returns
/// nullptr when (degree, output_dim) is outside the supported set. Defined in
/// the header so each per-(T, IN) factory TU instantiates only its own slice.
template <class T, std::size_t IN>
auto make_eval_impl(int degree, int output_dim, c_func_t<T> f, void *data,
                    const T *a, const T *b, double tol,
                    const baobzi::options &opts) -> IEval<T> * {
    constexpr EvalPolicy kPolicy = EvalPolicy::Balanced;

    auto by_out = [&]<int Deg>() -> IEval<T> * {
        switch (output_dim) {
        case 1: return EvalImpl<T, Deg, IN, 1, kPolicy>::create(f, data, a, b, tol, opts);
        case 2: return EvalImpl<T, Deg, IN, 2, kPolicy>::create(f, data, a, b, tol, opts);
        case 3: return EvalImpl<T, Deg, IN, 3, kPolicy>::create(f, data, a, b, tol, opts);
        default: return nullptr;
        }
    };

    switch (degree) {
    case 7:  return by_out.template operator()<7>();
    case 11: return by_out.template operator()<11>();
    case 15: return by_out.template operator()<15>();
    default: return nullptr;
    }
}

/// Per-(value_type, input_dim) factory. Declared here, defined (one each) in
/// src/capi/dispatch_{f64,f32}_dim{1,2,3}.cpp so the heavy template
/// instantiations compile in parallel.
auto make_eval_f64_dim1(int degree, int output_dim, baobzi_func_f64_t f,
                        void *data, const double *a, const double *b,
                        double tol, const baobzi::options &opts) -> IEval<double> *;
auto make_eval_f64_dim2(int degree, int output_dim, baobzi_func_f64_t f,
                        void *data, const double *a, const double *b,
                        double tol, const baobzi::options &opts) -> IEval<double> *;
auto make_eval_f64_dim3(int degree, int output_dim, baobzi_func_f64_t f,
                        void *data, const double *a, const double *b,
                        double tol, const baobzi::options &opts) -> IEval<double> *;
auto make_eval_f32_dim1(int degree, int output_dim, baobzi_func_f32_t f,
                        void *data, const float *a, const float *b,
                        double tol, const baobzi::options &opts) -> IEval<float> *;
auto make_eval_f32_dim2(int degree, int output_dim, baobzi_func_f32_t f,
                        void *data, const float *a, const float *b,
                        double tol, const baobzi::options &opts) -> IEval<float> *;
auto make_eval_f32_dim3(int degree, int output_dim, baobzi_func_f32_t f,
                        void *data, const float *a, const float *b,
                        double tol, const baobzi::options &opts) -> IEval<float> *;

} // namespace baobzi::capi

#endif // BAOBZI_DETAIL_C_BINDING_HPP
