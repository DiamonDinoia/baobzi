/// src/capi/baobzi.cpp — the extern "C" surface declared in baobzi.h.
///
/// Holds the opaque handle definition, the thread-local last-error buffer,
/// the fit dispatch over input_dim (delegating to the per-(value_type,
/// input_dim) factories compiled in the sibling dispatch_*.cpp TUs), and the
/// dtype-checked eval / introspection / lifetime shims. No C++ exception is
/// allowed to escape any function here.

#include <exception>
#include <string>

#include <baobzi.h>

#include <baobzi/detail/c_binding.hpp>

namespace {

/// Thread-local description of the most recent error on the calling thread.
auto error_buffer() -> std::string & {
    thread_local std::string buf;
    return buf;
}

void clear_error() noexcept { error_buffer().clear(); }

} // namespace

namespace baobzi::capi {

void set_last_error(const char *msg) noexcept {
    try {
        error_buffer() = (msg != nullptr) ? msg : "";
    } catch (...) {
        // A std::string assignment that throws (bad_alloc) leaves the prior
        // message; nothing else we can usefully do at the C boundary.
    }
}

} // namespace baobzi::capi

/// Opaque handle: a dtype tag plus a type-erased evaluator. `impl` is an
/// `IEval<double>*` when dtype == F64 and an `IEval<float>*` when F32.
struct baobzi_function {
    void          *impl;
    baobzi_dtype_t dtype;
    int            input_dim;
    int            output_dim;
    int            degree;
};

extern "C" {

const baobzi_options_t baobzi_default_options = {
    /* tol_kind               */ BAOBZI_RELATIVE_MAX,
    /* max_depth              */ 50,
    /* max_memory_mib         */ 4,
    /* allow_max_depth_leaves */ 0,
    /* min_uniform_depth      */ 0,
};

baobzi_t baobzi_fit_f64(baobzi_func_f64_t f, void *data, int input_dim,
                        int output_dim, int degree, const double *a,
                        const double *b, double tol,
                        baobzi_eval_policy_t policy,
                        const baobzi_options_t *opts) {
    using namespace baobzi::capi;
    (void)policy; // only the Balanced evaluator is instantiated today
    clear_error();
    if (f == nullptr || a == nullptr || b == nullptr) {
        set_last_error("baobzi_fit_f64: null callback or domain pointer");
        return nullptr;
    }

    const baobzi::options o = to_options(opts);
    IEval<double> *impl = nullptr;
    try {
        switch (input_dim) {
        case 1: impl = make_eval_f64_dim1(degree, output_dim, f, data, a, b, tol, o); break;
        case 2: impl = make_eval_f64_dim2(degree, output_dim, f, data, a, b, tol, o); break;
        case 3: impl = make_eval_f64_dim3(degree, output_dim, f, data, a, b, tol, o); break;
        default:
            set_last_error("baobzi_fit_f64: input_dim must be 1, 2, or 3");
            return nullptr;
        }
    } catch (const std::exception &e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("baobzi_fit_f64: unknown exception during fit");
        return nullptr;
    }

    if (impl == nullptr) {
        set_last_error("baobzi_fit_f64: unsupported (degree, output_dim); "
                       "degree must be 7, 11, or 15 and output_dim 1, 2, or 3");
        return nullptr;
    }
    return new baobzi_function{impl, BAOBZI_F64, input_dim, output_dim, degree};
}

baobzi_t baobzi_fit_f32(baobzi_func_f32_t f, void *data, int input_dim,
                        int output_dim, int degree, const float *a,
                        const float *b, double tol,
                        baobzi_eval_policy_t policy,
                        const baobzi_options_t *opts) {
    using namespace baobzi::capi;
    (void)policy;
    clear_error();
    if (f == nullptr || a == nullptr || b == nullptr) {
        set_last_error("baobzi_fit_f32: null callback or domain pointer");
        return nullptr;
    }

    const baobzi::options o = to_options(opts);
    IEval<float> *impl = nullptr;
    try {
        switch (input_dim) {
        case 1: impl = make_eval_f32_dim1(degree, output_dim, f, data, a, b, tol, o); break;
        case 2: impl = make_eval_f32_dim2(degree, output_dim, f, data, a, b, tol, o); break;
        case 3: impl = make_eval_f32_dim3(degree, output_dim, f, data, a, b, tol, o); break;
        default:
            set_last_error("baobzi_fit_f32: input_dim must be 1, 2, or 3");
            return nullptr;
        }
    } catch (const std::exception &e) {
        set_last_error(e.what());
        return nullptr;
    } catch (...) {
        set_last_error("baobzi_fit_f32: unknown exception during fit");
        return nullptr;
    }

    if (impl == nullptr) {
        set_last_error("baobzi_fit_f32: unsupported (degree, output_dim); "
                       "degree must be 7, 11, or 15 and output_dim 1, 2, or 3");
        return nullptr;
    }
    return new baobzi_function{impl, BAOBZI_F32, input_dim, output_dim, degree};
}

} // extern "C"

/// Shared dtype guard for the eval shims. Returns the typed evaluator or
/// nullptr (after setting last_error) on a null handle or dtype mismatch.
namespace {
template <class T>
auto checked_impl(baobzi_t f, baobzi_dtype_t want) -> baobzi::capi::IEval<T> * {
    clear_error();
    if (f == nullptr) {
        baobzi::capi::set_last_error("baobzi eval: null handle");
        return nullptr;
    }
    if (f->dtype != want) {
        baobzi::capi::set_last_error(
            want == BAOBZI_F64
                ? "baobzi eval: dtype mismatch (called _f64 on an f32 handle)"
                : "baobzi eval: dtype mismatch (called _f32 on an f64 handle)");
        return nullptr;
    }
    return static_cast<baobzi::capi::IEval<T> *>(f->impl);
}
} // namespace

extern "C" {

void baobzi_eval_f64(baobzi_t f, const double *x, double *y) {
    if (auto *impl = checked_impl<double>(f, BAOBZI_F64)) impl->eval(x, y);
}
void baobzi_eval_f32(baobzi_t f, const float *x, float *y) {
    if (auto *impl = checked_impl<float>(f, BAOBZI_F32)) impl->eval(x, y);
}

void baobzi_eval_multi_f64(baobzi_t f, const double *x, double *res,
                           size_t n) {
    if (auto *impl = checked_impl<double>(f, BAOBZI_F64))
        impl->eval_multi(x, res, n);
}
void baobzi_eval_multi_f32(baobzi_t f, const float *x, float *res, size_t n) {
    if (auto *impl = checked_impl<float>(f, BAOBZI_F32))
        impl->eval_multi(x, res, n);
}

void baobzi_eval_sorted_f64(baobzi_t f, const double *x, double *res,
                            size_t n) {
    if (auto *impl = checked_impl<double>(f, BAOBZI_F64))
        impl->eval_sorted(x, res, n);
}
void baobzi_eval_sorted_f32(baobzi_t f, const float *x, float *res, size_t n) {
    if (auto *impl = checked_impl<float>(f, BAOBZI_F32))
        impl->eval_sorted(x, res, n);
}

void baobzi_eval_multi_soa_f64(baobzi_t f, const double *x, double *const *soa,
                               size_t n) {
    if (auto *impl = checked_impl<double>(f, BAOBZI_F64))
        impl->eval_multi_soa(x, soa, n);
}
void baobzi_eval_multi_soa_f32(baobzi_t f, const float *x, float *const *soa,
                               size_t n) {
    if (auto *impl = checked_impl<float>(f, BAOBZI_F32))
        impl->eval_multi_soa(x, soa, n);
}

baobzi_dtype_t baobzi_dtype(baobzi_t f) { return f->dtype; }
int            baobzi_input_dim(baobzi_t f) { return f->input_dim; }
int            baobzi_output_dim(baobzi_t f) { return f->output_dim; }

size_t baobzi_memory_usage(baobzi_t f) {
    if (f == nullptr) return 0;
    if (f->dtype == BAOBZI_F64)
        return static_cast<baobzi::capi::IEval<double> *>(f->impl)->memory_usage();
    return static_cast<baobzi::capi::IEval<float> *>(f->impl)->memory_usage();
}

void baobzi_print_stats(baobzi_t f) {
    if (f == nullptr) return;
    if (f->dtype == BAOBZI_F64)
        static_cast<baobzi::capi::IEval<double> *>(f->impl)->print_stats();
    else
        static_cast<baobzi::capi::IEval<float> *>(f->impl)->print_stats();
}

baobzi_t baobzi_free(baobzi_t f) {
    if (f == nullptr) return nullptr;
    if (f->dtype == BAOBZI_F64)
        delete static_cast<baobzi::capi::IEval<double> *>(f->impl);
    else
        delete static_cast<baobzi::capi::IEval<float> *>(f->impl);
    delete f;
    return nullptr;
}

const char *baobzi_last_error(void) { return error_buffer().c_str(); }

} // extern "C"
