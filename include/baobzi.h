/* baobzi.h — C ABI for the baobzi piecewise-Chebyshev approximator.
 *
 * This surface mirrors the C++ `baobzi::fit(f, a, b, tol, options)` API
 * (see include/baobzi/baobzi.hpp): the domain is given as its lower/upper
 * corners `a`/`b`, the fit knobs live in `baobzi_options_t`, and the leaf
 * degree / eval policy are explicit. It is *not* the legacy
 * `baobzi_input_t` / `baobzi_init` shape.
 *
 * Eval-only: a fitted function can be evaluated (scalar, AoS-multi,
 * sorted-1D, SoA-multi) but not serialized. There is one opaque handle
 * type carrying a runtime dtype tag (f64 or f32); every value-carrying
 * entry point comes as an `_f64` / `_f32` twin because the callback and
 * buffer element types differ per value type.
 *
 * Thread safety: once `baobzi_fit_*` returns a handle, its `baobzi_eval_*`
 * functions are safe to call concurrently from multiple threads provided
 * each call writes a disjoint output slice (the underlying C++ Function
 * allocates per-call scratch). `baobzi_last_error()` is thread-local.
 *
 * No C++ exception ever crosses this boundary: fit failures return NULL
 * and set `baobzi_last_error()`.
 */
#ifndef BAOBZI_H
#define BAOBZI_H

#include <stddef.h> /* size_t */

#ifdef __cplusplus
extern "C" {
#endif

/* Tolerance interpretation — numeric values match baobzi::TolKind. */
typedef enum {
    BAOBZI_RELATIVE_TAIL = 0, /* relative tail-coefficient estimate (1D) */
    BAOBZI_ABSOLUTE_TAIL = 1, /* absolute tail-coefficient estimate (1D) */
    BAOBZI_RELATIVE_MAX  = 2, /* sample-based, max-abs relative error    */
    BAOBZI_ABSOLUTE_MAX  = 3, /* sample-based, max-abs absolute error    */
    BAOBZI_RELATIVE_L2   = 4, /* sample-based, L2 relative error         */
    BAOBZI_ABSOLUTE_L2   = 5  /* sample-based, L2 absolute error         */
} baobzi_tol_kind_t;

/* Eval-strategy knob — numeric values match baobzi::EvalPolicy.
 *
 * All three policies currently generate identical (Horner) eval code, so
 * the choice is presently a no-op carried for forward compatibility; only
 * the Balanced evaluator is instantiated and any value maps to it. */
typedef enum {
    BAOBZI_LATENCY    = 0,
    BAOBZI_THROUGHPUT = 1,
    BAOBZI_BALANCED   = 2
} baobzi_eval_policy_t;

/* Value type carried by a handle. */
typedef enum { BAOBZI_F64 = 0, BAOBZI_F32 = 1 } baobzi_dtype_t;

/* Fit knobs — mirrors baobzi::options. `int` is used for the bool field so
 * the struct is plain C. */
typedef struct {
    baobzi_tol_kind_t tol_kind;
    int               max_depth;
    int               max_memory_mib;
    int               allow_max_depth_leaves;
    int               min_uniform_depth;
} baobzi_options_t;

/* Defaults matching baobzi::options{}. Pass NULL for `opts` to use these. */
extern const baobzi_options_t baobzi_default_options;

/* Opaque, dtype-tagged handle. */
typedef struct baobzi_function *baobzi_t;

/* User callback: read `input_dim` coordinates from `x`, write `output_dim`
 * results to `y`. `data` is the opaque pointer passed to `baobzi_fit_*`. */
typedef void (*baobzi_func_f64_t)(const double *x, double *y, void *data);
typedef void (*baobzi_func_f32_t)(const float  *x, float  *y, void *data);

/* ---- fit -------------------------------------------------------------- *
 * `a`/`b` are the domain's lower/upper corners (`input_dim` elements each).
 * Returns NULL — and sets baobzi_last_error() — when the (dtype, input_dim,
 * output_dim, degree) tuple is unsupported, the arguments are invalid, or
 * the fit throws (MaxDepthExceeded / MemoryBudgetExceeded).
 *
 * Supported: input_dim in {1,2,3}, output_dim in {1,2,3}, degree in
 * {7,11,15}. */
baobzi_t baobzi_fit_f64(baobzi_func_f64_t f, void *data,
                        int input_dim, int output_dim, int degree,
                        const double *a, const double *b, double tol,
                        baobzi_eval_policy_t policy,
                        const baobzi_options_t *opts);
baobzi_t baobzi_fit_f32(baobzi_func_f32_t f, void *data,
                        int input_dim, int output_dim, int degree,
                        const float *a, const float *b, double tol,
                        baobzi_eval_policy_t policy,
                        const baobzi_options_t *opts);

/* ---- eval ------------------------------------------------------------- *
 * Each typed eval validates that the handle's dtype matches; on mismatch it
 * sets baobzi_last_error() and returns without writing. Out-of-domain
 * points yield NaN. */

/* Single point: x has input_dim elements, y has output_dim elements. */
void baobzi_eval_f64(baobzi_t f, const double *x, double *y);
void baobzi_eval_f32(baobzi_t f, const float  *x, float  *y);

/* AoS batch: x is n*input_dim packed coords, res is n*output_dim. */
void baobzi_eval_multi_f64(baobzi_t f, const double *x, double *res, size_t n);
void baobzi_eval_multi_f32(baobzi_t f, const float  *x, float  *res, size_t n);

/* Sorted 1D batch (caller promises x[i] <= x[i+1]); input_dim must be 1.
 * res is n*output_dim. */
void baobzi_eval_sorted_f64(baobzi_t f, const double *x, double *res, size_t n);
void baobzi_eval_sorted_f32(baobzi_t f, const float  *x, float  *res, size_t n);

/* SoA batch: x is n*input_dim packed coords; soa[d] is output component d's
 * buffer (output_dim buffers of n elements each). output_dim must be > 1. */
void baobzi_eval_multi_soa_f64(baobzi_t f, const double *x,
                               double *const *soa, size_t n);
void baobzi_eval_multi_soa_f32(baobzi_t f, const float *x,
                               float *const *soa, size_t n);

/* ---- introspection / lifetime (dtype-independent) -------------------- */
baobzi_dtype_t baobzi_dtype(baobzi_t f);
int            baobzi_input_dim(baobzi_t f);
int            baobzi_output_dim(baobzi_t f);
size_t         baobzi_memory_usage(baobzi_t f);
void           baobzi_print_stats(baobzi_t f);

/* Frees the handle (NULL-safe) and returns NULL, so callers can write
 * `h = baobzi_free(h);`. */
baobzi_t baobzi_free(baobzi_t f);

/* Thread-local description of the most recent error on this thread, or an
 * empty string if none. The returned pointer is valid until the next
 * baobzi call on the same thread. */
const char *baobzi_last_error(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* BAOBZI_H */
