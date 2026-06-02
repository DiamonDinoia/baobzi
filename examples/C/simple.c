/* simple.c — minimal baobzi C API usage, compiled as C11 (not C++).
 *
 * Doubles as the ABI smoke test: if this links and runs, the header is
 * C-clean and no C++ exception escaped the boundary. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <baobzi.h>

/* f(x) = exp(x) on [0, 1]. input_dim == output_dim == 1, so x and y each
 * point at a single value. */
static void kernel(const double *x, double *y, void *data) {
    (void)data;
    y[0] = exp(x[0]);
}

int main(void) {
    const double a = 0.0;
    const double b = 1.0;

    baobzi_t fn = baobzi_fit_f64(kernel, NULL,
                                 /*input_dim=*/1, /*output_dim=*/1,
                                 /*degree=*/11, &a, &b, /*tol=*/1e-10,
                                 BAOBZI_BALANCED, NULL);
    if (fn == NULL) {
        fprintf(stderr, "baobzi_fit_f64 failed: %s\n", baobzi_last_error());
        return EXIT_FAILURE;
    }

    printf("dtype=%d input_dim=%d output_dim=%d memory=%zu bytes\n",
           (int)baobzi_dtype(fn), baobzi_input_dim(fn), baobzi_output_dim(fn),
           baobzi_memory_usage(fn));

    /* Scalar eval at a few points; compare to the exact value. */
    double max_abs_err = 0.0;
    for (int i = 0; i <= 10; ++i) {
        const double x = (double)i / 10.0;
        double y = 0.0;
        baobzi_eval_f64(fn, &x, &y);
        const double err = fabs(y - exp(x));
        if (err > max_abs_err) max_abs_err = err;
    }
    printf("max |approx - exp| over 11 points: %.3e\n", max_abs_err);

    /* Batch (AoS) eval. */
    double xs[5] = {0.05, 0.25, 0.5, 0.75, 0.95};
    double ys[5] = {0};
    baobzi_eval_multi_f64(fn, xs, ys, 5);
    printf("exp(0.5) approx = %.12f (exact %.12f)\n", ys[2], exp(0.5));

    fn = baobzi_free(fn);
    return (max_abs_err < 1e-8) ? EXIT_SUCCESS : EXIT_FAILURE;
}
