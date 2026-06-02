/* with_options.c — custom fit knobs and error reporting from C11.
 *
 * Mirrors examples/c++/with_options.cpp. Shows how to populate a
 * `baobzi_options_t` (instead of passing NULL for the defaults), how to read
 * back fit statistics, and how a fit that cannot converge under a tight
 * `max_depth` reports failure: it returns NULL and leaves a message in
 * `baobzi_last_error()`. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <baobzi.h>

/* sin(50 x) / x on [0.01, 5): oscillatory and nearly singular near 0, so it
 * needs real subdivision depth to resolve. */
static void kernel(const double *x, double *y, void *data) {
    (void)data;
    y[0] = sin(50.0 * x[0]) / x[0];
}

int main(void) {
    const double a = 0.01;
    const double b = 5.0;

    /* Start from the documented defaults, then override a few knobs. Copying
     * baobzi_default_options keeps any field we do not set at its default. */
    baobzi_options_t opts = baobzi_default_options;
    opts.tol_kind          = BAOBZI_ABSOLUTE_MAX;
    opts.max_depth         = 50;
    opts.min_uniform_depth = 2;

    baobzi_t fn = baobzi_fit_f64(kernel, NULL,
                                 /*input_dim=*/1, /*output_dim=*/1,
                                 /*degree=*/11, &a, &b, /*tol=*/1e-8,
                                 BAOBZI_BALANCED, &opts);
    if (fn == NULL) {
        fprintf(stderr, "fit with custom options failed: %s\n",
                baobzi_last_error());
        return EXIT_FAILURE;
    }

    printf("fit converged; memory = %zu bytes\n", baobzi_memory_usage(fn));
    printf("--- baobzi_print_stats ---\n");
    baobzi_print_stats(fn);

    const double x = 1.0;
    double y = 0.0;
    baobzi_eval_f64(fn, &x, &y);
    printf("fn(1.0) = %.12f  exact = %.12f\n", y, sin(50.0) / 1.0);

    fn = baobzi_free(fn);

    /* Now force a failure: the same kernel cannot be resolved to 1e-8 with a
     * max_depth of only 4, so the fit returns NULL and sets last_error. */
    baobzi_options_t shallow = baobzi_default_options;
    shallow.max_depth = 4;
    baobzi_t bad = baobzi_fit_f64(kernel, NULL, 1, 1, 11, &a, &b, 1e-8,
                                  BAOBZI_BALANCED, &shallow);
    if (bad != NULL) {
        fprintf(stderr, "expected the shallow fit to fail, but it succeeded\n");
        baobzi_free(bad);
        return EXIT_FAILURE;
    }
    printf("shallow (max_depth=4) fit failed as expected:\n  %s\n",
           baobzi_last_error());

    return EXIT_SUCCESS;
}
