/* float32.c — single-precision (f32) fit and eval from C11.
 *
 * The handle carries a runtime dtype tag, so every value-carrying entry point
 * comes as an `_f64`/`_f32` twin: the callback, domain corners, and eval
 * buffers are all `float` here. Accuracy is checked at a float-appropriate
 * tolerance (f32 has ~7 significant digits). */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include <baobzi.h>

/* f(x) = exp(0.5 x) + sin(3 x) on [0, 1], in single precision. */
static void kernel(const float *x, float *y, void *data) {
    (void)data;
    y[0] = expf(0.5F * x[0]) + sinf(3.0F * x[0]);
}

int main(void) {
    const float a = 0.0F;
    const float b = 1.0F;

    /* tol is a double even on the f32 path (it matches baobzi::fit's signature);
     * 1e-5 is about as tight as single precision can meaningfully resolve. */
    baobzi_t fn = baobzi_fit_f32(kernel, NULL,
                                 /*input_dim=*/1, /*output_dim=*/1,
                                 /*degree=*/11, &a, &b, /*tol=*/1e-5,
                                 BAOBZI_BALANCED, NULL);
    if (fn == NULL) {
        fprintf(stderr, "baobzi_fit_f32 failed: %s\n", baobzi_last_error());
        return EXIT_FAILURE;
    }

    printf("dtype=%d (1 == F32) input_dim=%d output_dim=%d memory=%zu bytes\n",
           (int)baobzi_dtype(fn), baobzi_input_dim(fn), baobzi_output_dim(fn),
           baobzi_memory_usage(fn));

    /* Scalar eval at a few points. */
    float max_abs_err = 0.0F;
    for (int i = 0; i <= 20; ++i) {
        const float x = (float)i / 20.0F;
        float y = 0.0F;
        baobzi_eval_f32(fn, &x, &y);
        float exact = 0.0F;
        kernel(&x, &exact, NULL);
        const float err = fabsf(y - exact);
        if (err > max_abs_err) max_abs_err = err;
    }
    printf("max |approx - exact| (scalar) = %.3e\n", (double)max_abs_err);

    /* AoS batch eval. */
    const float xs[5] = {0.05F, 0.25F, 0.5F, 0.75F, 0.95F};
    float ys[5] = {0};
    baobzi_eval_multi_f32(fn, xs, ys, 5);
    float exact_mid = 0.0F;
    kernel(&xs[2], &exact_mid, NULL);
    printf("fn(0.5) approx = %.7f (exact %.7f)\n", (double)ys[2],
           (double)exact_mid);

    fn = baobzi_free(fn);
    return (max_abs_err < 1e-4F) ? EXIT_SUCCESS : EXIT_FAILURE;
}
