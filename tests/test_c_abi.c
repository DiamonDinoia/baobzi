/* test_c_abi.c — pure-C conformance test for the baobzi C ABI (baobzi.h).
 *
 * Unlike tests/test_c.cpp (a C++ TU that checks the C API against a direct
 * baobzi::fit reference), this is compiled by the *C* compiler as C11 and
 * cannot call into C++ at all — so it exercises every entry point exactly as a
 * downstream C / Fortran consumer would, catching ABI and language-linkage
 * problems the C++ parity test cannot.
 *
 * Because C cannot reach baobzi::fit, correctness is checked two ways:
 *   - closed-form: eval(x) is within a generous margin of the exact kernel;
 *   - self-consistency: two paths on the same handle agree (often bit-exact).
 *
 * A tiny CHECK macro counts failures, prints each, and main() returns that
 * count — so 0 means pass and the process exit code is the failure count. */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <baobzi.h>

/* ---- failure-counting harness --------------------------------------- */

static int g_failures = 0;

static void check_impl(int cond, const char *expr, const char *file, int line) {
    if (!cond) {
        fprintf(stderr, "FAIL %s:%d: %s\n", file, line, expr);
        ++g_failures;
    }
}
#define CHECK(cond) check_impl((cond) ? 1 : 0, #cond, __FILE__, __LINE__)

/* ---- kernels and their closed forms (smooth, O(1) on the unit box) --- */

static void k_1d_1(const double *x, double *y, void *d) {
    (void)d;
    y[0] = exp(0.5 * x[0]) + sin(3.0 * x[0]);
}
static double exact_1d_1(double x) { return exp(0.5 * x) + sin(3.0 * x); }

static void k_1d_2(const double *x, double *y, void *d) {
    (void)d;
    y[0] = exp(0.5 * x[0]);
    y[1] = sin(3.0 * x[0]) + 2.0;
}

static void k_2d_1(const double *x, double *y, void *d) {
    (void)d;
    y[0] = exp(0.3 * x[0]) + sin(2.0 * x[1]);
}
static double exact_2d_1(double x0, double x1) {
    return exp(0.3 * x0) + sin(2.0 * x1);
}

static void k_2d_3(const double *x, double *y, void *d) {
    (void)d;
    y[0] = exp(0.3 * x[0]);
    y[1] = sin(2.0 * x[1]) + 2.0;
    y[2] = cos(x[0] * x[1]) + 2.0;
}

static void k_3d_1(const double *x, double *y, void *d) {
    (void)d;
    y[0] = exp(0.2 * x[0]) + sin(x[1]) + cos(x[2]);
}
static double exact_3d_1(double x0, double x1, double x2) {
    return exp(0.2 * x0) + sin(x1) + cos(x2);
}

static void k_1d_1f(const float *x, float *y, void *d) {
    (void)d;
    y[0] = expf(0.5F * x[0]) + sinf(3.0F * x[0]);
}
static float exact_1d_1f(float x) { return expf(0.5F * x) + sinf(3.0F * x); }

/* A deterministic LCG scatter in [0, 1); no <stdlib.h> rand() dependence. */
static double next_unit(unsigned int *state) {
    *state = *state * 1103515245U + 12345U;
    return (double)(*state >> 8) / (double)(1U << 24);
}

/* ---- 1D scalar: all degrees, eval parity, multi parity -------------- */

static void test_1d_scalar_all_degrees(void) {
    const int degrees[3] = {7, 11, 15};
    const double a = 0.0, b = 1.0;
    for (int di = 0; di < 3; ++di) {
        const int deg = degrees[di];
        baobzi_t h = baobzi_fit_f64(k_1d_1, NULL, 1, 1, deg, &a, &b, 1e-9,
                                    BAOBZI_BALANCED, NULL);
        CHECK(h != NULL);
        if (h == NULL) continue;
        CHECK(baobzi_dtype(h) == BAOBZI_F64);
        CHECK(baobzi_input_dim(h) == 1);
        CHECK(baobzi_output_dim(h) == 1);
        CHECK(baobzi_memory_usage(h) > 0);

        enum { N = 256 };
        double xs[N], multi[N];
        unsigned int seed = 1U + (unsigned int)deg;
        for (int i = 0; i < N; ++i) xs[i] = next_unit(&seed);
        baobzi_eval_multi_f64(h, xs, multi, N);

        double max_err = 0.0, max_parity = 0.0;
        for (int i = 0; i < N; ++i) {
            double y = 0.0;
            baobzi_eval_f64(h, &xs[i], &y);
            const double ce = fabs(y - exact_1d_1(xs[i]));
            if (ce > max_err) max_err = ce;
            const double pe = fabs(y - multi[i]);
            if (pe > max_parity) max_parity = pe;
        }
        CHECK(max_err < 1e-5);    /* closed-form, generous vs fit tol */
        CHECK(max_parity == 0.0); /* scalar and multi share the evaluator */
        baobzi_free(h);
    }
}

/* ---- sorted fast path == general multi on sorted input -------------- */

static int cmp_double(const void *pa, const void *pb) {
    const double a = *(const double *)pa, b = *(const double *)pb;
    return (a > b) - (a < b);
}

static void test_sorted_matches_multi(void) {
    const double a = 0.0, b = 1.0;
    baobzi_t h = baobzi_fit_f64(k_1d_1, NULL, 1, 1, 11, &a, &b, 1e-10,
                                BAOBZI_BALANCED, NULL);
    CHECK(h != NULL);
    if (h == NULL) return;

    enum { N = 300 };
    double xs[N], multi[N], sorted[N];
    unsigned int seed = 99U;
    for (int i = 0; i < N; ++i) xs[i] = next_unit(&seed);
    qsort(xs, N, sizeof(double), cmp_double);
    baobzi_eval_multi_f64(h, xs, multi, N);
    baobzi_eval_sorted_f64(h, xs, sorted, N);
    for (int i = 0; i < N; ++i) CHECK(sorted[i] == multi[i]);
    baobzi_free(h);
}

/* ---- SoA == AoS, component by component ----------------------------- */

static void test_soa_matches_aos(void) {
    const double a[2] = {0.0, 0.0}, b[2] = {1.0, 1.0};
    baobzi_t h = baobzi_fit_f64(k_2d_3, NULL, 2, 3, 11, a, b, 1e-8,
                                BAOBZI_BALANCED, NULL);
    CHECK(h != NULL);
    if (h == NULL) return;
    CHECK(baobzi_output_dim(h) == 3);

    enum { N = 128 };
    double xs[N * 2], aos[N * 3];
    double c0[N], c1[N], c2[N];
    double *soa[3] = {c0, c1, c2};
    unsigned int seed = 7U;
    for (int i = 0; i < N * 2; ++i) xs[i] = next_unit(&seed);
    baobzi_eval_multi_f64(h, xs, aos, N);
    baobzi_eval_multi_soa_f64(h, xs, soa, N);

    double max_err = 0.0;
    for (int i = 0; i < N; ++i) {
        CHECK(c0[i] == aos[3 * i + 0]);
        CHECK(c1[i] == aos[3 * i + 1]);
        CHECK(c2[i] == aos[3 * i + 2]);
        double exact[3] = {0};
        k_2d_3(&xs[2 * i], exact, NULL);
        for (int k = 0; k < 3; ++k) {
            const double e = fabs(aos[3 * i + k] - exact[k]);
            if (e > max_err) max_err = e;
        }
    }
    CHECK(max_err < 1e-5);
    baobzi_free(h);
}

/* ---- 2D / 3D scalar + 1D vector closed-form parity ------------------ */

static void test_higher_dim_fits(void) {
    /* 2D -> 1D */
    {
        const double a[2] = {0.0, 0.0}, b[2] = {1.0, 1.0};
        baobzi_t h = baobzi_fit_f64(k_2d_1, NULL, 2, 1, 11, a, b, 1e-8,
                                    BAOBZI_BALANCED, NULL);
        CHECK(h != NULL);
        if (h != NULL) {
            CHECK(baobzi_input_dim(h) == 2);
            double max_err = 0.0;
            unsigned int seed = 3U;
            for (int i = 0; i < 200; ++i) {
                const double p[2] = {next_unit(&seed), next_unit(&seed)};
                double y = 0.0;
                baobzi_eval_f64(h, p, &y);
                const double e = fabs(y - exact_2d_1(p[0], p[1]));
                if (e > max_err) max_err = e;
            }
            CHECK(max_err < 1e-5);
            baobzi_free(h);
        }
    }
    /* 3D -> 1D */
    {
        const double a[3] = {0.0, 0.0, 0.0}, b[3] = {1.0, 1.0, 1.0};
        baobzi_t h = baobzi_fit_f64(k_3d_1, NULL, 3, 1, 11, a, b, 1e-7,
                                    BAOBZI_BALANCED, NULL);
        CHECK(h != NULL);
        if (h != NULL) {
            CHECK(baobzi_input_dim(h) == 3);
            double max_err = 0.0;
            unsigned int seed = 5U;
            for (int i = 0; i < 200; ++i) {
                const double p[3] = {next_unit(&seed), next_unit(&seed),
                                     next_unit(&seed)};
                double y = 0.0;
                baobzi_eval_f64(h, p, &y);
                const double e = fabs(y - exact_3d_1(p[0], p[1], p[2]));
                if (e > max_err) max_err = e;
            }
            CHECK(max_err < 1e-5);
            baobzi_free(h);
        }
    }
    /* 1D -> 2 vector (input spelled as 1D, output_dim == 2) */
    {
        const double a = 0.0, b = 1.0;
        baobzi_t h = baobzi_fit_f64(k_1d_2, NULL, 1, 2, 11, &a, &b, 1e-9,
                                    BAOBZI_BALANCED, NULL);
        CHECK(h != NULL);
        if (h != NULL) {
            CHECK(baobzi_output_dim(h) == 2);
            double max_err = 0.0;
            unsigned int seed = 11U;
            for (int i = 0; i < 200; ++i) {
                const double x = next_unit(&seed);
                double y[2] = {0};
                baobzi_eval_f64(h, &x, y);
                double exact[2] = {0};
                k_1d_2(&x, exact, NULL);
                if (fabs(y[0] - exact[0]) > max_err) max_err = fabs(y[0] - exact[0]);
                if (fabs(y[1] - exact[1]) > max_err) max_err = fabs(y[1] - exact[1]);
            }
            CHECK(max_err < 1e-5);
            baobzi_free(h);
        }
    }
}

/* ---- f32 path ------------------------------------------------------- */

static void test_f32_path(void) {
    const float a = 0.0F, b = 1.0F;
    baobzi_t h = baobzi_fit_f32(k_1d_1f, NULL, 1, 1, 11, &a, &b, 1e-5,
                                BAOBZI_BALANCED, NULL);
    CHECK(h != NULL);
    if (h == NULL) return;
    CHECK(baobzi_dtype(h) == BAOBZI_F32);

    enum { N = 128 };
    float xs[N], multi[N];
    for (int i = 0; i < N; ++i) xs[i] = (float)i / (float)N;
    baobzi_eval_multi_f32(h, xs, multi, N);

    float max_err = 0.0F, max_parity = 0.0F;
    for (int i = 0; i < N; ++i) {
        float y = 0.0F;
        baobzi_eval_f32(h, &xs[i], &y);
        const float ce = fabsf(y - exact_1d_1f(xs[i]));
        if (ce > max_err) max_err = ce;
        const float pe = fabsf(y - multi[i]);
        if (pe > max_parity) max_parity = pe;
    }
    CHECK(max_err < 1e-3F);
    CHECK(max_parity == 0.0F);
    baobzi_free(h);
}

/* ---- out-of-domain -> NaN (scalar and multi) ------------------------ */

static void test_out_of_domain_nan(void) {
    const double a = 0.0, b = 1.0;
    baobzi_t h = baobzi_fit_f64(k_1d_1, NULL, 1, 1, 11, &a, &b, 1e-9,
                                BAOBZI_BALANCED, NULL);
    CHECK(h != NULL);
    if (h == NULL) return;

    double xo = b + 1.0, y = 0.0;
    baobzi_eval_f64(h, &xo, &y);
    CHECK(isnan(y));

    double xs[3] = {0.25, b + 2.0, 0.75};
    double ys[3] = {0};
    baobzi_eval_multi_f64(h, xs, ys, 3);
    CHECK(!isnan(ys[0]));
    CHECK(isnan(ys[1]));
    CHECK(!isnan(ys[2]));
    baobzi_free(h);
}

/* ---- introspection / defaults --------------------------------------- */

static void test_introspection_and_defaults(void) {
    /* baobzi_default_options must match baobzi::options{} (verified in the
     * C++ surface; mirrored here as concrete values). */
    CHECK(baobzi_default_options.tol_kind == BAOBZI_RELATIVE_MAX);
    CHECK(baobzi_default_options.max_depth == 50);
    CHECK(baobzi_default_options.max_memory_mib == 4);
    CHECK(baobzi_default_options.allow_max_depth_leaves == 0);
    CHECK(baobzi_default_options.min_uniform_depth == 0);

    /* NULL opts path must succeed (it selects the defaults). */
    const double a = 0.0, b = 1.0;
    baobzi_t h = baobzi_fit_f64(k_1d_1, NULL, 1, 1, 11, &a, &b, 1e-9,
                                BAOBZI_BALANCED, NULL);
    CHECK(h != NULL);
    if (h == NULL) return;
    CHECK(baobzi_memory_usage(h) > 0);
    baobzi_print_stats(h); /* must run without crashing */
    baobzi_free(h);
}

/* ---- error paths ---------------------------------------------------- */

static void test_error_paths(void) {
    const double a = 0.0, b = 1.0;

    /* unsupported degree (8) -> NULL + nonempty error */
    {
        baobzi_t h = baobzi_fit_f64(k_1d_1, NULL, 1, 1, 8, &a, &b, 1e-9,
                                    BAOBZI_BALANCED, NULL);
        CHECK(h == NULL);
        CHECK(strlen(baobzi_last_error()) > 0);
    }
    /* unsupported input_dim (4) -> NULL + error mentioning input_dim */
    {
        const double a4[4] = {0, 0, 0, 0}, b4[4] = {1, 1, 1, 1};
        baobzi_t h = baobzi_fit_f64(k_1d_1, NULL, 4, 1, 11, a4, b4, 1e-9,
                                    BAOBZI_BALANCED, NULL);
        CHECK(h == NULL);
        CHECK(strstr(baobzi_last_error(), "input_dim") != NULL);
    }
    /* tol <= 0 -> NULL + nonempty error */
    {
        baobzi_t h = baobzi_fit_f64(k_1d_1, NULL, 1, 1, 11, &a, &b, 0.0,
                                    BAOBZI_BALANCED, NULL);
        CHECK(h == NULL);
        CHECK(strlen(baobzi_last_error()) > 0);
    }
    /* dtype mismatch: _f32 eval on an f64 handle leaves output untouched */
    {
        baobzi_t h = baobzi_fit_f64(k_1d_1, NULL, 1, 1, 11, &a, &b, 1e-9,
                                    BAOBZI_BALANCED, NULL);
        CHECK(h != NULL);
        if (h != NULL) {
            float x = 0.5F, y = 42.0F;
            baobzi_eval_f32(h, &x, &y);
            CHECK(y == 42.0F);
            CHECK(strlen(baobzi_last_error()) > 0);
            baobzi_free(h);
        }
    }
    /* sorted on a 2D handle -> no write + error mentioning input_dim */
    {
        const double a2[2] = {0.0, 0.0}, b2[2] = {1.0, 1.0};
        baobzi_t h = baobzi_fit_f64(k_2d_1, NULL, 2, 1, 11, a2, b2, 1e-8,
                                    BAOBZI_BALANCED, NULL);
        CHECK(h != NULL);
        if (h != NULL) {
            double y = 123.0;
            baobzi_eval_sorted_f64(h, a2, &y, 1);
            CHECK(y == 123.0);
            CHECK(strstr(baobzi_last_error(), "input_dim") != NULL);
            baobzi_free(h);
        }
    }
    /* SoA on an output_dim == 1 handle -> no write + error mentioning output_dim */
    {
        baobzi_t h = baobzi_fit_f64(k_1d_1, NULL, 1, 1, 11, &a, &b, 1e-9,
                                    BAOBZI_BALANCED, NULL);
        CHECK(h != NULL);
        if (h != NULL) {
            double x = 0.5, c0 = 7.0;
            double *soa[1] = {&c0};
            baobzi_eval_multi_soa_f64(h, &x, soa, 1);
            CHECK(c0 == 7.0);
            CHECK(strstr(baobzi_last_error(), "output_dim") != NULL);
            baobzi_free(h);
        }
    }
    /* null-handle eval -> no crash + nonempty error */
    {
        double x = 0.0, y = 0.0;
        baobzi_eval_f64(NULL, &x, &y);
        CHECK(strlen(baobzi_last_error()) > 0);
    }
    /* free(NULL) is a no-op returning NULL */
    CHECK(baobzi_free(NULL) == NULL);
}

int main(void) {
    test_1d_scalar_all_degrees();
    test_sorted_matches_multi();
    test_soa_matches_aos();
    test_higher_dim_fits();
    test_f32_path();
    test_out_of_domain_nan();
    test_introspection_and_defaults();
    test_error_paths();

    printf("%d failures\n", g_failures);
    return g_failures;
}
