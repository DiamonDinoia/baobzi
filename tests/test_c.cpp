/// test_c.cpp — exercises the extern "C" surface (baobzi.h) and checks it
/// against a direct C++ `baobzi::fit` of the same kernel. The C API builds
/// the same Function internally, so parity must be exact up to floating
/// noise; the comparison validates the dispatch / buffer / dtype plumbing,
/// not the approximation math (that lives in test_cpp.cpp).

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <baobzi.h>

#include <baobzi/baobzi.hpp>

using Catch::Approx;

// Deterministic seeds: reproducible parity sweeps.
// NOLINTBEGIN(cert-msc51-cpp,cert-msc32-c)

namespace {

// --- Kernels, shared between the C callbacks and the C++ reference fits ---
// Smooth and comfortably nonzero on the unit-ish boxes used below.

template <class T> T k_1d_1(T x) { return std::exp(T(0.5) * x) + std::sin(T(3) * x); }

template <class T> std::array<T, 2> k_1d_2(std::array<T, 1> x) {
    return {std::exp(T(0.5) * x[0]), std::sin(T(3) * x[0]) + T(2)};
}

template <class T> std::array<T, 1> k_2d_1(std::array<T, 2> x) {
    return {std::exp(T(0.3) * x[0]) + std::sin(T(2) * x[1])};
}

template <class T> std::array<T, 3> k_2d_3(std::array<T, 2> x) {
    return {std::exp(T(0.3) * x[0]),
            std::sin(T(2) * x[1]) + T(2),
            std::cos(x[0] * x[1]) + T(2)};
}

template <class T> std::array<T, 1> k_3d_1(std::array<T, 3> x) {
    return {std::exp(T(0.2) * x[0]) + std::sin(x[1]) + std::cos(x[2])};
}

// --- C callbacks (extern "C" so the function-pointer language linkage
//     matches baobzi_func_*_t exactly under -Werror). ---
extern "C" {
void c_1d_1(const double *x, double *y, void *) { y[0] = k_1d_1<double>(x[0]); }
void c_1d_2(const double *x, double *y, void *) {
    auto r = k_1d_2<double>({x[0]});
    y[0] = r[0];
    y[1] = r[1];
}
void c_2d_1(const double *x, double *y, void *) {
    y[0] = k_2d_1<double>({x[0], x[1]})[0];
}
void c_2d_3(const double *x, double *y, void *) {
    auto r = k_2d_3<double>({x[0], x[1]});
    y[0] = r[0]; y[1] = r[1]; y[2] = r[2];
}
void c_3d_1(const double *x, double *y, void *) {
    y[0] = k_3d_1<double>({x[0], x[1], x[2]})[0];
}
void c_1d_1f(const float *x, float *y, void *) { y[0] = k_1d_1<float>(x[0]); }
void c_2d_3f(const float *x, float *y, void *) {
    auto r = k_2d_3<float>({x[0], x[1]});
    y[0] = r[0]; y[1] = r[1]; y[2] = r[2];
}
} // extern "C"

constexpr int kDeg = 11;
constexpr std::size_t kNPts = 2000;

} // namespace

TEST_CASE("C API: 1D scalar parity + multi + sorted + OOD", "[c][1d]") {
    const double a = 0.0, b = 1.0;
    auto ref = baobzi::fit<kDeg>([](double x) { return k_1d_1<double>(x); }, a,
                                 b, 1e-10);
    baobzi_t h = baobzi_fit_f64(c_1d_1, nullptr, 1, 1, kDeg, &a, &b, 1e-10,
                                BAOBZI_BALANCED, nullptr);
    REQUIRE(h != nullptr);
    REQUIRE(baobzi_dtype(h) == BAOBZI_F64);
    REQUIRE(baobzi_input_dim(h) == 1);
    REQUIRE(baobzi_output_dim(h) == 1);

    std::mt19937 gen(1);
    std::uniform_real_distribution<double> d(a, b);
    std::vector<double> xs(kNPts);
    for (auto &x : xs) x = d(gen);

    SECTION("scalar eval matches C++ reference") {
        for (double x : xs) {
            double y = 0;
            baobzi_eval_f64(h, &x, &y);
            REQUIRE(y == Approx(ref(x)).epsilon(1e-12));
        }
    }
    SECTION("multi eval matches scalar") {
        std::vector<double> ys(kNPts, 0.0);
        baobzi_eval_multi_f64(h, xs.data(), ys.data(), kNPts);
        for (std::size_t i = 0; i < kNPts; ++i)
            REQUIRE(ys[i] == Approx(ref(xs[i])).epsilon(1e-12));
    }
    SECTION("sorted eval matches multi on sorted input") {
        std::vector<double> sx = xs;
        std::sort(sx.begin(), sx.end());
        std::vector<double> ym(kNPts, 0.0), ysort(kNPts, 0.0);
        baobzi_eval_multi_f64(h, sx.data(), ym.data(), kNPts);
        baobzi_eval_sorted_f64(h, sx.data(), ysort.data(), kNPts);
        for (std::size_t i = 0; i < kNPts; ++i)
            REQUIRE(ysort[i] == Approx(ym[i]).epsilon(1e-14));
    }
    SECTION("out-of-domain yields NaN") {
        double xo = b + 1.0, y = 0.0;
        baobzi_eval_f64(h, &xo, &y);
        REQUIRE(std::isnan(y));
    }
    h = baobzi_free(h);
    REQUIRE(h == nullptr);
}

TEST_CASE("C API: 1D vector output (SoA == AoS) + sorted", "[c][1d][vector]") {
    const double a = 0.0, b = 1.0;
    auto ref = baobzi::fit<kDeg>([](std::array<double, 1> x) {
        return k_1d_2<double>(x);
    }, std::array<double, 1>{a}, std::array<double, 1>{b}, 1e-9);
    baobzi_t h = baobzi_fit_f64(c_1d_2, nullptr, 1, 2, kDeg, &a, &b, 1e-9,
                                BAOBZI_BALANCED, nullptr);
    REQUIRE(h != nullptr);
    REQUIRE(baobzi_output_dim(h) == 2);

    std::mt19937 gen(2);
    std::uniform_real_distribution<double> d(a, b);
    std::vector<double> xs(kNPts);
    for (auto &x : xs) x = d(gen);

    std::vector<double> aos(kNPts * 2, 0.0);
    baobzi_eval_multi_f64(h, xs.data(), aos.data(), kNPts);
    for (std::size_t i = 0; i < kNPts; ++i) {
        auto r = ref(std::array<double, 1>{xs[i]});
        REQUIRE(aos[2 * i + 0] == Approx(r[0]).epsilon(1e-12));
        REQUIRE(aos[2 * i + 1] == Approx(r[1]).epsilon(1e-12));
    }

    SECTION("SoA matches AoS") {
        std::vector<double> c0(kNPts, 0.0), c1(kNPts, 0.0);
        double *soa[2] = {c0.data(), c1.data()};
        baobzi_eval_multi_soa_f64(h, xs.data(), soa, kNPts);
        for (std::size_t i = 0; i < kNPts; ++i) {
            REQUIRE(c0[i] == Approx(aos[2 * i + 0]).epsilon(1e-14));
            REQUIRE(c1[i] == Approx(aos[2 * i + 1]).epsilon(1e-14));
        }
    }
    SECTION("sorted matches multi (vector output)") {
        std::vector<double> sx = xs;
        std::sort(sx.begin(), sx.end());
        std::vector<double> ym(kNPts * 2, 0.0), ysort(kNPts * 2, 0.0);
        baobzi_eval_multi_f64(h, sx.data(), ym.data(), kNPts);
        baobzi_eval_sorted_f64(h, sx.data(), ysort.data(), kNPts);
        for (std::size_t i = 0; i < kNPts * 2; ++i)
            REQUIRE(ysort[i] == Approx(ym[i]).epsilon(1e-14));
    }
    baobzi_free(h);
}

TEST_CASE("C API: 2D scalar + 2D vector(SoA) + 3D scalar", "[c][2d][3d]") {
    SECTION("2D -> 1D parity") {
        const double a[2] = {0.0, 0.0}, b[2] = {1.0, 1.0};
        auto ref = baobzi::fit<kDeg>(
            [](std::array<double, 2> x) { return k_2d_1<double>(x); },
            std::array<double, 2>{a[0], a[1]}, std::array<double, 2>{b[0], b[1]},
            1e-8);
        baobzi_t h = baobzi_fit_f64(c_2d_1, nullptr, 2, 1, kDeg, a, b, 1e-8,
                                    BAOBZI_BALANCED, nullptr);
        REQUIRE(h != nullptr);

        std::mt19937 gen(3);
        std::uniform_real_distribution<double> d(0.0, 1.0);
        std::vector<double> xs(kNPts * 2);
        for (auto &x : xs) x = d(gen);
        std::vector<double> ys(kNPts, 0.0);
        baobzi_eval_multi_f64(h, xs.data(), ys.data(), kNPts);
        for (std::size_t i = 0; i < kNPts; ++i) {
            auto r = ref(std::array<double, 2>{xs[2 * i], xs[2 * i + 1]});
            REQUIRE(ys[i] == Approx(r[0]).epsilon(1e-11));
        }
        // sorted is 1D-only: must error and not write.
        double y = 123.0;
        baobzi_eval_sorted_f64(h, xs.data(), &y, 1);
        REQUIRE(std::string(baobzi_last_error()).find("input_dim") !=
                std::string::npos);
        baobzi_free(h);
    }
    SECTION("2D -> 3D SoA parity") {
        const double a[2] = {0.0, 0.0}, b[2] = {1.0, 1.0};
        auto ref = baobzi::fit<kDeg>(
            [](std::array<double, 2> x) { return k_2d_3<double>(x); },
            std::array<double, 2>{a[0], a[1]}, std::array<double, 2>{b[0], b[1]},
            1e-8);
        baobzi_t h = baobzi_fit_f64(c_2d_3, nullptr, 2, 3, kDeg, a, b, 1e-8,
                                    BAOBZI_BALANCED, nullptr);
        REQUIRE(h != nullptr);

        std::mt19937 gen(4);
        std::uniform_real_distribution<double> d(0.0, 1.0);
        std::vector<double> xs(kNPts * 2);
        for (auto &x : xs) x = d(gen);
        std::vector<double> aos(kNPts * 3, 0.0);
        baobzi_eval_multi_f64(h, xs.data(), aos.data(), kNPts);

        std::vector<double> c0(kNPts), c1(kNPts), c2(kNPts);
        double *soa[3] = {c0.data(), c1.data(), c2.data()};
        baobzi_eval_multi_soa_f64(h, xs.data(), soa, kNPts);
        for (std::size_t i = 0; i < kNPts; ++i) {
            auto r = ref(std::array<double, 2>{xs[2 * i], xs[2 * i + 1]});
            REQUIRE(aos[3 * i + 0] == Approx(r[0]).epsilon(1e-11));
            REQUIRE(c0[i] == Approx(aos[3 * i + 0]).epsilon(1e-14));
            REQUIRE(c1[i] == Approx(aos[3 * i + 1]).epsilon(1e-14));
            REQUIRE(c2[i] == Approx(aos[3 * i + 2]).epsilon(1e-14));
        }
        baobzi_free(h);
    }
    SECTION("3D -> 1D parity") {
        const double a[3] = {0.0, 0.0, 0.0}, b[3] = {1.0, 1.0, 1.0};
        auto ref = baobzi::fit<kDeg>(
            [](std::array<double, 3> x) { return k_3d_1<double>(x); },
            std::array<double, 3>{a[0], a[1], a[2]},
            std::array<double, 3>{b[0], b[1], b[2]}, 1e-7);
        baobzi_t h = baobzi_fit_f64(c_3d_1, nullptr, 3, 1, kDeg, a, b, 1e-7,
                                    BAOBZI_BALANCED, nullptr);
        REQUIRE(h != nullptr);

        std::mt19937 gen(5);
        std::uniform_real_distribution<double> d(0.0, 1.0);
        std::vector<double> xs(kNPts * 3);
        for (auto &x : xs) x = d(gen);
        std::vector<double> ys(kNPts, 0.0);
        baobzi_eval_multi_f64(h, xs.data(), ys.data(), kNPts);
        for (std::size_t i = 0; i < kNPts; ++i) {
            auto r = ref(std::array<double, 3>{xs[3 * i], xs[3 * i + 1],
                                               xs[3 * i + 2]});
            REQUIRE(ys[i] == Approx(r[0]).epsilon(1e-10));
        }
        baobzi_free(h);
    }
}

TEST_CASE("C API: float (f32) parity", "[c][f32]") {
    SECTION("1D scalar") {
        const float a = 0.0F, b = 1.0F;
        auto ref = baobzi::fit<kDeg>([](float x) { return k_1d_1<float>(x); }, a,
                                     b, 1e-5);
        baobzi_t h = baobzi_fit_f32(c_1d_1f, nullptr, 1, 1, kDeg, &a, &b, 1e-5,
                                    BAOBZI_BALANCED, nullptr);
        REQUIRE(h != nullptr);
        REQUIRE(baobzi_dtype(h) == BAOBZI_F32);

        std::mt19937 gen(6);
        std::uniform_real_distribution<float> d(a, b);
        std::vector<float> xs(kNPts);
        for (auto &x : xs) x = d(gen);
        std::vector<float> ys(kNPts, 0.0F);
        baobzi_eval_multi_f32(h, xs.data(), ys.data(), kNPts);
        for (std::size_t i = 0; i < kNPts; ++i)
            REQUIRE(ys[i] == Approx(ref(xs[i])).epsilon(1e-5));
        baobzi_free(h);
    }
    SECTION("2D -> 3D SoA") {
        const float a[2] = {0.0F, 0.0F}, b[2] = {1.0F, 1.0F};
        baobzi_t h = baobzi_fit_f32(c_2d_3f, nullptr, 2, 3, kDeg, a, b, 1e-5,
                                    BAOBZI_BALANCED, nullptr);
        REQUIRE(h != nullptr);
        std::mt19937 gen(7);
        std::uniform_real_distribution<float> d(0.0F, 1.0F);
        std::vector<float> xs(kNPts * 2);
        for (auto &x : xs) x = d(gen);
        std::vector<float> aos(kNPts * 3, 0.0F);
        baobzi_eval_multi_f32(h, xs.data(), aos.data(), kNPts);
        std::vector<float> c0(kNPts), c1(kNPts), c2(kNPts);
        float *soa[3] = {c0.data(), c1.data(), c2.data()};
        baobzi_eval_multi_soa_f32(h, xs.data(), soa, kNPts);
        for (std::size_t i = 0; i < kNPts; ++i) {
            REQUIRE(c0[i] == Approx(aos[3 * i + 0]).epsilon(1e-6));
            REQUIRE(c1[i] == Approx(aos[3 * i + 1]).epsilon(1e-6));
            REQUIRE(c2[i] == Approx(aos[3 * i + 2]).epsilon(1e-6));
        }
        baobzi_free(h);
    }
}

TEST_CASE("C API: error handling", "[c][errors]") {
    const double a = 0.0, b = 1.0;

    SECTION("dtype mismatch errors cleanly without writing") {
        baobzi_t h = baobzi_fit_f64(c_1d_1, nullptr, 1, 1, kDeg, &a, &b, 1e-10,
                                    BAOBZI_BALANCED, nullptr);
        REQUIRE(h != nullptr);
        float x = 0.5F, y = 42.0F;
        baobzi_eval_f32(h, &x, &y); // wrong dtype
        REQUIRE(y == 42.0F);        // untouched
        REQUIRE(std::strlen(baobzi_last_error()) > 0);
        baobzi_free(h);
    }
    SECTION("unsupported degree returns NULL + error") {
        baobzi_t h = baobzi_fit_f64(c_1d_1, nullptr, 1, 1, /*degree=*/8, &a, &b,
                                    1e-10, BAOBZI_BALANCED, nullptr);
        REQUIRE(h == nullptr);
        REQUIRE(std::strlen(baobzi_last_error()) > 0);
    }
    SECTION("unsupported input_dim returns NULL + error") {
        const double a4[4] = {0, 0, 0, 0}, b4[4] = {1, 1, 1, 1};
        baobzi_t h = baobzi_fit_f64(c_1d_1, nullptr, 4, 1, kDeg, a4, b4, 1e-10,
                                    BAOBZI_BALANCED, nullptr);
        REQUIRE(h == nullptr);
        REQUIRE(std::string(baobzi_last_error()).find("input_dim") !=
                std::string::npos);
    }
    SECTION("non-positive tol returns NULL + error") {
        baobzi_t h = baobzi_fit_f64(c_1d_1, nullptr, 1, 1, kDeg, &a, &b, 0.0,
                                    BAOBZI_BALANCED, nullptr);
        REQUIRE(h == nullptr);
        REQUIRE(std::strlen(baobzi_last_error()) > 0);
    }
    SECTION("default options struct is sane") {
        REQUIRE(baobzi_default_options.max_depth == 50);
        REQUIRE(baobzi_default_options.tol_kind == BAOBZI_RELATIVE_MAX);
    }
    SECTION("free(NULL) and eval(NULL) are safe") {
        REQUIRE(baobzi_free(nullptr) == nullptr);
        double x = 0.0, y = 0.0;
        baobzi_eval_f64(nullptr, &x, &y); // must not crash
        REQUIRE(std::strlen(baobzi_last_error()) > 0);
    }
}

// NOLINTEND(cert-msc51-cpp,cert-msc32-c)
