#include <baobzi/baobzi.hpp>

#include <array>
#include <cmath>
#include <numbers>
#include <random>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using baobzi::fit;
using baobzi::options;

namespace {
constexpr int N_SAMPLE = 5000;

template <class F1, class F2>
double max_rel_err_1d(F1 &&exact, F2 &&approx, double a, double b, int n) {
    std::mt19937 gen(1);
    std::uniform_real_distribution<double> d(a, b);
    double mx = 0.0;
    for (int i = 0; i < n; ++i) {
        const double x = d(gen);
        const double y = exact(x);
        const double yh = approx(x);
        if (std::abs(y) > 1e-12)
            mx = std::max(mx, std::abs((y - yh) / y));
    }
    return mx;
}

template <class F1, class F2>
double max_rel_err_2d(F1 &&exact, F2 &&approx,
                     std::array<double, 2> a, std::array<double, 2> b, int n) {
    std::mt19937 gen(1);
    std::uniform_real_distribution<double> dx(a[0], b[0]);
    std::uniform_real_distribution<double> dy(a[1], b[1]);
    double mx = 0.0;
    for (int i = 0; i < n; ++i) {
        std::array<double, 2> x{dx(gen), dy(gen)};
        const double y = exact(x);
        const double yh = approx(x);
        if (std::abs(y) > 1e-12)
            mx = std::max(mx, std::abs((y - yh) / y));
    }
    return mx;
}
} // namespace

TEST_CASE("1D smooth sin on [0, 2π], compile-time degree 8", "[baobzi][smooth]") {
    auto f = [](double x) { return std::sin(5.0 * x); };
    const double a = 0.0;
    const double b = 2.0 * std::numbers::pi_v<double>;

    auto fn = fit<8>(f, a, b, options{.tol_kind = baobzi::TolKind::RelativeMax});
    // Narrow the check away from the very boundary where the fit is
    // approximate; single-point boundary eval is still fine.
    REQUIRE(max_rel_err_1d(f, fn, a + 1e-6, b - 1e-6, N_SAMPLE) < 1e-6);
}

TEST_CASE("2D smooth polynomial, compile-time degree 8", "[baobzi][smooth][2d]") {
    // polyfit's FuncEvalND requires a tuple-like output even for a single
    // scalar; wrap in std::array<double, 1>.
    auto f = [](std::array<double, 2> x) -> std::array<double, 1> {
        return {x[0] * x[0] * x[0] * x[1] * x[1] + 0.1};
    };
    auto exact = [](std::array<double, 2> x) {
        return x[0] * x[0] * x[0] * x[1] * x[1] + 0.1;
    };
    std::array<double, 2> a{0.0, 0.0};
    std::array<double, 2> b{2.0, 2.0};

    auto fn = fit<8>(f, a, b);
    auto approx = [&](std::array<double, 2> x) { return fn(x)[0]; };
    REQUIRE(max_rel_err_2d(exact, approx, a, b, 2000) < 1e-6);
}

TEST_CASE("Runge function forces paneling", "[baobzi][runge]") {
    auto f = [](double x) { return 1.0 / (1.0 + 25.0 * x * x); };
    const double a = -1.0, b = 1.0;

    auto fn = fit<8>(f, a, b);
    REQUIRE(max_rel_err_1d(f, fn, a + 1e-6, b - 1e-6, N_SAMPLE) < 1e-6);
}

TEST_CASE("Near-singular log forces subdivision", "[baobzi][log]") {
    const double shift = 1e-3;
    auto f = [shift](double x) { return std::log(x + shift); };
    const double a = 0.0, b = 1.0;

    auto fn = fit<10>(f, a, b);
    REQUIRE(max_rel_err_1d(f, fn, a + 1e-3, b - 1e-6, N_SAMPLE) < 1e-5);
}

TEST_CASE("Out-of-domain returns NaN", "[baobzi][ood]") {
    auto f = [](double x) { return std::sin(x); };
    auto fn = fit<8>(f, 0.0, 1.0);

    const double y = fn(-0.5);
    REQUIRE(std::isnan(y));
}

TEST_CASE("Rejects non-positive tolerance", "[baobzi][errors]") {
    auto f = [](double x) { return x * x; };
    REQUIRE_THROWS_AS(fit(f, /*eps=*/0.0, 0.0, 1.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(fit(f, /*eps=*/-1.0, 0.0, 1.0),
                      std::invalid_argument);
}

TEST_CASE("Rejects unsupported runtime degree", "[baobzi][errors]") {
    auto f = [](double x) { return x * x; };
    REQUIRE_THROWS_AS(fit(f, /*n=*/2, /*tol=*/1e-6, 0.0, 1.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(fit(f, /*n=*/100, /*tol=*/1e-6, 0.0, 1.0),
                      std::invalid_argument);
}

TEST_CASE("Runtime-integer degree builds a working fit", "[baobzi][runtime-degree]") {
    auto f = [](double x) { return std::cos(3.0 * x); };
    const int n = 8;

    auto fn = fit(f, n, /*tol=*/1e-8, 0.0, 1.0);
    std::mt19937 gen(1);
    std::uniform_real_distribution<double> d(0.0, 1.0);
    double mx = 0.0;
    for (int i = 0; i < N_SAMPLE; ++i) {
        const double x = d(gen);
        mx = std::max(mx, std::abs(f(x) - fn(x)));
    }
    REQUIRE(mx < 1e-6);
}

TEST_CASE("Tolerance-driven overload converges", "[baobzi][tol-driven]") {
    auto f = [](double x) { return std::sin(5.0 * x); };
    auto fn = fit(f, /*eps=*/1e-8, 0.0, 1.0);
    REQUIRE(max_rel_err_1d(f, fn, 1e-6, 1.0 - 1e-6, N_SAMPLE) < 1e-5);
}

TEST_CASE("3D scalar exp(-r^2)", "[baobzi][smooth][3d]") {
    auto f = [](std::array<double, 3> x) -> std::array<double, 1> {
        return {std::exp(-x[0] * x[0] - x[1] * x[1] - x[2] * x[2])};
    };
    auto fn = fit<8>(f, std::array{-1.0, -1.0, -1.0},
                               std::array{1.0, 1.0, 1.0});

    std::mt19937 gen(2);
    std::uniform_real_distribution<double> d(-0.99, 0.99);
    double mx = 0.0;
    for (int i = 0; i < 3000; ++i) {
        std::array<double, 3> x{d(gen), d(gen), d(gen)};
        const double exact = f(x)[0];
        const double approx = fn(x)[0];
        if (std::abs(exact) > 1e-12)
            mx = std::max(mx, std::abs(exact - approx) / std::abs(exact));
    }
    REQUIRE(mx < 1e-6);
}

TEST_CASE("Vector-valued 2D -> 2D output", "[baobzi][vector-output]") {
    auto f = [](std::array<double, 2> x) -> std::array<double, 2> {
        return {std::sin(x[0] + x[1]), std::cos(x[0] - x[1])};
    };
    auto fn = fit<8>(f, std::array{0.0, 0.0}, std::array{1.0, 1.0});

    std::mt19937 gen(3);
    std::uniform_real_distribution<double> d(1e-3, 1.0 - 1e-3);
    double mx = 0.0;
    for (int i = 0; i < 2000; ++i) {
        std::array<double, 2> x{d(gen), d(gen)};
        const auto exact = f(x);
        const auto approx = fn(x);
        for (std::size_t k = 0; k < 2; ++k)
            mx = std::max(mx, std::abs(exact[k] - approx[k]));
    }
    REQUIRE(mx < 1e-6);
}

TEST_CASE("Sharp tanh step forces subdivision", "[baobzi][sharp]") {
    auto f = [](double x) { return std::tanh(50.0 * (x - 0.3)); };
    auto fn = fit<8>(f, 0.0, 1.0,
                               options{.tol_kind = baobzi::TolKind::AbsoluteMax,
                                           .max_depth = 30});
    // Check away from the exact step where we expect subdivision to give
    // good accuracy; rel_err blows up near zero-crossings so use abs.
    std::mt19937 gen(4);
    std::uniform_real_distribution<double> d(1e-3, 1.0 - 1e-3);
    double mx = 0.0;
    for (int i = 0; i < N_SAMPLE; ++i) {
        const double x = d(gen);
        mx = std::max(mx, std::abs(f(x) - fn(x)));
    }
    REQUIRE(mx < 1e-5);
}

TEST_CASE("2D anisotropic gaussian bump", "[baobzi][2d][bump]") {
    auto f = [](std::array<double, 2> x) -> std::array<double, 1> {
        return {std::exp(-100.0 * (x[0] - 0.5) * (x[0] - 0.5)
                         - (x[1] - 0.5) * (x[1] - 0.5))};
    };
    auto exact = [](std::array<double, 2> x) {
        return std::exp(-100.0 * (x[0] - 0.5) * (x[0] - 0.5)
                        - (x[1] - 0.5) * (x[1] - 0.5));
    };
    auto fn = fit<10>(f, std::array{0.0, 0.0}, std::array{1.0, 1.0});
    auto approx = [&](std::array<double, 2> x) { return fn(x)[0]; };
    REQUIRE(max_rel_err_2d(exact, approx,
                           {0.001, 0.001}, {0.999, 0.999}, 5000) < 1e-6);
}

TEST_CASE("sqrt|x - 0.5| — not C^1, max_depth guards runaway",
          "[baobzi][sharp]") {
    auto f = [](double x) { return std::sqrt(std::abs(x - 0.5)); };
    // Low max_depth — the fit will hit the ceiling at the singularity.
    REQUIRE_THROWS_AS(
        fit<8>(f, 0.0, 1.0,
                         options{.tol_kind = baobzi::TolKind::AbsoluteMax,
                                     .max_depth = 4}),
        baobzi::MaxDepthExceeded);
    // Generous max_depth — should succeed and meet a loose tolerance away
    // from the singularity.
    auto fn = fit<10>(f, 0.0, 1.0,
                                options{.tol_kind = baobzi::TolKind::AbsoluteMax,
                                            .max_depth = 50});
    // Sample away from the non-smooth point.
    std::mt19937 gen(5);
    std::uniform_real_distribution<double> d(0.0, 0.45);
    double mx = 0.0;
    for (int i = 0; i < 1000; ++i) {
        const double x = d(gen);
        mx = std::max(mx, std::abs(f(x) - fn(x)));
    }
    REQUIRE(mx < 1e-4);
}

TEST_CASE("Batch vs single evaluation agree", "[baobzi][batch]") {
    auto f = [](double x) { return std::sin(4.0 * x); };
    auto fn = fit<8>(f, 0.0, 1.0);

    std::mt19937 gen(6);
    std::uniform_real_distribution<double> d(1e-3, 1.0 - 1e-3);
    constexpr int N = 500;
    std::vector<double> xs(N);
    for (auto &x : xs) x = d(gen);

    std::vector<double> batch(N);
    fn(xs.data(), batch.data(), N);

    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
        const double single = fn(xs[i]);
        REQUIRE(single == batch[i]);
    }
}
