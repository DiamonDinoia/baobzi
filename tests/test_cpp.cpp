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

    auto fn = fit<8>(f, a, b, /*tol=*/1e-10,
                     options{.tol_kind = baobzi::TolKind::RelativeMax});
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

    auto fn = fit<8>(f, a, b, /*tol=*/1e-10);
    auto approx = [&](std::array<double, 2> x) { return fn(x)[0]; };
    REQUIRE(max_rel_err_2d(exact, approx, a, b, 2000) < 1e-6);
}

TEST_CASE("Runge function forces paneling", "[baobzi][runge]") {
    auto f = [](double x) { return 1.0 / (1.0 + 25.0 * x * x); };
    const double a = -1.0, b = 1.0;

    auto fn = fit<8>(f, a, b, /*tol=*/1e-10);
    REQUIRE(max_rel_err_1d(f, fn, a + 1e-6, b - 1e-6, N_SAMPLE) < 1e-6);
}

TEST_CASE("Near-singular log forces subdivision", "[baobzi][log]") {
    const double shift = 1e-3;
    auto f = [shift](double x) { return std::log(x + shift); };
    const double a = 0.0, b = 1.0;

    auto fn = fit<10>(f, a, b, /*tol=*/1e-10);
    REQUIRE(max_rel_err_1d(f, fn, a + 1e-3, b - 1e-6, N_SAMPLE) < 1e-5);
}

TEST_CASE("Out-of-domain returns NaN", "[baobzi][ood]") {
    auto f = [](double x) { return std::sin(x); };
    auto fn = fit<8>(f, 0.0, 1.0, /*tol=*/1e-10);

    const double y = fn(-0.5);
    REQUIRE(std::isnan(y));
}

TEST_CASE("Rejects non-positive tolerance", "[baobzi][errors]") {
    auto f = [](double x) { return x * x; };
    REQUIRE_THROWS_AS(fit(f, 0.0, 1.0, /*eps=*/0.0),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(fit(f, 0.0, 1.0, /*eps=*/-1.0),
                      std::invalid_argument);
}

TEST_CASE("Tolerance-driven overload converges", "[baobzi][tol-driven]") {
    auto f = [](double x) { return std::sin(5.0 * x); };
    auto fn = fit(f, 0.0, 1.0, /*eps=*/1e-8);
    REQUIRE(max_rel_err_1d(f, fn, 1e-6, 1.0 - 1e-6, N_SAMPLE) < 1e-5);
}

TEST_CASE("3D scalar exp(-r^2)", "[baobzi][smooth][3d]") {
    auto f = [](std::array<double, 3> x) -> std::array<double, 1> {
        return {std::exp(-x[0] * x[0] - x[1] * x[1] - x[2] * x[2])};
    };
    auto fn = fit<8>(f, std::array{-1.0, -1.0, -1.0},
                     std::array{1.0, 1.0, 1.0}, /*tol=*/1e-10);

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
    auto fn = fit<8>(f, std::array{0.0, 0.0}, std::array{1.0, 1.0},
                     /*tol=*/1e-10);

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
    auto fn = fit<8>(f, 0.0, 1.0, /*tol=*/1e-10,
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
    auto fn = fit<10>(f, std::array{0.0, 0.0}, std::array{1.0, 1.0},
                      /*tol=*/1e-10);
    auto approx = [&](std::array<double, 2> x) { return fn(x)[0]; };
    REQUIRE(max_rel_err_2d(exact, approx,
                           {0.001, 0.001}, {0.999, 0.999}, 5000) < 1e-6);
}

TEST_CASE("sqrt|x - 0.5| — not C^1, max_depth guards runaway",
          "[baobzi][sharp]") {
    auto f = [](double x) { return std::sqrt(std::abs(x - 0.5)); };
    // Low max_depth — the fit will hit the ceiling at the singularity.
    REQUIRE_THROWS_AS(
        fit<8>(f, 0.0, 1.0, /*tol=*/1e-10,
               options{.tol_kind = baobzi::TolKind::AbsoluteMax,
                       .max_depth = 4}),
        baobzi::MaxDepthExceeded);
    // Generous max_depth — should succeed and meet a loose tolerance away
    // from the singularity.
    auto fn = fit<10>(f, 0.0, 1.0, /*tol=*/1e-10,
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
    auto fn = fit<8>(f, 0.0, 1.0, /*tol=*/1e-10);

    std::mt19937 gen(6);
    std::uniform_real_distribution<double> d(1e-3, 1.0 - 1e-3);
    constexpr int N = 500;
    std::vector<double> xs(N);
    for (auto &x : xs) x = d(gen);

    std::vector<double> batch(N);
    fn(xs.data(), batch.data(), N);

    // Batch path uses polyfit's SIMD Horner, scalar path uses scalar Horner —
    // identical mathematically but FMA reordering can drop a ULP.
    constexpr double ulp = std::numeric_limits<double>::epsilon();
    for (std::size_t i = 0; i < static_cast<std::size_t>(N); ++i) {
        const double single = fn(xs[i]);
        REQUIRE(std::abs(single - batch[i])
                <= 8.0 * ulp * std::max(1.0, std::abs(single)));
    }
}

TEST_CASE("Batch vs single evaluation agree — 2D vector output", "[baobzi][batch][2d]") {
    auto f = [](std::array<double, 2> x) -> std::array<double, 2> {
        return {std::sin(x[0] + x[1]), std::cos(x[0] - x[1])};
    };
    auto fn = fit<8>(f, std::array{0.0, 0.0}, std::array{1.0, 1.0}, /*tol=*/1e-10);

    std::mt19937 gen(42);
    std::uniform_real_distribution<double> d(1e-3, 1.0 - 1e-3);
    constexpr int N = 1024; // above counting-sort threshold
    std::vector<double> flat(2 * N);
    std::vector<std::array<double, 2>> pts(N);
    for (int i = 0; i < N; ++i) {
        pts[i] = {d(gen), d(gen)};
        flat[2 * i]     = pts[i][0];
        flat[2 * i + 1] = pts[i][1];
    }

    std::vector<double> batch(2 * N);
    fn(flat.data(), batch.data(), N);

    constexpr double ulp = std::numeric_limits<double>::epsilon();
    for (int i = 0; i < N; ++i) {
        const auto single = fn(pts[i]);
        REQUIRE(std::abs(single[0] - batch[2 * i])
                <= 4.0 * ulp * std::max(1.0, std::abs(single[0])));
        REQUIRE(std::abs(single[1] - batch[2 * i + 1])
                <= 4.0 * ulp * std::max(1.0, std::abs(single[1])));
    }
}

TEST_CASE("Batch vs single evaluation agree — 3D scalar output", "[baobzi][batch][3d]") {
    auto f = [](std::array<double, 3> x) -> std::array<double, 1> {
        return {std::exp(-x[0] * x[0] - x[1] * x[1] - x[2] * x[2])};
    };
    auto fn = fit<8>(f, std::array{-1.0, -1.0, -1.0},
                     std::array{1.0, 1.0, 1.0}, /*tol=*/1e-10);

    std::mt19937 gen(7);
    std::uniform_real_distribution<double> d(-0.99, 0.99);
    constexpr int N = 2000;
    std::vector<double> flat(3 * N);
    std::vector<std::array<double, 3>> pts(N);
    for (int i = 0; i < N; ++i) {
        pts[i] = {d(gen), d(gen), d(gen)};
        for (int j = 0; j < 3; ++j) flat[3 * i + j] = pts[i][j];
    }

    std::vector<double> batch(N);
    fn(flat.data(), batch.data(), N);

    constexpr double ulp = std::numeric_limits<double>::epsilon();
    for (int i = 0; i < N; ++i) {
        const auto single = fn(pts[i]);
        REQUIRE(std::abs(single[0] - batch[i])
                <= 4.0 * ulp * std::max(1.0, std::abs(single[0])));
    }
}

TEST_CASE("Batch vs single evaluation agree across L4 tile boundary",
          "[baobzi][batch][tile]") {
    // Phase 16 / Layer L4: the batch path now tiles when n_trg exceeds
    // tile_K (default 64 K). Pin the boundary by feeding a batch large
    // enough to span multiple tiles and assert per-point agreement with
    // the scalar path.
    auto f = [](double x) { return std::sin(4.0 * x) + std::cos(7.0 * x); };
    auto fn = fit<8>(f, 0.0, 1.0, /*tol=*/1e-10);

    std::mt19937 gen(13);
    std::uniform_real_distribution<double> d(1e-3, 1.0 - 1e-3);
    constexpr std::size_t N = 200'000; // > default tile_K (65 536) → ≥ 4 tiles
    std::vector<double> xs(N);
    for (auto &x : xs) x = d(gen);

    std::vector<double> batch(N);
    fn(xs.data(), batch.data(), N);

    constexpr double ulp = std::numeric_limits<double>::epsilon();
    for (std::size_t i = 0; i < N; ++i) {
        const double single = fn(xs[i]);
        REQUIRE(std::abs(single - batch[i])
                <= 8.0 * ulp * std::max(1.0, std::abs(single)));
    }
}

TEST_CASE("Memory budget aborts a runaway near-singular fit",
          "[baobzi][memory-budget]") {
    // 3D Yukawa with a *very* tight tolerance over a domain straddling the
    // origin singularity refines aggressively. Each leaf is ~6 KiB
    // (deg=8 in 3D), so a 1 MiB budget caps at ~170 leaves before bailing
    // — well below what the smooth-tol target would otherwise pursue.
    auto f = [](std::array<double, 3> x) -> std::array<double, 1> {
        const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        return {std::exp(-r) / r};
    };
    REQUIRE_THROWS_AS(
        fit<8>(f, std::array{0.01, 0.01, 0.01}, std::array{1.5, 1.5, 1.5},
               /*tol=*/1e-12,
               options{.tol_kind = baobzi::TolKind::AbsoluteMax,
                       .max_depth = 50, .max_memory_mib = 1}),
        baobzi::MemoryBudgetExceeded);
    // Disabling the budget but keeping the depth ceiling still catches the
    // runaway via the existing MaxDepthExceeded path.
    auto g = [](double x) { return std::sqrt(std::abs(x - 0.5)); };
    REQUIRE_THROWS_AS(
        fit<8>(g, 0.0, 1.0, /*tol=*/1e-12,
               options{.tol_kind = baobzi::TolKind::AbsoluteMax,
                       .max_depth = 4, .max_memory_mib = 0}),
        baobzi::MaxDepthExceeded);
}

TEST_CASE("allow_max_depth_leaves accepts unconverged panels",
          "[baobzi][maxdepth][lossy]") {
    auto f = [](double x) { return std::sqrt(std::abs(x - 0.5)); };

    // Default path (allow_max_depth_leaves=false) must throw and the
    // exception must carry every unconverged panel, not just the first.
    try {
        (void)fit<8>(f, 0.0, 1.0, /*tol=*/1e-10,
                     options{.tol_kind = baobzi::TolKind::AbsoluteMax,
                             .max_depth = 4});
        FAIL("expected MaxDepthExceeded");
    } catch (const baobzi::MaxDepthExceeded &e) {
        REQUIRE_FALSE(e.panels().empty());
        for (const auto &p : e.panels()) {
            REQUIRE(p.a.size() == 1);
            REQUIRE(p.b.size() == 1);
            REQUIRE(p.a[0] < p.b[0]);
            REQUIRE(p.depth == 4);
        }
        REQUIRE(e.a() == e.panels().front().a);
        REQUIRE(e.b() == e.panels().front().b);
    }

    // Opt-in path: same fit completes, and the unconverged panels are
    // surfaced via Function::non_converged_panels(). Eval still produces
    // a finite (best-effort) value at the singular point.
    auto fn = fit<8>(f, 0.0, 1.0, /*tol=*/1e-10,
                     options{.tol_kind = baobzi::TolKind::AbsoluteMax,
                             .max_depth = 4,
                             .allow_max_depth_leaves = true});
    REQUIRE_FALSE(fn.non_converged_panels().empty());
    REQUIRE(std::isfinite(fn(0.5)));
}

TEST_CASE("Batch handles out-of-domain points as NaN", "[baobzi][batch][ood]") {
    auto f = [](double x) { return std::sin(x); };
    auto fn = fit<8>(f, 0.0, 1.0, /*tol=*/1e-10);

    std::vector<double> xs{0.1, -0.5, 0.3, 2.0, 0.9, 5.0};
    std::vector<double> out(xs.size());
    fn(xs.data(), out.data(), xs.size());

    REQUIRE(out[0] == fn(0.1));
    REQUIRE(std::isnan(out[1]));
    REQUIRE(out[2] == fn(0.3));
    REQUIRE(std::isnan(out[3]));
    REQUIRE(out[4] == fn(0.9));
    REQUIRE(std::isnan(out[5]));
}
