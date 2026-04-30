// Microbench harness driven by martinus/nanobench: handles warmup, MdAPE
// stability checks, TSC-frequency calibration, and produces machine-readable
// output (including JSON / CSV via Bench::output(...) if needed).
//
// Sweeps {1D, 2D, 3D} × scientific-kernel × {deg 6, 8, 10} × N ∈ {1, 32, 1024,
// 10⁶}. Pin the process to one core (e.g. `taskset -c 2 ./baobzi_microbench`)
// for stable numbers — nanobench reports MdAPE, so unstable measurements
// surface as a high error percentage rather than silent noise.

#define ANKERL_NANOBENCH_IMPLEMENT
#include <nanobench.h>

#include <baobzi/baobzi.hpp>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include <xsimd/xsimd.hpp>

namespace {

// 1D scientific kernels. Suite spans smooth, oscillatory, sigmoid-shaped,
// sharp-transition, and near-singular cases so the leaf-eval and tree-depth
// regimes both get exercised.
auto make_runge1d() { return [](double x) { return 1.0 / (1.0 + 25.0 * x * x); }; }
auto make_erf1d()   { return [](double x) { return std::erf(x); }; }
auto make_j0_1d()   { return [](double x) { return std::cyl_bessel_j(0, x); }; }
auto make_tanh1d()  { return [](double x) { return std::tanh(50.0 * x); }; }
auto make_log1p1d() { return [](double x) { return std::log1p(x); }; }

// 2D / 3D kernels typical of scientific computing: Gaussians, oscillatory,
// RBFs, screened-Coulomb (Yukawa).
auto make_bump2d() {
    return [](std::array<double, 2> x) -> std::array<double, 1> {
        return {std::exp(-100.0 * (x[0] - 0.5) * (x[0] - 0.5)
                         - (x[1] - 0.5) * (x[1] - 0.5))};
    };
}
auto make_osc2d() {
    return [](std::array<double, 2> x) -> std::array<double, 1> {
        return {std::cos(8.0 * x[0]) * std::cos(8.0 * x[1])};
    };
}
auto make_mq2d() {
    return [](std::array<double, 2> x) -> std::array<double, 1> {
        return {std::sqrt(1.0 + x[0] * x[0] + x[1] * x[1])};
    };
}

auto make_gauss3d() {
    return [](std::array<double, 3> x) -> std::array<double, 1> {
        return {std::exp(-x[0] * x[0] - x[1] * x[1] - x[2] * x[2])};
    };
}
auto make_yukawa3d() {
    return [](std::array<double, 3> x) -> std::array<double, 1> {
        const double r = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        return {std::exp(-r) / r};
    };
}
auto make_imq3d() {
    return [](std::array<double, 3> x) -> std::array<double, 1> {
        return {1.0 / std::sqrt(1.0 + x[0] * x[0] + x[1] * x[1] + x[2] * x[2])};
    };
}

// Scale a nanobench Bench so the per-cell sample volume matches the cost
// of `n_pts` work per call. nanobench's auto-tuning targets ~1 s total
// runtime and uses MdAPE to flag instability.
ankerl::nanobench::Bench make_bench(std::size_t n_pts) {
    ankerl::nanobench::Bench b;
    b.title("baobzi eval pipeline")
     .unit("eval")        // 1 unit = one f(x) evaluation
     .batch(static_cast<double>(n_pts))
     .relative(true)      // print MEvals/s relative to the first sample
     .warmup(3)
     .minEpochIterations(1);
    if (n_pts >= 1'000'000) b.minEpochTime(std::chrono::milliseconds(50));
    return b;
}

template <std::size_t Deg, class Fmaker>
void sweep_1d(ankerl::nanobench::Bench &b, const char *label, Fmaker make_f,
              double a, double b_) {
    auto f = make_f();
    auto fn = baobzi::fit<Deg>(f, a, b_, /*tol=*/1e-10);
    std::mt19937 gen(7);
    std::uniform_real_distribution<double> d(a + 1e-3, b_ - 1e-3);

    for (std::size_t n_pts : {std::size_t(1), std::size_t(32),
                              std::size_t(1024), std::size_t(1'000'000)}) {
        std::vector<double> xs(n_pts);
        for (auto &x : xs) x = d(gen);
        std::vector<double> out(n_pts);

        std::string name = std::string(label) + " deg=" + std::to_string(Deg) +
                           " dim=1 N=" + std::to_string(n_pts);
        b.batch(static_cast<double>(n_pts));
        b.run(name, [&] {
            fn(xs.data(), out.data(), n_pts);
            ankerl::nanobench::doNotOptimizeAway(out.data());
        });
    }
}

template <std::size_t Deg, std::size_t Dim, class Fmaker>
void sweep_nd(ankerl::nanobench::Bench &b, const char *label, Fmaker make_f,
              std::array<double, Dim> a, std::array<double, Dim> b_) {
    auto f = make_f();
    auto fn = baobzi::fit<Deg>(f, a, b_, /*tol=*/1e-10);
    std::mt19937 gen(7);
    std::uniform_real_distribution<double> ud(0.0, 1.0);
    auto pick = [&](std::size_t d) {
        return a[d] + (b_[d] - a[d] - 1e-3) * ud(gen) + 5e-4;
    };

    for (std::size_t n_pts : {std::size_t(1), std::size_t(32),
                              std::size_t(1024), std::size_t(1'000'000)}) {
        std::vector<double> flat(Dim * n_pts);
        for (std::size_t i = 0; i < n_pts; ++i)
            for (std::size_t d = 0; d < Dim; ++d)
                flat[Dim * i + d] = pick(d);
        std::vector<double> out(n_pts);

        std::string name = std::string(label) + " deg=" + std::to_string(Deg) +
                           " dim=" + std::to_string(Dim) +
                           " N=" + std::to_string(n_pts);
        b.batch(static_cast<double>(n_pts));
        b.run(name, [&] {
            fn(flat.data(), out.data(), n_pts);
            ankerl::nanobench::doNotOptimizeAway(out.data());
        });
    }
}

} // namespace

int main() {
    std::printf("# baobzi microbench (nanobench, xsimd lane_w=%zu)\n",
                xsimd::batch<double>::size);

    auto b = make_bench(1);

    // 1D suite
    sweep_1d<6>(b,  "1d_runge",       make_runge1d, -1.0, 1.0);
    sweep_1d<8>(b,  "1d_runge",       make_runge1d, -1.0, 1.0);
    sweep_1d<10>(b, "1d_runge",       make_runge1d, -1.0, 1.0);
    sweep_1d<8>(b,  "1d_erf",         make_erf1d,   -3.0, 3.0);
    sweep_1d<8>(b,  "1d_bessel_j0",   make_j0_1d,    0.5, 30.0);
    sweep_1d<10>(b, "1d_tanh_sharp",  make_tanh1d,  -1.0, 1.0);
    sweep_1d<8>(b,  "1d_log1p",       make_log1p1d, -0.9, 5.0);

    // 2D suite
    sweep_nd<6, 2>(b,  "2d_bump",  make_bump2d, {0.0, 0.0}, {1.0, 1.0});
    sweep_nd<8, 2>(b,  "2d_bump",  make_bump2d, {0.0, 0.0}, {1.0, 1.0});
    sweep_nd<10, 2>(b, "2d_bump",  make_bump2d, {0.0, 0.0}, {1.0, 1.0});
    sweep_nd<8, 2>(b,  "2d_osc",   make_osc2d,  {-1.0, -1.0}, {1.0, 1.0});
    sweep_nd<8, 2>(b,  "2d_mq",    make_mq2d,   {-1.0, -1.0}, {1.0, 1.0});

    // 3D suite
    sweep_nd<6, 3>(b,  "3d_gauss",  make_gauss3d,  {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0});
    sweep_nd<8, 3>(b,  "3d_gauss",  make_gauss3d,  {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0});
    sweep_nd<10, 3>(b, "3d_gauss",  make_gauss3d,  {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0});
    sweep_nd<8, 3>(b,  "3d_yukawa", make_yukawa3d, {0.2, 0.2, 0.2}, {1.5, 1.5, 1.5});
    sweep_nd<8, 3>(b,  "3d_imq",    make_imq3d,    {-1.0, -1.0, -1.0}, {1.0, 1.0, 1.0});

    return 0;
}
