// Tests for Function::Scratch — the caller-owned batch-path scratch.
//
// Covers:
//   * bit-exact equivalence of the no-scratch and Scratch overloads
//     (1D scalar, 1D array<T,1>, 2D, vector-output)
//   * pre-sized Scratch performs zero allocations during the batch call
//   * reuse across multiple batch calls
//   * clear() releases capacity and the next call re-allocates
//   * std::pmr::polymorphic_allocator works end-to-end through a
//     monotonic_buffer_resource

#include <baobzi/baobzi.hpp>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <memory_resource>
#include <numbers>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>

using baobzi::fit;
using baobzi::options;

namespace {

// Counting allocator: forwards to std::allocator while incrementing a
// shared atomic counter on every allocate() call. Used to assert that
// pre-sized scratch performs zero allocations on subsequent calls.
template <class T>
struct CountingAllocator {
    using value_type = T;
    std::atomic<std::size_t> *count = nullptr;

    CountingAllocator() = default;
    explicit CountingAllocator(std::atomic<std::size_t> &c) noexcept : count(&c) {}
    template <class U>
    CountingAllocator(const CountingAllocator<U> &o) noexcept : count(o.count) {}

    [[nodiscard]] T *allocate(std::size_t n) {
        if (count) count->fetch_add(1, std::memory_order_relaxed);
        return std::allocator<T>{}.allocate(n);
    }
    void deallocate(T *p, std::size_t n) noexcept {
        std::allocator<T>{}.deallocate(p, n);
    }
    template <class U>
    bool operator==(const CountingAllocator<U> &o) const noexcept { return count == o.count; }
};

constexpr double kLo = -1.0;
constexpr double kHi =  1.0;
constexpr std::size_t kN = 5000;

std::vector<double> random_xs(std::size_t n, double lo, double hi, unsigned seed = 1) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> d(lo, hi);
    std::vector<double> xs(n);
    for (auto &x : xs) x = d(gen);
    return xs;
}

}  // namespace

TEST_CASE("Scratch: bit-exact 1D scalar parity", "[scratch]") {
    auto f = [](double x) -> double { return std::sin(5.0 * x); };
    auto fn = fit(f, kLo, kHi, 1e-12);

    const auto xs = random_xs(kN, kLo, kHi);
    std::vector<double> ref(kN), got(kN);

    fn(xs.data(), ref.data(), kN);                         // no-scratch overload

    decltype(fn)::Scratch<> s;
    fn(xs.data(), got.data(), kN, s);                       // Scratch overload

    for (std::size_t i = 0; i < kN; ++i) REQUIRE(got[i] == ref[i]);
}

TEST_CASE("Scratch: bit-exact 1D array<T,1> vector-output parity", "[scratch]") {
    auto f = [](std::array<double, 1> x) -> std::array<double, 4> {
        const double v = x[0];
        return {std::sin(v), std::cos(v), v, v * v};
    };
    auto fn = fit(f, std::array<double, 1>{kLo}, std::array<double, 1>{kHi}, 1e-12);

    const auto xs = random_xs(kN, kLo, kHi);
    std::vector<double> ref(4 * kN), got(4 * kN);

    fn(xs.data(), ref.data(), kN);
    decltype(fn)::Scratch<> s(fn, kN);
    fn(xs.data(), got.data(), kN, s);

    for (std::size_t i = 0; i < 4 * kN; ++i) REQUIRE(got[i] == ref[i]);
}

TEST_CASE("Scratch: bit-exact 2D parity", "[scratch]") {
    auto f = [](std::array<double, 2> x) -> std::array<double, 1> {
        return {std::sin(3.0 * x[0]) * std::cos(2.0 * x[1])};
    };
    auto fn = fit(f, std::array<double, 2>{kLo, kLo},
                     std::array<double, 2>{kHi, kHi}, 1e-10);

    std::mt19937 gen(2);
    std::uniform_real_distribution<double> d(kLo, kHi);
    std::vector<double> xs(2 * kN);
    for (auto &x : xs) x = d(gen);

    std::vector<double> ref(kN), got(kN);
    fn(xs.data(), ref.data(), kN);
    decltype(fn)::Scratch<> s(fn, kN);
    fn(xs.data(), got.data(), kN, s);

    for (std::size_t i = 0; i < kN; ++i) REQUIRE(got[i] == ref[i]);
}

TEST_CASE("Scratch: pre-sized scratch makes zero allocations on subsequent calls", "[scratch]") {
    auto f = [](double x) -> double { return std::sin(5.0 * x); };
    auto fn = fit(f, kLo, kHi, 1e-12);

    const auto xs = random_xs(kN, kLo, kHi);
    std::vector<double> out(kN);

    std::atomic<std::size_t> alloc_count{0};
    using Scratch = typename decltype(fn)::template Scratch<CountingAllocator<double>>;
    Scratch s(fn, kN, CountingAllocator<double>{alloc_count});

    // After construction, all allocations are done up-front.
    const std::size_t allocs_after_ctor = alloc_count.load();
    REQUIRE(allocs_after_ctor > 0);

    // Subsequent batch calls must not allocate.
    for (int rep = 0; rep < 5; ++rep) fn(xs.data(), out.data(), kN, s);
    REQUIRE(alloc_count.load() == allocs_after_ctor);
}

TEST_CASE("Scratch: clear() releases capacity; next call re-allocates", "[scratch]") {
    auto f = [](double x) -> double { return std::cos(x); };
    auto fn = fit(f, kLo, kHi, 1e-12);

    const auto xs = random_xs(kN, kLo, kHi);
    std::vector<double> out(kN);

    std::atomic<std::size_t> alloc_count{0};
    using Scratch = typename decltype(fn)::template Scratch<CountingAllocator<double>>;
    Scratch s(fn, kN, CountingAllocator<double>{alloc_count});

    const std::size_t base = alloc_count.load();
    fn(xs.data(), out.data(), kN, s);
    REQUIRE(alloc_count.load() == base);

    s.clear();
    fn(xs.data(), out.data(), kN, s);
    REQUIRE(alloc_count.load() > base);
}

TEST_CASE("Scratch: std::pmr::polymorphic_allocator with monotonic_buffer_resource", "[scratch]") {
    auto f = [](double x) -> double { return std::exp(-x * x); };
    auto fn = fit(f, kLo, kHi, 1e-12);

    const auto xs = random_xs(kN, kLo, kHi);
    std::vector<double> ref(kN), got(kN);
    fn(xs.data(), ref.data(), kN);

    // 2 MiB arena — plenty for kN points across all 5 scratch buffers.
    std::pmr::monotonic_buffer_resource arena{2u << 20};
    using Scratch = typename decltype(fn)::template Scratch<
        std::pmr::polymorphic_allocator<double>>;
    Scratch s(fn, kN, std::pmr::polymorphic_allocator<double>{&arena});

    fn(xs.data(), got.data(), kN, s);
    for (std::size_t i = 0; i < kN; ++i) REQUIRE(got[i] == ref[i]);
}

TEST_CASE("Scratch: make_scratch factory matches direct Scratch construction", "[scratch]") {
    auto f = [](double x) -> double { return std::tanh(2.0 * x); };
    auto fn = fit(f, kLo, kHi, 1e-12);

    const auto xs = random_xs(kN, kLo, kHi);
    std::vector<double> ref(kN), got(kN);
    fn(xs.data(), ref.data(), kN);

    auto s = fn.make_scratch(kN);  // template deduces default allocator
    fn(xs.data(), got.data(), kN, s);

    for (std::size_t i = 0; i < kN; ++i) REQUIRE(got[i] == ref[i]);

    // Custom-allocator factory
    std::atomic<std::size_t> alloc_count{0};
    auto s2 = fn.make_scratch<CountingAllocator<double>>(
        kN, CountingAllocator<double>{alloc_count});
    REQUIRE(alloc_count.load() > 0);
    const auto base = alloc_count.load();
    fn(xs.data(), got.data(), kN, s2);
    REQUIRE(alloc_count.load() == base);
}

TEST_CASE("Scratch: small-n path (below kSortThreshold) bypasses scratch", "[scratch]") {
    auto f = [](double x) -> double { return std::sin(x); };
    auto fn = fit(f, kLo, kHi, 1e-12);

    // n=8 < kSortThreshold (32) — the batch path should fall back to the
    // per-point loop and skip scratch reservation entirely.
    std::atomic<std::size_t> alloc_count{0};
    using Scratch = typename decltype(fn)::template Scratch<CountingAllocator<double>>;
    Scratch s{CountingAllocator<double>{alloc_count}};  // unsized

    const auto xs = random_xs(8, kLo, kHi);
    std::vector<double> ref(8), got(8);
    fn(xs.data(), ref.data(), 8);
    fn(xs.data(), got.data(), 8, s);

    for (std::size_t i = 0; i < 8; ++i) REQUIRE(got[i] == ref[i]);
    REQUIRE(alloc_count.load() == 0);
}

// C1 — drop leaf_ids[] / recompute at scatter — must keep the batch
// path bit-exact against per-point single-eval for a vector-valued 1D
// fit. The interleaved-output shape exercises the scatter recompute
// (it has to recover the same id twice: once for the histogram, once
// for placement). Seed is fixed so the failure is reproducible.
TEST_CASE("Scratch: batch matches per-point on 1D vector-output (C1 recompute)",
          "[scratch][c1][recompute]") {
    auto f = [](std::array<double, 1> x) -> std::array<double, 2> {
        const double v = x[0];
        return {std::sin(3.0 * v), std::cos(2.0 * v)};
    };
    auto fn = fit(f, std::array<double, 1>{kLo}, std::array<double, 1>{kHi}, 1e-12);

    constexpr std::size_t kPts = 1000;
    std::mt19937 gen(0xC1C0DE);
    std::uniform_real_distribution<double> d(kLo, kHi);
    std::vector<double> xs(kPts);
    for (auto &x : xs) x = d(gen);

    // Per-point reference via the scalar single-point operator()(input).
    std::vector<double> ref(2 * kPts);
    for (std::size_t i = 0; i < kPts; ++i) {
        const auto y = fn(std::array<double, 1>{xs[i]});
        ref[2 * i + 0] = y[0];
        ref[2 * i + 1] = y[1];
    }

    // Batch path via Scratch.
    std::vector<double> got(2 * kPts);
    auto s = fn.make_scratch(kPts);
    fn(xs.data(), got.data(), kPts, s);

    double max_rel = 0.0;
    for (std::size_t i = 0; i < 2 * kPts; ++i) {
        const double denom = std::max(std::abs(ref[i]), 1e-300);
        max_rel = std::max(max_rel, std::abs(got[i] - ref[i]) / denom);
    }
    REQUIRE(max_rel <= 1e-15);
}
