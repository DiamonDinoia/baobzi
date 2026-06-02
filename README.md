# Baobzi

An adaptive fast function approximator based on tree search. `baobzi` turns a
CPU-intensive function into a cheap one (trading memory for time) by fitting
it piecewise with low-order Chebyshev polynomials on an adaptive grid. It is
conceptually similar to `chebfun`/`chebeval` but panelled, and typically
reaches the same tolerance at lower polynomial order — which is faster on
modern SIMD hardware.

Internally, `baobzi` represents your function as a grid of binary / quad /
oct / N-trees whose leaves carry the local Chebyshev expansion. Evaluation
locates the leaf containing your point and runs a Horner-order polynomial.

## Status

The library is a pure C++20 header — there are no longer any C / Fortran /
Python / MATLAB bindings. The fit/eval code path is built directly on
[polyfit](https://github.com/DiamonDinoia/polyfit) for the leaf polynomials
and [POET](https://github.com/DiamonDinoia/POET) for compile-time dispatch.

## Install

```bash
git clone https://github.com/flatironinstitute/baobzi.git
cmake -S baobzi -B baobzi/build -DCMAKE_BUILD_TYPE=Release
cmake --build baobzi/build -j 8
```

Header-only — no install step is needed beyond pointing your include path
at `baobzi/include`. See [Building](#building) for the test build.

## Quick start

A minimal 1D fit + scalar `operator()` (full file at
[`examples/c++/simple1d.cpp`](examples/c++/simple1d.cpp)):

```cpp
#include <baobzi/baobzi.hpp>
#include <cmath>

int main() {
    auto runge = [](double x) { return 1.0 / (1.0 + 25.0 * x * x); };

    // Fit `runge` on [-1, 1] to 1e-10. Panel count falls out of the
    // adaptive tree; the leaf degree defaults to 8 (AVX-512 lane count).
    auto fn = baobzi::fit(runge, -1.0, 1.0, /*tol=*/1e-10);

    return fn(0.3) != 0.0;
}
```

Multi-dimensional fits use `std::array` for both input and output:

```cpp
auto bump = [](std::array<double, 2> x) -> std::array<double, 1> {
    return {std::exp(-100.0 * (x[0] - 0.5) * (x[0] - 0.5)
                   - (x[1] - 0.5) * (x[1] - 0.5))};
};
auto fn = baobzi::fit(bump,
                      std::array{0.0, 0.0}, std::array{1.0, 1.0},
                      /*tol=*/1e-8);
```

Override the leaf degree via the template parameter:

```cpp
auto fn = baobzi::fit<12>(f, a, b, /*tol=*/1e-10);
```

`tol` is positional (no default) so every call site spells out its target
accuracy — there is no "magic tol" overload.

`baobzi::options` exposes the fit-time knobs (`tol_kind`, `min_depth`,
`max_depth`, `n_samples_per_dim`, `minimum_leaf_fraction`). `tol_kind` is a
`baobzi::TolKind` enum:

| Kind                         | Meaning                                  |
|------------------------------|------------------------------------------|
| `TolKind::RelativeMax`       | max-abs relative error on a sample grid  |
| `TolKind::AbsoluteMax`       | max-abs absolute error                    |
| `TolKind::RelativeL2`        | L2 relative error                         |
| `TolKind::AbsoluteL2`        | L2 absolute error                         |
| `TolKind::RelativeTail`      | 1D only — coefficient tail estimate       |
| `TolKind::AbsoluteTail`      | 1D only — coefficient tail estimate       |

## Building

```bash
cmake -S . -B build -DBAOBZI_BUILD_TESTS=ON
cmake --build build -j 8
ctest --test-dir build
```

Requirements: CMake ≥ 3.25, a C++20 compiler. polyfit / POET / Catch2 are
pulled in via `FetchContent` (see `cmake/baobzi_deps.cmake`).

## Language bindings

A C ABI (`libbaobzi_c`, header [`include/baobzi.h`](include/baobzi.h)) plus
**Python**, **Julia**, and **MATLAB** wrappers live under
[`bindings/`](bindings/). Each lets you fit using a function handle written in
the host language and surfaces C++ fit failures as native exceptions. They are
opt-in CMake options (all default OFF):

```bash
cmake -S . -B build -DBAOBZI_BUILD_PYTHON=ON -DBAOBZI_BUILD_JULIA=ON -DBAOBZI_BUILD_MATLAB=ON
ctest --test-dir build -R "python_baobzi|julia_baobzi|matlab_baobzi"
```

Missing toolchains are skipped gracefully. See [`bindings/README.md`](bindings/README.md)
for per-language build/usage, the cross-language parity check, and notes.

## Including in your CMake project

baobzi exposes two surfaces, with different consumption paths:

**C ABI (`libbaobzi_c`) — installable, `find_package`-able.** Install baobzi
(`cmake --install`) and link the namespaced target. It is self-contained
(only needs the installed `baobzi.h`):

```cmake
find_package(baobzi REQUIRED)
target_link_libraries(your_target PRIVATE baobzi::baobzi_c)         # shared
# or baobzi::baobzi_c_static for the static archive
```

**C++ header-only template API — in-tree only.** The `baobzi::baobzi` target
carries the `include/` tree and links polyfit/POET transitively, so just link
it (no manual include dirs or polyfit/poet lines):

```cmake
add_subdirectory(extern/baobzi)      # or FetchContent_Declare(baobzi ...)
target_link_libraries(your_target PRIVATE baobzi::baobzi)
```

The C++ template API is *not* part of the installed `find_package(baobzi)`
package: it instantiates against polyfit/POET headers, which are
FetchContent-only (not separately installable). So consume it via
`add_subdirectory` / `FetchContent` (where those deps resolve), and use the
installed package for the C ABI.

## Driving the SIMD-quantize fast path

`PolyTree::find_leaf_id` has a SIMD-quantize + table-lookup fast path:
one `vcvttpd2qq` (or scalar `vcvttsd2si`) per point plus one `uint32_t`
load from a `2^(input_dim * D)`-entry table, in place of recursive tree
descent. The table is built automatically when `input_dim * D <= 16`
bits of leaf index *and* the BFS produced a single subtree with uniform
refinement at depth `D`. The batch path's scatter loop then recomputes
the leaf id in place of a materialised `leaf_ids[]` buffer.

For smooth functions, tol-based refinement usually stops early and the
tree never reaches uniform depth — the fast path stays off. Two ways to
drive it on deliberately: tighten `tol` until refinement is uniform, or
set `options::min_uniform_depth` explicitly to force BFS to refine to a
known floor before the tolerance test exits. `Function::print_stats()`
reports `Leaf table: live (N entries, K KiB)` or `Leaf table: descent-only`.
Tradeoff: table memory grows as `2^(input_dim * D) * 4 B` (capped at
~256 KiB; past that the table is skipped) and build time is linear in
`2^(input_dim * D)` function evaluations.

## Thread safety

Once `baobzi::fit(...)` returns, the resulting `Function` is immutable and
its `operator()` is safe to call concurrently from multiple threads. Each
call must write to a disjoint slice of `res[]`; per-call scratch is held in
`thread_local` storage and the eval path never mutates shared state. Baobzi
does not parallelize internally — chunk your inputs and spawn threads
yourself; this contract makes that pattern safe. Pinned by
`tests/test_threadsafe.cpp`.

## Limitations

* Baobzi can use a _lot_ of memory on oscillatory or rapidly-varying
  functions. If your function is periodic, fit one period.
* Baobzi can't fit through a singularity — either shift the domain off the
  singular point or piecewise a small set of fits around it.
* `baobzi` dimensions are defined on the semi-open interval `[x0, x1)`.
  Out-of-domain evaluations return `NaN`.
* No bounds checking on user-provided inputs — it is up to the caller to
  sanitise.

## Why the name?

It's a cute version of *baobab*, the tree of life. The baobab lives an
extraordinarily long time, which is what this interpolator is meant to do —
fit once, evaluate forever.
