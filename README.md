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

## Quick start

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

## Including in your CMake project

```cmake
add_subdirectory(extern/baobzi)      # or FetchContent_Declare(baobzi ...)
target_link_libraries(your_target PUBLIC polyfit::polyfit poet::poet)
target_include_directories(your_target PRIVATE extern/baobzi/include)
```

Baobzi itself is header-only; polyfit and POET provide the leaf evaluators
and compile-time dispatch.

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
