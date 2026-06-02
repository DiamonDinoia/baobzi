# baobzi host-language bindings

Thin wrappers that let you fit and evaluate a baobzi piecewise-Chebyshev
approximant from **Python**, **Julia**, or **MATLAB**, passing a *function
handle written in the host language* into the fit and getting C++ fit failures
back as native exceptions.

All three sit on the same C ABI (`libbaobzi_c`, see [`../include/baobzi.h`](../include/baobzi.h))
and reuse its pre-instantiated shapes — they do **not** re-instantiate the C++
templates. The mechanism is identical in spirit across languages: the wrapper
hands the C fit a **C function pointer** (a *trampoline*) plus an opaque `data`
pointer carrying the host callable; the trampoline re-enters the host language
once per Chebyshev sample.

| Supported | values |
|-----------|--------|
| input dim  | 1, 2, 3 |
| output dim | 1, 2, 3 (a 1-D vector-valued fit is `dim=1, out_dim>1`) |
| degree     | 7, 11, 15 |
| dtype      | `f64` (double), `f32` (float) |

**Domain note (all languages):** the lower corner `a` is inclusive but the
**upper corner `b` is exclusive** — evaluating exactly at `b` returns `NaN`
(the out-of-domain sentinel). Sample `[a, b)`.

**Reverse-exception safety (all languages):** if your host callback raises
*during* the fit, the trampoline catches it, writes `NaN`, short-circuits the
remaining samples (it never re-enters the host language again), lets the C fit
unwind cleanly, and then re-raises the original exception. A throwing callback
never crashes the interpreter / corrupts the C++ stack.

---

## Python (nanobind)

```bash
pip install ./bindings/python        # scikit-build-core fetches nanobind, builds the extension
pytest bindings/python/tests
```

```python
import numpy as np, baobzi

f = baobzi.fit(lambda x: np.exp(0.5*x[0]) + np.sin(3*x[0]), 0.0, 1.0, tol=1e-8)
f(0.5)                 # scalar
f(np.linspace(0, 1, 100, endpoint=False))   # vectorized -> (100,)
f.memory_usage, f.dim, f.out_dim, f.dtype

# 2D -> 3D vector-valued
g = baobzi.fit(lambda x: np.array([np.exp(0.3*x[0]) + np.sin(2*x[1]),
                                   np.cos(x[0]*x[1]) + 2.0,
                                   x[0]**2 + x[1] + 1.0]),
               [0.2, 0.2], [1.5, 1.5], tol=1e-7, out_dim=3)
g(np.random.uniform([0.2, 0.2], [1.5, 1.5], size=(64, 2)))   # -> (64, 3)
```

A C++ fit failure raises `RuntimeError(baobzi_last_error())`; a too-tight `tol`
with a low `max_depth` raises with `MaxDepthExceeded` / `MemoryBudgetExceeded`
in the message. See [`python/README.md`](python/README.md).

## Julia

```bash
# No env var needed: the package walks up to find a sibling build*/ tree.
julia --project=bindings/julia/Baobzi -e 'using Pkg; Pkg.test()'
```

The library is located in order: `LIBBAOBZI_C` (explicit path) → `deps/deps.jl`
(written by `Pkg.build("Baobzi")`) → the loader search path → a sibling CMake
`build*/libbaobzi_c.<ext>`. So an in-repo `Pkg.test()` works once the project
has been built; set `LIBBAOBZI_C=/path/to/libbaobzi_c.so` to override.

```julia
using Baobzi
b = fit(x -> exp(0.5x) + sin(3x), 0.0, 1.0, 1e-8)   # dim==1: f(scalar)
b(0.5)
eval_multi(b, collect(range(0, 1, length=100)))      # dim==1 -> length-100 vector

# 2D -> 3D: f(coords...) returns an indexable of length out_dim
g = fit((x, y) -> (exp(0.3x) + sin(2y), cos(x*y) + 2, x^2 + y + 1),
        [0.2, 0.2], [1.5, 1.5], 1e-7; out_dim=3)
g([0.5, 0.7])                                         # length-3 Vector
```

A `NULL` fit raises `error(baobzi_last_error())`; a raising closure is re-thrown
after the fit. See [`julia/Baobzi/README.md`](julia/Baobzi/README.md).

## MATLAB (MEX)

mwrap is the real generator: [`baobzi.mw`](matlab/baobzi.mw) is the source of
truth, and the generated gateway (`baobzi_mex_gen.cpp`) + `bz_*.m` stubs are
committed. A plain build compiles the committed gateway — no mwrap or network
needed:

```bash
# Build the gateway (no MATLAB license needed; -static-libstdc++ keeps the
# .mexa64 independent of MATLAB's bundled libstdc++).
make -C bindings/matlab MATLAB_ROOT=/usr/local/MATLAB/R2025b
matlab -batch "run('bindings/matlab/test_baobzi.m')"
```

To change the binding, edit `baobzi.mw` and regenerate with `make -C
bindings/matlab mwrap_gen` (or the CMake `regenerate-matlab` target). The CMake
glue regenerates automatically when `baobzi.mw` is newer than the committed
gateway, discovering `mwrap` on `PATH` or cloning + building the official mwrap
(plain `make` — flex/bison are needed only to modify mwrap's grammar).

```matlab
f   = @(x) exp(0.5*x(1)) + sin(3*x(1));
obj = baobzi(f, 0, 1, 1e-8, 'dim', 1, 'out_dim', 1);
obj(linspace(0, 1, 100)')          % subsref syntax; or obj.eval(...)

g = baobzi(@(x) [exp(0.3*x(1))+sin(2*x(2)), cos(x(1)*x(2))+2, x(1)^2+x(2)+1], ...
           [0.2 0.2], [1.5 1.5], 1e-7, 'dim', 2, 'out_dim', 3);
g(rand(100,2)*1.3 + 0.2)           % -> 100x3
```

A C++ fit failure raises `baobzi:fit`; a callback error raises `baobzi:callback`.
See [`matlab/README.md`](matlab/README.md).

> **libstdc++ / GLIBCXX note (fallback only).** The canonical `Makefile` (and
> the CMake glue) link the C++ runtime statically (`-static-libstdc++
> -static-libgcc`), so the `.mexa64` does not depend on MATLAB's bundled
> libstdc++. Only if you build the mex some other way and MATLAB reports
> `version GLIBCXX_3.4.xx not found` do you need a workaround — e.g. launch
> MATLAB with `LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6`.

---

## Building via the top-level CMake

The wrappers are also wired into CTest, each behind an option (all default OFF):

```bash
cmake -B build -DBAOBZI_BUILD_PYTHON=ON -DBAOBZI_BUILD_JULIA=ON -DBAOBZI_BUILD_MATLAB=ON
cmake --build build
ctest --test-dir build -R "python_baobzi|julia_baobzi|matlab_baobzi"
```

Missing toolchains are detected and skipped with a STATUS message rather than
failing configuration. (The Python CTest additionally needs `numpy` + `pytest`
in the interpreter CMake selected; the Julia CTest passes `LIBBAOBZI_C` pointing
at the freshly-built shared library.)

## Cross-language parity

[`parity/run_parity.sh`](parity/run_parity.sh) fits the same 2-D → 3-D kernel in
C (the reference), Python, and Julia, evaluates a fixed point set, and checks
that every language agrees with the C result. See that directory for details.
