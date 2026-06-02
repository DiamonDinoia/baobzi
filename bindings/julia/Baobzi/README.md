# Baobzi.jl

Julia bindings for [baobzi](https://github.com/flatironinstitute/baobzi) — a
piecewise-Chebyshev function approximator for smooth functions over
axis-aligned box domains in 1–3 dimensions.

## Requirements

- Julia 1.9+
- A built `libbaobzi_c.so` (the C ABI layer over the C++ library). It is
  located automatically, in order: the `LIBBAOBZI_C` environment variable →
  `deps/deps.jl` (written by `Pkg.build("Baobzi")`) → the loader search path →
  a sibling CMake `build*/libbaobzi_c.<ext>`. So no env var is needed once the
  project has been built; set `LIBBAOBZI_C` to override.

> **Note:** JLL / artifact packaging is not yet implemented.  Follow-up work
> should wrap the library in a `Baobzi_jll` package so that `Pkg.add("Baobzi")`
> works without manual library management.

## Quick start

```julia
using Pkg
Pkg.activate("path/to/bindings/julia/Baobzi")
Pkg.instantiate()

# ENV["LIBBAOBZI_C"] = "/path/to/libbaobzi_c.so"   # optional; auto-discovered
using Baobzi

# 1D scalar
b = fit(x -> exp(0.5x) + sin(3x), 0.0, 1.0, 1e-10)
b(0.5)                   # evaluate

# 2D -> 3D vector
b2 = fit((x,y) -> (sin(x)*cos(y), x+y, x*y), [0.0,0.0], [1.0,1.0], 1e-8; out_dim=3)
b2([0.3, 0.7])           # Vector{Float64} of length 3

# Batch eval (100 points, 2D)
X = rand(100, 2)
R = eval_multi(b2, X)    # 100×3 Matrix{Float64}

# Sorted 1D batch
xs = sort(rand(500))
r  = eval_sorted(b, xs)  # Vector{Float64}
```

## API

### `fit(f, a, b, tol; kwargs...) -> BaobziFn{T}`

| kwarg      | type / default       | meaning                                         |
|------------|----------------------|-------------------------------------------------|
| `dim`      | `Int`, `length(a)`   | input dimension (1–3)                           |
| `out_dim`  | `Int`, `1`           | output dimension (1–3)                          |
| `degree`   | `Int`, `11`          | Chebyshev degree per leaf (7, 11, or 15)        |
| `dtype`    | `Float64`            | element type (`Float64` or `Float32`)           |
| `policy`   | `BAOBZI_BALANCED`    | eval policy (latency / throughput / balanced)   |
| `options`  | `BaobziOptions()`    | advanced knobs (depth, memory budget, …)        |

### User-function calling convention

| dim  | call signature          | return (out_dim==1) | return (out_dim>1)                    |
|------|-------------------------|---------------------|---------------------------------------|
| 1    | `f(x::T)`               | scalar              | indexable length-`out_dim`            |
| > 1  | `f(x1::T, x2::T, …)`   | scalar              | indexable length-`out_dim`            |

Coordinates are passed as separate scalar arguments (splatted from an
`NTuple`), so a 2D function is simply `(x, y) -> ...`.

### `BaobziOptions`

```julia
BaobziOptions(;
    tol_kind               = BAOBZI_RELATIVE_MAX,   # tolerance interpretation
    max_depth              = 50,                     # max tree depth
    max_memory_mib         = 4,                      # memory budget in MiB
    allow_max_depth_leaves = 0,                      # bool: allow leaves at max depth
    min_uniform_depth      = 0)                      # force uniform refinement up to this depth
```

### Eval functions

| function                     | description                                    |
|------------------------------|------------------------------------------------|
| `b(x)`                       | single point, scalar or `Vector` result        |
| `eval_multi(b, X)`           | `n` points; `X` is `Vector` (1D) or `n×dim` matrix |
| `eval_sorted(b, xs)`         | sorted 1D batch, requires `dim==1`             |
| `memory_usage(b)`            | bytes used by the approximation tree           |
| `print_stats(b)`             | print internal stats to stdout                 |

## Running tests

```bash
LIBBAOBZI_C=/path/to/libbaobzi_c.so \
  julia --project=bindings/julia/Baobzi -e 'using Pkg; Pkg.test()'
```
