# baobzi Python bindings

Python (nanobind) wrapper for the [baobzi](../../README.md) piecewise-Chebyshev
function approximator.

## Installation

```bash
pip install scikit-build-core nanobind numpy
pip install .          # regular install
# or for development:
pip install -e . --no-build-isolation
```

## Quick start

```python
import math
import numpy as np
import baobzi

# Fit exp(x) on [0, 1] to 1e-10 relative tolerance.
def func(x):
    return math.exp(x[0])   # x is a (dim,) ndarray

approx = baobzi.fit(func, 0.0, 1.0, tol=1e-10)

# Scalar eval
print(approx(0.5))          # ≈ exp(0.5)

# Batch eval
xs = np.linspace(0, 1, 1000)
ys = approx(xs)             # shape (1000,)
```

## API

### `baobzi.fit(f, a, b, tol, *, dim=None, out_dim=1, degree=11, dtype="f64", ...)`

Fits callable `f` and returns a `BaobziFunction`.

- `a`, `b`: domain corners (scalars for 1D, length-`dim` lists otherwise).
- `tol`: approximation tolerance.
- `out_dim`: number of output components (`f` returns a scalar or `(out_dim,)` array).
- `dtype`: `"f64"` (default) or `"f32"`.
- `degree`: Chebyshev degree per leaf — 7, 11 (default), or 15.

### `BaobziFunction`

| Method / property | Description |
|---|---|
| `fn(x)` | Evaluate; accepts scalar, 1-D point, or `(N, dim)` array |
| `fn.eval_multi(x)` | AoS batch; returns `(N,)` or `(N, out_dim)` |
| `fn.sorted(x)` | Sorted 1D batch (dim must be 1) |
| `fn.eval_multi_soa(x)` | SoA batch; returns list of `(N,)` arrays |
| `.dim` | Input dimensionality |
| `.out_dim` | Output dimensionality |
| `.dtype` | `"f64"` or `"f32"` |
| `.memory_usage` | Bytes used by the approximation |
| `.print_stats()` | Print internal tree statistics |

## Running tests

```bash
pip install pytest
pytest tests/ -q
```
