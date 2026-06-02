"""simple_1d.py — minimal 1D -> 1D fit and evaluation."""
import math
import numpy as np
import baobzi


def func(x):
    return math.exp(x[0])


approx = baobzi.fit(func, 0.0, 1.0, tol=1e-10)
print(f"dtype={approx.dtype}  dim={approx.dim}  out_dim={approx.out_dim}  memory={approx.memory_usage} B")

xs = np.linspace(0.0, 1.0, 11)
ys = approx(xs)
exact = np.exp(xs)
max_err = float(np.max(np.abs(ys - exact)))
print(f"max |approx - exp| over 11 points: {max_err:.3e}")
assert max_err < 1e-8, f"too large: {max_err}"
print("OK")
