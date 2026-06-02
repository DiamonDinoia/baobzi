"""float32_fit.py — f32 dtype fit and evaluation."""
import math
import numpy as np
import baobzi


def func(x):
    return float(math.sin(x[0]))


approx = baobzi.fit(func, 0.0, math.pi, tol=1e-4, dtype="f32")
print(f"dtype={approx.dtype}  dim={approx.dim}")

xs = np.linspace(0.0, math.pi, 50, dtype=np.float32)
ys = approx(xs)
assert ys.dtype == np.float32
max_err = float(np.max(np.abs(ys.astype(np.float64) - np.sin(xs))))
print(f"max |approx - sin| = {max_err:.3e}")
assert max_err < 1e-3
print("OK")
