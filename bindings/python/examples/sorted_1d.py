"""sorted_1d.py — sorted 1D batch evaluation."""
import math
import numpy as np
import baobzi


def func(x):
    return math.sin(x[0]) * math.exp(-0.2 * x[0])


approx = baobzi.fit(func, 0.0, 5.0, tol=1e-9)

xs = np.sort(np.random.default_rng(42).uniform(0.0, 5.0, 256))
multi = approx.eval_multi(xs)
sorted_ = approx.sorted(xs)

diff = float(np.max(np.abs(multi - sorted_)))
print(f"max |eval_multi - sorted| = {diff:.3e} (expect 0)")
assert diff == 0.0
print("OK")
