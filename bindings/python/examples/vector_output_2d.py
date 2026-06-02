"""vector_output_2d.py — 2D -> 3D vector-valued fit; AoS/SoA parity check."""
import math
import numpy as np
import baobzi


def func(x):
    return np.array([
        math.exp(0.3 * x[0]) + math.sin(2.0 * x[1]),
        math.cos(x[0] * x[1]) + 2.0,
        x[0] ** 2 + x[1] + 1.0,
    ])


approx = baobzi.fit(func, [0.2, 0.2], [1.5, 1.5], tol=1e-8, out_dim=3)
print(f"dim={approx.dim}  out_dim={approx.out_dim}  memory={approx.memory_usage} B")

N = 64
rng = np.random.default_rng(0)
xs = rng.uniform([[0.2, 0.2]], [[1.5, 1.5]], size=(N, 2))

aos = approx.eval_multi(xs)
soa = approx.eval_multi_soa(xs)

max_diff = 0.0
for d in range(3):
    diff = float(np.max(np.abs(aos[:, d] - soa[d])))
    max_diff = max(max_diff, diff)

print(f"max |AoS - SoA| = {max_diff:.3e} (expect 0)")
assert max_diff == 0.0
print("OK")
