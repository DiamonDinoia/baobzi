"""pytest suite for the baobzi Python bindings.

Each test is self-contained and exercises a specific part of the API.
"""

import math
import numpy as np
import pytest
import baobzi


# ---------------------------------------------------------------------------
# 1. Accuracy: 1D scalar fit
# ---------------------------------------------------------------------------

def test_accuracy_1d():
    """fit exp(0.5x)+sin(3x) on [0,1]; eval agrees to < 1e-5."""

    def func(x):
        v = x[0]
        return math.exp(0.5 * v) + math.sin(3.0 * v)

    approx = baobzi.fit(func, 0.0, 1.0, tol=1e-8)
    assert approx.dim == 1
    assert approx.out_dim == 1
    assert approx.dtype == "f64"
    assert approx.memory_usage > 0

    # baobzi's upper domain corner is exclusive (eval at exactly b yields NaN,
    # the out-of-domain sentinel), so sample [a, b) with endpoint=False.
    xs = np.linspace(0.0, 1.0, 200, endpoint=False)
    exact = np.exp(0.5 * xs) + np.sin(3.0 * xs)
    approx_vals = approx(xs)
    assert approx_vals.shape == (200,)
    max_err = np.max(np.abs(approx_vals - exact))
    assert max_err < 1e-5, f"max error {max_err} exceeds 1e-5"


# ---------------------------------------------------------------------------
# 2. AoS / SoA / sorted parity — 2D → 3D
# ---------------------------------------------------------------------------

def test_soa_aos_parity_2d_3out():
    """AoS eval_multi equals SoA eval_multi_soa component-by-component."""

    def func(x):
        return np.array([
            math.exp(0.3 * x[0]) + math.sin(2.0 * x[1]),
            math.cos(x[0] * x[1]) + 2.0,
            x[0] ** 2 + x[1] + 1.0,
        ])

    approx = baobzi.fit(func, [0.2, 0.2], [1.5, 1.5], tol=1e-7, out_dim=3)
    assert approx.dim == 2
    assert approx.out_dim == 3

    N = 64
    rng = np.random.default_rng(42)
    xs = rng.uniform([[0.2, 0.2]], [[1.5, 1.5]], size=(N, 2))

    aos = approx.eval_multi(xs)
    soa = approx.eval_multi_soa(xs)

    assert aos.shape == (N, 3)
    assert len(soa) == 3
    for d in range(3):
        np.testing.assert_array_equal(
            aos[:, d], soa[d],
            err_msg=f"AoS and SoA differ for component {d}"
        )


# ---------------------------------------------------------------------------
# 3. sorted == eval_multi for 1D (bit-exact on sorted input)
# ---------------------------------------------------------------------------

def test_sorted_equals_eval_multi_1d():
    """sorted() on sorted input is bit-exact with eval_multi()."""

    def func(x):
        return math.sin(x[0]) * math.exp(-0.2 * x[0])

    approx = baobzi.fit(func, 0.0, 5.0, tol=1e-9)

    xs = np.sort(np.random.default_rng(7).uniform(0.0, 5.0, 256))
    multi_res = approx.eval_multi(xs)
    sorted_res = approx.sorted(xs)
    np.testing.assert_array_equal(
        multi_res, sorted_res,
        err_msg="sorted() and eval_multi() differ on sorted input"
    )


# ---------------------------------------------------------------------------
# 4. NaN for out-of-domain points
# ---------------------------------------------------------------------------

def test_nan_out_of_domain():
    """Evaluating outside [a, b] returns NaN."""

    def func(x):
        return math.exp(x[0])

    approx = baobzi.fit(func, 0.0, 1.0, tol=1e-8)

    x_outside = np.array([-0.5])
    result = approx(x_outside)
    assert np.all(np.isnan(result)), f"Expected NaN, got {result}"

    x_outside2 = np.array([2.0])
    result2 = approx(x_outside2)
    assert np.all(np.isnan(result2)), f"Expected NaN, got {result2}"


# ---------------------------------------------------------------------------
# 5. Raising Python callback propagates the exception
# ---------------------------------------------------------------------------

def test_raising_callback():
    """A callback that raises ValueError should make fit() raise."""
    first_call = [True]

    def bad_func(x):
        if first_call[0]:
            first_call[0] = False
            raise ValueError("intentional test error")
        return 0.0

    with pytest.raises((ValueError, RuntimeError)):
        baobzi.fit(bad_func, 0.0, 1.0, tol=1e-6)


# ---------------------------------------------------------------------------
# 6. Too-tight tolerance / low max_depth raises RuntimeError
# ---------------------------------------------------------------------------

def test_max_depth_exceeded():
    """Overly tight tolerance with tiny max_depth raises RuntimeError."""

    def hard_func(x):
        # Highly oscillatory — hard to fit tightly in 2 levels.
        return math.sin(200.0 * x[0])

    with pytest.raises(RuntimeError, match="(?i)(maxdepth|depth|memory|budget|failed)"):
        baobzi.fit(
            hard_func, 0.0, 1.0,
            tol=1e-14,
            max_depth=2,
        )


# ---------------------------------------------------------------------------
# 7. float32 path
# ---------------------------------------------------------------------------

def test_float32_path():
    """The f32 dtype path fits and evaluates correctly."""

    def func(x):
        return float(math.exp(x[0]))

    approx = baobzi.fit(func, 0.0, 1.0, tol=1e-4, dtype="f32")
    assert approx.dtype == "f32"

    xs = np.linspace(0.0, 1.0, 50, endpoint=False, dtype=np.float32)
    result = approx(xs)
    assert result.dtype == np.float32

    exact = np.exp(xs)
    max_err = float(np.max(np.abs(result.astype(np.float64) - exact)))
    assert max_err < 1e-3, f"f32 max error {max_err} too large"


# ---------------------------------------------------------------------------
# 8. Scalar __call__ for 1D
# ---------------------------------------------------------------------------

def test_scalar_call_1d():
    """BaobziFunction.__call__ accepts a Python scalar for dim==1."""

    def func(x):
        return math.exp(x[0])

    approx = baobzi.fit(func, 0.0, 1.0, tol=1e-8)
    result = approx(0.5)
    assert isinstance(result, float)
    assert abs(result - math.exp(0.5)) < 1e-7


# ---------------------------------------------------------------------------
# 9. Vector-valued 1D fit (input_dim=1, output_dim=2)
# ---------------------------------------------------------------------------

def test_vector_output_1d():
    """1D → 2D vector-valued fit."""

    def func(x):
        return np.array([math.sin(x[0]), math.cos(x[0])])

    approx = baobzi.fit(func, 0.0, math.pi, tol=1e-7, out_dim=2)
    assert approx.dim == 1
    assert approx.out_dim == 2

    xs = np.linspace(0.1, math.pi - 0.1, 100)
    result = approx.eval_multi(xs)
    assert result.shape == (100, 2)
    np.testing.assert_allclose(result[:, 0], np.sin(xs), atol=1e-5)
    np.testing.assert_allclose(result[:, 1], np.cos(xs), atol=1e-5)
