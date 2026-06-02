"""baobzi — piecewise-Chebyshev function approximation.

Public API
----------
fit(f, a, b, tol, *, ...)  -> BaobziFunction
    Fit a Python callable and return an evaluator.

BaobziFunction
    Callable evaluator with .eval_multi(), .sorted(), .eval_multi_soa(),
    .memory_usage, .dtype, .dim, .out_dim properties.
"""

from __future__ import annotations

from typing import Callable, Sequence, Union
import math
import numpy as np

from . import _baobzi  # compiled extension

__all__ = ["fit", "BaobziFunction"]

# ---------------------------------------------------------------------------
# String → integer maps used by fit()
# ---------------------------------------------------------------------------

_TOL_KIND = {
    "relative_tail": 0,
    "absolute_tail": 1,
    "relative_max":  2,
    "absolute_max":  3,
    "relative_l2":   4,
    "absolute_l2":   5,
}

_POLICY = {
    "latency":    0,
    "throughput": 1,
    "balanced":   2,
}


# ---------------------------------------------------------------------------
# Public BaobziFunction wrapper
# ---------------------------------------------------------------------------

class BaobziFunction:
    """Evaluator for a fitted baobzi approximation.

    Do not construct directly; use :func:`baobzi.fit`.
    """

    def __init__(self, inner: _baobzi.BaobziFunction) -> None:
        self._inner = inner

    # ---- properties -------------------------------------------------------

    @property
    def dim(self) -> int:
        """Input dimensionality."""
        return self._inner.input_dim

    @property
    def out_dim(self) -> int:
        """Output dimensionality."""
        return self._inner.output_dim

    @property
    def memory_usage(self) -> int:
        """Approximate memory used by the approximation (bytes)."""
        return self._inner.memory_usage

    @property
    def dtype(self) -> str:
        """Value type: ``'f64'`` or ``'f32'``."""
        return self._inner.dtype

    # ---- evaluation -------------------------------------------------------

    def __call__(self, x):
        """Evaluate at *x*.

        Parameters
        ----------
        x : scalar | array-like of shape (dim,) | array-like of shape (N, dim)
            Input point(s). For ``dim == 1``, a 1-D array ``(N,)`` is also
            accepted for batch evaluation.

        Returns
        -------
        scalar, (out_dim,) array, (N,) array, or (N, out_dim) array
            Scalar when ``x`` is a scalar and ``out_dim == 1``; squeezed
            shapes otherwise.
        """
        x = np.asarray(x)
        scalar_in = x.ndim == 0

        if scalar_in:
            # Single 0-d input
            return self._inner.eval_one(x.item())

        if x.ndim == 1:
            if self.dim == 1:
                # Could be a batch of N 1-D points, or a single multi-dim point.
                # If len == dim we treat it as a single point (non-ambiguous for
                # dim==1 N==1 case, which is fine as both interpretations agree).
                # For dim==1 we use batch if the array is longer than 1 element.
                if x.shape[0] == 1 or (self.dim == 1 and x.shape[0] >= 1):
                    # Use batch path: (N,) input, (N,) or (N, out_dim) output.
                    return self.eval_multi(x)
                else:
                    return self._inner.eval_one(x)
            else:
                # (dim,) single point
                return self._inner.eval_one(x)

        if x.ndim == 2:
            return self.eval_multi(x)

        raise ValueError(f"x must have ndim <= 2, got {x.ndim}")

    def eval_multi(self, x) -> np.ndarray:
        """AoS batch evaluation.

        Parameters
        ----------
        x : array-like of shape ``(N,)`` (dim==1) or ``(N, dim)``

        Returns
        -------
        (N,) or (N, out_dim) ndarray
        """
        np_mod = np
        dtype = np_mod.float32 if self.dtype == "f32" else np_mod.float64
        x = np_mod.ascontiguousarray(x, dtype=dtype)
        return self._inner.eval_multi(x)

    def sorted(self, x) -> np.ndarray:
        """Sorted-1D batch evaluation (``dim`` must be 1).

        Parameters
        ----------
        x : sorted 1-D array-like of length N

        Returns
        -------
        (N,) or (N, out_dim) ndarray
        """
        np_mod = np
        dtype = np_mod.float32 if self.dtype == "f32" else np_mod.float64
        x = np_mod.ascontiguousarray(x, dtype=dtype)
        return self._inner.sorted(x)

    def eval_multi_soa(self, x) -> list:
        """SoA batch evaluation.

        Returns a Python list of ``out_dim`` arrays of shape ``(N,)``.

        Parameters
        ----------
        x : array-like of shape ``(N,)`` or ``(N, dim)``
        """
        np_mod = np
        dtype = np_mod.float32 if self.dtype == "f32" else np_mod.float64
        x = np_mod.ascontiguousarray(x, dtype=dtype)
        return self._inner.eval_multi_soa(x)

    def print_stats(self) -> None:
        """Print internal tree statistics to stdout."""
        self._inner.print_stats()

    def __repr__(self) -> str:
        return (
            f"BaobziFunction(dim={self.dim}, out_dim={self.out_dim}, "
            f"dtype={self.dtype!r}, memory={self.memory_usage} B)"
        )


# ---------------------------------------------------------------------------
# fit() — main entry point
# ---------------------------------------------------------------------------

def fit(
    f: Callable,
    a,
    b,
    tol: float,
    *,
    dim: int | None = None,
    out_dim: int = 1,
    degree: int = 11,
    dtype: str = "f64",
    policy: str = "balanced",
    tol_kind: str = "relative_max",
    max_depth: int = 50,
    max_memory_mib: int = 4,
    allow_max_depth_leaves: bool = False,
    min_uniform_depth: int = 0,
) -> BaobziFunction:
    """Fit a Python callable and return a :class:`BaobziFunction` evaluator.

    Parameters
    ----------
    f : callable
        ``f(x) -> scalar`` or ``f(x) -> array(out_dim,)``.
        For f64 fits *x* is a ``float64`` ndarray of shape ``(dim,)``; for f32
        fits it is a ``float32`` ndarray.  For ``dim == 1`` x is a 1-element
        array (shape ``(1,)``).
    a, b : scalar or sequence of length *dim*
        Domain corners.  If scalars, ``dim`` is inferred as 1.
    tol : float
        Approximation tolerance.
    dim : int, optional
        Input dimension.  Inferred from ``a``/``b`` when not given.
    out_dim : int
        Output dimension (number of values returned by *f*).
    degree : int
        Chebyshev degree per leaf (7, 11, or 15).
    dtype : {'f64', 'f32'}
        Floating-point precision.
    policy : {'balanced', 'latency', 'throughput'}
        Eval strategy (currently a no-op in the backend, kept for API
        forward-compatibility).
    tol_kind : str
        Tolerance interpretation.  One of ``'relative_max'``,
        ``'absolute_max'``, ``'relative_l2'``, ``'absolute_l2'``,
        ``'relative_tail'``, ``'absolute_tail'``.
    max_depth : int
        Maximum adaptive tree depth.
    max_memory_mib : int
        Memory budget in MiB.
    allow_max_depth_leaves : bool
        Allow leaves at max depth (relaxes convergence).
    min_uniform_depth : int
        Minimum uniform refinement depth before adaptivity kicks in.

    Returns
    -------
    BaobziFunction

    Raises
    ------
    RuntimeError
        If the fit fails (MaxDepthExceeded, MemoryBudgetExceeded, …) or if the
        callback *f* raises (in which case the original exception propagates).
    """
    # ---- infer / validate dim -------------------------------------------
    try:
        a_seq = list(a)
    except TypeError:
        a_seq = [a]
    try:
        b_seq = list(b)
    except TypeError:
        b_seq = [b]

    inferred_dim = len(a_seq)
    if len(b_seq) != inferred_dim:
        raise ValueError("a and b must have the same length")

    if dim is None:
        dim = inferred_dim
    elif dim != inferred_dim:
        raise ValueError(
            f"dim={dim} but len(a)={inferred_dim}; they must agree"
        )

    # ---- validate choices -----------------------------------------------
    if degree not in (7, 11, 15):
        raise ValueError(f"degree must be 7, 11, or 15; got {degree}")
    if out_dim < 1 or out_dim > 3:
        raise ValueError(f"out_dim must be 1-3; got {out_dim}")
    if dim < 1 or dim > 3:
        raise ValueError(f"dim must be 1-3; got {dim}")
    if dtype not in ("f64", "f32"):
        raise ValueError(f"dtype must be 'f64' or 'f32'; got {dtype!r}")

    policy_int  = _POLICY.get(policy.lower())
    if policy_int is None:
        raise ValueError(f"Unknown policy {policy!r}; choose from {list(_POLICY)}")
    tol_kind_int = _TOL_KIND.get(tol_kind.lower())
    if tol_kind_int is None:
        raise ValueError(f"Unknown tol_kind {tol_kind!r}; choose from {list(_TOL_KIND)}")

    # ---- dispatch to the typed C-level fit ------------------------------
    common_kw = dict(
        input_dim=dim, output_dim=out_dim, degree=degree,
        tol=float(tol),
        policy=policy_int,
        tol_kind=tol_kind_int,
        max_depth=max_depth,
        max_memory_mib=max_memory_mib,
        allow_max_depth_leaves=int(allow_max_depth_leaves),
        min_uniform_depth=min_uniform_depth,
    )

    if dtype == "f64":
        inner = _baobzi.fit_f64(
            f,
            a=[float(v) for v in a_seq],
            b=[float(v) for v in b_seq],
            **common_kw,
        )
    else:
        inner = _baobzi.fit_f32(
            f,
            a=[float(v) for v in a_seq],
            b=[float(v) for v in b_seq],
            **common_kw,
        )

    return BaobziFunction(inner)
