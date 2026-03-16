# scurve/models/gompertz.py
"""
Gompertz S-curve model.

The Gompertz curve is an asymmetric sigmoid often used for growth/adoption:
    y(t) = L * exp( -exp( -(t - t0)/k ) )

Where:
- L  : upper asymptote (cap / saturation level), L > 0
- t0 : inflection time (where growth rate peaks)
- k  : scale / time constant controlling steepness, k > 0

This module provides:
- GompertzCurve: parameterized curve with predict() and basic utilities.
- GompertzFitter: simple numerical fit for (L, t0, k) using coarse-to-fine grid
  search + local refinement (derivative-free). Designed for determinism and
  low-dependency environments.

Notes:
- This is not intended to be a high-performance optimizer.
- Works with numpy arrays or pandas Series.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

Array1D = Union[np.ndarray, pd.Series, list]


def _as_1d(x: Array1D) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float, copy=False)
    xn = np.asarray(x, dtype=float)
    if xn.ndim != 1:
        xn = xn.reshape(-1)
    return xn


def _safe_exp(x: np.ndarray, clip: float = 60.0) -> np.ndarray:
    # Clip exponent to avoid overflow
    return np.exp(np.clip(x, -clip, clip))


@dataclass(frozen=True)
class GompertzParams:
    L: float
    t0: float
    k: float

    def validate(self) -> None:
        if not np.isfinite(self.L) or self.L <= 0:
            raise ValueError("L must be finite and > 0.")
        if not np.isfinite(self.t0):
            raise ValueError("t0 must be finite.")
        if not np.isfinite(self.k) or self.k <= 0:
            raise ValueError("k must be finite and > 0.")


class GompertzCurve:
    def __init__(self, params: GompertzParams):
        params.validate()
        self.params = params

    def predict(self, t: Array1D) -> np.ndarray:
        t = _as_1d(t)
        L, t0, k = self.params.L, self.params.t0, self.params.k
        z = -(t - t0) / k
        # y = L * exp( -exp( z ) )
        y = L * _safe_exp(-_safe_exp(z))
        return y

    def derivative(self, t: Array1D) -> np.ndarray:
        """
        dy/dt for Gompertz:
            y(t) = L exp(-e^z), z = -(t-t0)/k
            dy/dt = (L/k) * e^z * exp(-e^z) * exp(-?)? Derivation yields:
            dy/dt = (L/k) * exp(z) * exp(-exp(z)) * exp(-?)? Actually:
            y = L * exp(-exp(z))
            dy/dz = y * (-exp(z))'?? -> dy/dz = y * (-exp(z))
            dz/dt = -1/k
            dy/dt = (y * (-exp(z))) * (-1/k) = (y * exp(z))/k
        """
        t = _as_1d(t)
        L, t0, k = self.params.L, self.params.t0, self.params.k
        z = -(t - t0) / k
        y = L * _safe_exp(-_safe_exp(z))
        return (y * _safe_exp(z)) / k

    def inflection_value(self) -> float:
        """
        For Gompertz, inflection occurs at t=t0 and y(t0)=L/e.
        """
        return float(self.params.L / np.e)

    def to_dict(self) -> Dict[str, Any]:
        return {"L": self.params.L, "t0": self.params.t0, "k": self.params.k}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GompertzCurve":
        return cls(GompertzParams(L=float(d["L"]), t0=float(d["t0"]), k=float(d["k"])))


@dataclass
class GompertzFitConfig:
    """
    Fit configuration.

    loss:
        "mse" or "mae"
    grid_sizes:
        Tuple of coarse grid sizes for (L, t0, k).
    refine_steps:
        Number of refinement rounds after coarse fit.
    refine_shrink:
        Shrink factor for search box each refinement round.
    k_bounds:
        Absolute bounds for k (min,max).
    L_bounds_mult:
        If L_bounds not provided, infer as [max(y)*m0, max(y)*m1] with this tuple.
    """

    loss: str = "mse"
    grid_sizes: Tuple[int, int, int] = (20, 25, 20)
    refine_steps: int = 6
    refine_shrink: float = 0.55
    k_bounds: Tuple[float, float] = (1e-3, 1e6)
    L_bounds_mult: Tuple[float, float] = (0.8, 1.6)


class GompertzFitter:
    """
    Deterministic derivative-free fitter for Gompertz parameters.

    Usage:
        fitter = GompertzFitter()
        curve, info = fitter.fit(t, y)
    """

    def __init__(self, config: Optional[GompertzFitConfig] = None):
        self.config = config or GompertzFitConfig()

    def _loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        if self.config.loss.lower() == "mae":
            return float(np.mean(np.abs(y_true - y_pred)))
        return float(np.mean((y_true - y_pred) ** 2))

    def _eval(
        self, t: np.ndarray, y: np.ndarray, L: float, t0: float, k: float
    ) -> float:
        if not (np.isfinite(L) and np.isfinite(t0) and np.isfinite(k)):
            return float("inf")
        if L <= 0 or k <= 0:
            return float("inf")
        # Prediction
        z = -(t - t0) / k
        yhat = L * _safe_exp(-_safe_exp(z))
        return self._loss(y, yhat)

    def _linspace_safe(self, a: float, b: float, n: int) -> np.ndarray:
        if n <= 1:
            return np.array([(a + b) / 2.0], dtype=float)
        if a == b:
            return np.full(n, a, dtype=float)
        return np.linspace(a, b, n, dtype=float)

    def fit(
        self,
        t: Array1D,
        y: Array1D,
        *,
        L_bounds: Optional[Tuple[float, float]] = None,
        t0_bounds: Optional[Tuple[float, float]] = None,
        k_bounds: Optional[Tuple[float, float]] = None,
        weights: Optional[Array1D] = None,
    ) -> Tuple[GompertzCurve, Dict[str, Any]]:
        t = _as_1d(t)
        y = _as_1d(y)

        if t.shape[0] != y.shape[0]:
            raise ValueError("t and y must have the same length.")
        if t.shape[0] < 5:
            raise ValueError("Need at least 5 observations to fit Gompertz.")

        # Sort by t for stability
        idx = np.argsort(t)
        t = t[idx]
        y = y[idx]

        # Handle weights (optional)
        if weights is not None:
            w = _as_1d(weights)[idx]
            w = np.clip(w, 0.0, np.inf)
            if np.all(w == 0):
                raise ValueError("All weights are zero.")
            w = w / np.mean(w)
        else:
            w = None

        # Default bounds inference
        y_max = float(np.nanmax(y))
        y_min = float(np.nanmin(y))
        if not np.isfinite(y_max) or y_max <= 0:
            raise ValueError("y must contain positive finite values to fit Gompertz.")

        if L_bounds is None:
            m0, m1 = self.config.L_bounds_mult
            L_bounds = (max(1e-9, y_max * m0), max(1e-9, y_max * m1))
        if t0_bounds is None:
            t0_bounds = (float(t[0]), float(t[-1]))
        if k_bounds is None:
            k_bounds = self.config.k_bounds

        # Ensure bounds are sane
        L_lo, L_hi = map(float, L_bounds)
        t0_lo, t0_hi = map(float, t0_bounds)
        k_lo, k_hi = map(float, k_bounds)

        L_lo = max(L_lo, 1e-12)
        L_hi = max(L_hi, L_lo * 1.0001)
        k_lo = max(k_lo, 1e-12)
        k_hi = max(k_hi, k_lo * 1.0001)
        if t0_hi <= t0_lo:
            t0_hi = t0_lo + 1e-9

        # Precompute for weighted loss, if used
        def loss_fn(y_true: np.ndarray, y_pred: np.ndarray) -> float:
            if w is None:
                return self._loss(y_true, y_pred)
            if self.config.loss.lower() == "mae":
                return float(np.sum(w * np.abs(y_true - y_pred)) / np.sum(w))
            return float(np.sum(w * (y_true - y_pred) ** 2) / np.sum(w))

        def eval_params(L: float, t0: float, k: float) -> float:
            if L <= 0 or k <= 0:
                return float("inf")
            z = -(t - t0) / k
            yhat = L * _safe_exp(-_safe_exp(z))
            return loss_fn(y, yhat)

        # Coarse grid search
        nL, nt0, nk = self.config.grid_sizes
        L_grid = self._linspace_safe(L_lo, L_hi, int(nL))
        t0_grid = self._linspace_safe(t0_lo, t0_hi, int(nt0))

        # Use log-space for k for better coverage
        logk_lo, logk_hi = np.log(k_lo), np.log(k_hi)
        k_grid = np.exp(self._linspace_safe(logk_lo, logk_hi, int(nk)))

        best = {"L": None, "t0": None, "k": None, "loss": float("inf")}
        for Lc in L_grid:
            for t0c in t0_grid:
                for kc in k_grid:
                    val = eval_params(float(Lc), float(t0c), float(kc))
                    if val < best["loss"]:
                        best = {"L": float(Lc), "t0": float(t0c), "k": float(kc), "loss": float(val)}

        # Local refinement: shrink a box around best
        Lc, t0c, kc = best["L"], best["t0"], best["k"]

        for _ in range(int(self.config.refine_steps)):
            shrink = float(self.config.refine_shrink)

            # Define local bounds
            L_span = (L_hi - L_lo) * shrink
            t0_span = (t0_hi - t0_lo) * shrink
            # multiplicative span for k in log space
            logk_span = (logk_hi - logk_lo) * shrink

            L_lo2 = max(1e-12, Lc - 0.5 * L_span)
            L_hi2 = max(L_lo2 * 1.0001, Lc + 0.5 * L_span)

            t0_lo2 = t0c - 0.5 * t0_span
            t0_hi2 = t0c + 0.5 * t0_span
            if t0_hi2 <= t0_lo2:
                t0_hi2 = t0_lo2 + 1e-9

            logkc = float(np.log(max(kc, 1e-12)))
            logk_lo2 = logkc - 0.5 * logk_span
            logk_hi2 = logkc + 0.5 * logk_span

            # Clamp to global bounds
            L_lo2 = max(L_lo2, L_lo)
            L_hi2 = min(L_hi2, L_hi)
            t0_lo2 = max(t0_lo2, t0_lo)
            t0_hi2 = min(t0_hi2, t0_hi)
            logk_lo2 = max(logk_lo2, logk_lo)
            logk_hi2 = min(logk_hi2, logk_hi)

            # Search a smaller grid
            L_grid2 = self._linspace_safe(L_lo2, L_hi2, max(8, nL // 3))
            t0_grid2 = self._linspace_safe(t0_lo2, t0_hi2, max(10, nt0 // 3))
            k_grid2 = np.exp(self._linspace_safe(logk_lo2, logk_hi2, max(8, nk // 3)))

            improved = False
            for Lr in L_grid2:
                for t0r in t0_grid2:
                    for kr in k_grid2:
                        val = eval_params(float(Lr), float(t0r), float(kr))
                        if val < best["loss"]:
                            best = {"L": float(Lr), "t0": float(t0r), "k": float(kr), "loss": float(val)}
                            improved = True

            # Update center and tighten global search box for next refinement
            Lc, t0c, kc = best["L"], best["t0"], best["k"]
            if improved:
                L_lo, L_hi = L_lo2, L_hi2
                t0_lo, t0_hi = t0_lo2, t0_hi2
                logk_lo, logk_hi = logk_lo2, logk_hi2

        params = GompertzParams(L=float(best["L"]), t0=float(best["t0"]), k=float(best["k"]))
        curve = GompertzCurve(params)

        info = {
            "loss": float(best["loss"]),
            "config": {
                "loss": self.config.loss,
                "grid_sizes": self.config.grid_sizes,
                "refine_steps": self.config.refine_steps,
                "refine_shrink": self.config.refine_shrink,
                "k_bounds": self.config.k_bounds,
                "L_bounds_mult": self.config.L_bounds_mult,
            },
            "bounds_used": {
                "L_bounds": (float(L_bounds[0]), float(L_bounds[1])),
                "t0_bounds": (float(t0_bounds[0]), float(t0_bounds[1])),
                "k_bounds": (float(k_bounds[0]), float(k_bounds[1])),
            },
            "y_summary": {
                "min": float(y_min),
                "max": float(y_max),
                "n": int(len(y)),
            },
        }
        return curve, info