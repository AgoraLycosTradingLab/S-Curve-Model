# scurve/fit/diagnostics.py
"""
Fit diagnostics for S-Curve models.

Purpose
- Provide deterministic, dependency-light diagnostics for curve fits.
- Standardize metrics across curve families (Gompertz, Bass).
- Provide guardrails for "sanity" checks that gate downstream scoring.

This module is used by:
- scurve.fit.fitters (after fitting) to compute metrics and flags
- scurve.fit.gates (to decide pass/fail/fallback)
- report/drift checks

Key metrics
- mse, rmse, mae
- r2
- rmse_norm_range (RMSE / (y_max - y_min))
- rmse_norm_scale (RMSE / max(|y|))
- bound_flags (if params at/near bounds)
- monotonicity of fitted curve on the observed grid
- residual diagnostics (mean/std, outlier rate)

No scipy/sklearn required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from scurve.models.gompertz import GompertzCurve
from scurve.models.bass import BassCurve

Array1D = Union[np.ndarray, pd.Series, list]


def _as_1d(x: Array1D) -> np.ndarray:
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float, copy=False)
    xn = np.asarray(x, dtype=float)
    if xn.ndim != 1:
        xn = xn.reshape(-1)
    return xn


def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return float(default)
    return float(a / b)


def _mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean((y - yhat) ** 2))


def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(_mse(y, yhat)))


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _monotonicity_score(y: np.ndarray) -> float:
    if len(y) < 2:
        return np.nan
    dy = np.diff(y)
    return float(np.mean(dy >= 0))


def _outlier_rate(resid: np.ndarray, z: float = 3.0) -> float:
    if resid.size == 0:
        return np.nan
    sd = float(np.std(resid))
    if not np.isfinite(sd) or sd <= 0:
        return 0.0
    return float(np.mean(np.abs(resid) >= float(z) * sd))


@dataclass
class DiagnosticsConfig:
    """
    Configuration for diagnostics and sanity checks.
    """

    outlier_z: float = 3.0
    bound_epsilon: float = 1e-6  # near-bound tolerance as fraction of span


@dataclass
class FitDiagnostics:
    """
    Flat diagnostics object.
    """

    metrics: Dict[str, float]
    flags: Dict[str, Any]
    residuals: Optional[np.ndarray] = None
    yhat: Optional[np.ndarray] = None


def predict_curve(curve: Union[GompertzCurve, BassCurve], t: np.ndarray) -> np.ndarray:
    if isinstance(curve, GompertzCurve):
        return curve.predict(t)
    if isinstance(curve, BassCurve):
        # Diagnostics are normally computed on cumulative space
        return curve.cumulative(t)
    raise TypeError("Unsupported curve type.")


def compute_diagnostics(
    curve: Union[GompertzCurve, BassCurve],
    t: Array1D,
    y: Array1D,
    *,
    cfg: Optional[DiagnosticsConfig] = None,
    param_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    include_residuals: bool = True,
) -> FitDiagnostics:
    """
    Compute fit metrics and flags for a fitted curve.

    Parameters
    ----------
    curve:
        Fitted curve object (GompertzCurve or BassCurve).
    t, y:
        Observed series.
    cfg:
        DiagnosticsConfig.
    param_bounds:
        Optional dict: {param_name: (lo, hi)} used to compute bound flags.
        Example for Gompertz: {"L": (L_lo, L_hi), "t0": (...), "k": (...)}
        Example for Bass: {"p": (...), "q": (...), "m": (...)}
    include_residuals:
        If True, attach residual and yhat arrays.

    Returns
    -------
    FitDiagnostics(metrics=<dict>, flags=<dict>, residuals=<np.ndarray|None>, yhat=<np.ndarray|None>)
    """
    cfg = cfg or DiagnosticsConfig()

    t0 = _as_1d(t)
    y0 = _as_1d(y)
    if len(t0) != len(y0):
        raise ValueError("t and y must have same length.")

    m = _finite_mask(t0, y0)
    t1 = t0[m]
    y1 = y0[m]

    if len(t1) == 0:
        return FitDiagnostics(metrics={"fit_ok": 0.0}, flags={"no_data": True}, residuals=None, yhat=None)

    # sort
    idx = np.argsort(t1)
    t1 = t1[idx]
    y1 = y1[idx]

    yhat = predict_curve(curve, t1)
    resid = y1 - yhat

    y_min = float(np.min(y1))
    y_max = float(np.max(y1))
    y_rng = float(y_max - y_min)
    y_scale = float(np.max(np.abs(y1))) if float(np.max(np.abs(y1))) > 0 else np.nan

    metrics: Dict[str, float] = {
        "mse": float(_mse(y1, yhat)),
        "rmse": float(_rmse(y1, yhat)),
        "mae": float(_mae(y1, yhat)),
        "r2": float(_r2(y1, yhat)),
        "rmse_norm_range": float(_safe_div(_rmse(y1, yhat), y_rng, default=np.nan)),
        "rmse_norm_scale": float(_safe_div(_rmse(y1, yhat), y_scale, default=np.nan)),
        "resid_mean": float(np.mean(resid)),
        "resid_std": float(np.std(resid)),
        "monotonicity_yhat": float(_monotonicity_score(yhat)),
        "n_obs": float(len(t1)),
        "time_span": float(t1[-1] - t1[0]) if len(t1) >= 2 else np.nan,
        "y_min": y_min,
        "y_max": y_max,
        "y_range": y_rng,
    }

    # flags
    flags: Dict[str, Any] = {
        "no_data": False,
        "curve_type": type(curve).__name__,
        "outlier_rate": float(_outlier_rate(resid, z=float(cfg.outlier_z))),
        "has_nan_pred": bool(np.any(~np.isfinite(yhat))),
        "has_nan_resid": bool(np.any(~np.isfinite(resid))),
    }

    # bound flags
    bound_flags: Dict[str, bool] = {}
    if param_bounds is not None:
        eps = float(cfg.bound_epsilon)
        # read params from curve
        if isinstance(curve, GompertzCurve):
            pmap = curve.to_dict()
        else:
            pmap = curve.to_dict()

        for k, (lo, hi) in param_bounds.items():
            lo = float(lo)
            hi = float(hi)
            v = float(pmap.get(k, np.nan))
            if not (np.isfinite(v) and np.isfinite(lo) and np.isfinite(hi) and hi > lo):
                bound_flags[f"{k}_near_bound"] = False
                continue
            span = hi - lo
            near_lo = (v - lo) <= eps * span
            near_hi = (hi - v) <= eps * span
            bound_flags[f"{k}_near_bound"] = bool(near_lo or near_hi)

    if bound_flags:
        flags.update(bound_flags)
        flags["any_param_near_bound"] = bool(any(bound_flags.values()))
    else:
        flags["any_param_near_bound"] = False

    # overall fit_ok heuristic (purely diagnostic; gating is in fit/gates.py)
    fit_ok = True
    if flags["has_nan_pred"] or flags["has_nan_resid"]:
        fit_ok = False
    metrics["fit_ok"] = 1.0 if fit_ok else 0.0

    return FitDiagnostics(
        metrics=metrics,
        flags=flags,
        residuals=resid if include_residuals else None,
        yhat=yhat if include_residuals else None,
    )