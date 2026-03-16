# scurve/features/post_fit.py
"""
Post-fit feature builders for S-Curve models.

Purpose
- Convert fitted curve objects (Gompertz / Bass) into a consistent flat feature dict.
- Provide deterministic derived features (inflection, slope, timing, fit quality).
- Provide a simple "fit evaluator" (yhat, residual stats, R^2, logloss-style where relevant).

This module deliberately avoids depending on sklearn/scipy.
It expects the curve objects from:
- scurve.models.gompertz.GompertzCurve
- scurve.models.bass.BassCurve

If you need other curve families, add a new adapter here.
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
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return float(default)
    return float(a / b)


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    if ss_tot <= 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def _mse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean((y - yhat) ** 2))


def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(_mse(y, yhat)))


def _resid_stats(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    r = y - yhat
    return {
        "resid_mean": float(np.mean(r)),
        "resid_std": float(np.std(r)),
        "resid_median": float(np.median(r)),
        "resid_q25": float(np.quantile(r, 0.25)),
        "resid_q75": float(np.quantile(r, 0.75)),
        "resid_skew_proxy": float(_safe_div(float(np.mean((r - np.mean(r)) ** 3)), float(np.std(r) ** 3), default=np.nan)),
    }


def _peak_of_series(t: np.ndarray, s: np.ndarray) -> Tuple[float, float]:
    if len(t) == 0:
        return (np.nan, np.nan)
    i = int(np.nanargmax(s))
    return (float(t[i]), float(s[i]))


@dataclass
class PostFitResult:
    features: Dict[str, float]
    diagnostics: Dict[str, Any]


def build_post_fit_features(
    curve: Union[GompertzCurve, BassCurve],
    t: Array1D,
    y: Array1D,
    *,
    prefix: str = "",
    include_residuals: bool = True,
) -> PostFitResult:
    """
    Build a flat feature dict from a fitted curve and observed (t,y).

    Parameters
    ----------
    curve:
        GompertzCurve or BassCurve.
    t, y:
        Observed series used for evaluation.
    prefix:
        Optional prefix applied to feature keys (e.g. "g_", "b_").
    include_residuals:
        If True, include residual statistics.

    Returns
    -------
    PostFitResult(features, diagnostics)
    """
    t0 = _as_1d(t)
    y0 = _as_1d(y)
    if len(t0) != len(y0):
        raise ValueError("t and y must have same length.")

    m = _finite_mask(t0, y0)
    t1 = t0[m]
    y1 = y0[m]

    diag: Dict[str, Any] = {
        "n_raw": int(len(t0)),
        "n_used": int(len(t1)),
        "dropped_na": int(len(t0) - len(t1)),
        "curve_type": type(curve).__name__,
    }

    if len(t1) == 0:
        return PostFitResult(features={f"{prefix}fit_ok": 0.0}, diagnostics=diag)

    # sort by t
    idx = np.argsort(t1)
    t1 = t1[idx]
    y1 = y1[idx]

    # prediction hooks
    if isinstance(curve, GompertzCurve):
        yhat = curve.predict(t1)
        dy = curve.derivative(t1)
        peak_t, peak_rate = _peak_of_series(t1, dy)
        inf_t = float(curve.params.t0)
        inf_y = float(curve.inflection_value())
        params = curve.to_dict()

        feats: Dict[str, float] = {
            f"{prefix}fit_ok": 1.0,
            f"{prefix}L": float(params["L"]),
            f"{prefix}t0": float(params["t0"]),
            f"{prefix}k": float(params["k"]),
            f"{prefix}inflection_t": inf_t,
            f"{prefix}inflection_y": inf_y,
            f"{prefix}peak_slope_t": float(peak_t),
            f"{prefix}peak_slope": float(peak_rate),
            f"{prefix}end_yhat": float(yhat[-1]),
            f"{prefix}start_yhat": float(yhat[0]),
        }

    elif isinstance(curve, BassCurve):
        yhat = curve.cumulative(t1)
        rate = curve.rate(t1)
        peak_t, peak_rate = _peak_of_series(t1, rate)
        params = curve.to_dict()

        # Bass "inflection" is often discussed via peak time of rate
        feats = {
            f"{prefix}fit_ok": 1.0,
            f"{prefix}p": float(params["p"]),
            f"{prefix}q": float(params["q"]),
            f"{prefix}m": float(params["m"]),
            f"{prefix}peak_rate_t": float(peak_t),
            f"{prefix}peak_rate": float(peak_rate),
            f"{prefix}peak_rate_theory_t": float(curve.peak_time()),
            f"{prefix}end_yhat": float(yhat[-1]),
            f"{prefix}start_yhat": float(yhat[0]),
        }
    else:
        raise TypeError("Unsupported curve type for post-fit features.")

    # Fit-quality features
    feats.update(
        {
            f"{prefix}mse": float(_mse(y1, yhat)),
            f"{prefix}rmse": float(_rmse(y1, yhat)),
            f"{prefix}mae": float(_mae(y1, yhat)),
            f"{prefix}r2": float(_r2(y1, yhat)),
        }
    )

    # Normalized error features
    y_rng = float(np.max(y1) - np.min(y1))
    y_scale = float(np.max(np.abs(y1))) if float(np.max(np.abs(y1))) > 0 else np.nan
    feats[f"{prefix}rmse_norm_range"] = float(_safe_div(feats[f"{prefix}rmse"], y_rng, default=np.nan))
    feats[f"{prefix}rmse_norm_scale"] = float(_safe_div(feats[f"{prefix}rmse"], y_scale, default=np.nan))

    # Residual stats
    if include_residuals:
        rs = _resid_stats(y1, yhat)
        for k, v in rs.items():
            feats[f"{prefix}{k}"] = float(v)

    # Generic timing features
    T = float(t1[-1] - t1[0])
    feats[f"{prefix}time_span"] = float(T)
    feats[f"{prefix}n_obs"] = float(len(t1))

    # Where is the series currently relative to the cap?
    end = float(y1[-1])
    start = float(y1[0])
    feats[f"{prefix}y_end"] = end
    feats[f"{prefix}y_start"] = start
    feats[f"{prefix}y_end_minus_yhat_end"] = float(end - float(yhat[-1]))
    feats[f"{prefix}y_end_over_yhat_end"] = float(_safe_div(end, float(yhat[-1]), default=np.nan))

    # Optional: "progress" fraction based on fitted cap (L or m)
    if isinstance(curve, GompertzCurve):
        cap = float(curve.params.L)
        feats[f"{prefix}progress_end"] = float(_safe_div(end, cap, default=np.nan))
        feats[f"{prefix}progress_yhat_end"] = float(_safe_div(float(yhat[-1]), cap, default=np.nan))
    elif isinstance(curve, BassCurve):
        cap = float(curve.params.m)
        feats[f"{prefix}progress_end"] = float(_safe_div(end, cap, default=np.nan))
        feats[f"{prefix}progress_yhat_end"] = float(_safe_div(float(yhat[-1]), cap, default=np.nan))

    return PostFitResult(features=feats, diagnostics=diag)