# scurve/features/pre_fit.py
"""
Pre-fit feature builders for S-Curve model.

Purpose
- Produce deterministic, cheap, model-agnostic features BEFORE any curve fitting.
- Provide basic diagnostics that help decide whether to fit (or fallback).
- These features are safe on sparse/noisy inputs and do not require scipy/sklearn.

Typical usage
- Build pre-fit features from (t, y)
- If the series looks fit-able (enough points, positive range, not crazy noisy),
  run curve fit, then use post_fit.py adapters
- If fit fails, use fallback.py

Feature families
- data quality: n_obs, missing, time span, dt regularity
- level/scale: start/end/min/max/range
- trend: linear slope, r2, sign stability
- volatility/noise: residual std (vs linear), diff volatility, CV
- shape: curvature proxy, monotonicity, crossings
- timing: times to reach percentiles of range (10/25/50/75/90)

All features are returned as a flat dict. Diagnostics returned separately.
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


def _finite_mask(*arrs: np.ndarray) -> np.ndarray:
    m = np.ones_like(arrs[0], dtype=bool)
    for a in arrs:
        m &= np.isfinite(a)
    return m


def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b == 0:
        return float(default)
    return float(a / b)


def _linear_fit(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, float, float]:
    """
    OLS y = a + b t.
    Returns (a, b, r2, resid_std).
    """
    if len(t) < 2:
        return (np.nan, np.nan, np.nan, np.nan)

    t0 = t - np.mean(t)
    y0 = y - np.mean(y)
    denom = np.sum(t0 * t0)
    if denom <= 0:
        a = float(np.mean(y))
        b = 0.0
        resid = y - a
        return (a, b, 0.0, float(np.std(resid)))

    b = np.sum(t0 * y0) / denom
    a = float(np.mean(y) - b * np.mean(t))

    yhat = a + b * t
    resid = y - yhat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return (float(a), float(b), float(r2), float(np.std(resid)))


def _curvature_proxy(y: np.ndarray) -> float:
    if len(y) < 3:
        return np.nan
    d2 = np.diff(y, n=2)
    rng = float(np.max(y) - np.min(y))
    if rng <= 0:
        return 0.0
    return float(np.mean(np.abs(d2)) / rng)


def _monotonicity_score(y: np.ndarray) -> float:
    if len(y) < 2:
        return np.nan
    dy = np.diff(y)
    return float(np.mean(dy >= 0))


def _sign_consistency(y: np.ndarray) -> float:
    if len(y) < 3:
        return np.nan
    dy = np.diff(y)
    pos = np.sum(dy > 0)
    neg = np.sum(dy < 0)
    dom = max(pos, neg)
    return float(dom / max(1, len(dy)))


def _percentile_time(t: np.ndarray, y: np.ndarray, pct: float) -> float:
    if len(t) < 2:
        return np.nan
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    rng = y_max - y_min
    if rng <= 0:
        return np.nan
    target = y_min + pct * rng
    idx = np.where(y >= target)[0]
    if idx.size == 0:
        return np.nan
    i = int(idx[0])
    if i == 0:
        return float(t[0])

    y0, y1 = float(y[i - 1]), float(y[i])
    t0, t1 = float(t[i - 1]), float(t[i])
    if y1 == y0:
        return float(t1)
    w = (target - y0) / (y1 - y0)
    return float(t0 + w * (t1 - t0))


def _dt_stats(t: np.ndarray) -> Dict[str, float]:
    if len(t) < 2:
        return {"dt_mean": np.nan, "dt_std": np.nan, "dt_cv": np.nan, "dt_min": np.nan, "dt_max": np.nan}
    dt = np.diff(t)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        return {"dt_mean": np.nan, "dt_std": np.nan, "dt_cv": np.nan, "dt_min": np.nan, "dt_max": np.nan}
    mu = float(np.mean(dt))
    sd = float(np.std(dt))
    cv = _safe_div(sd, mu, default=np.nan)
    return {"dt_mean": mu, "dt_std": sd, "dt_cv": float(cv), "dt_min": float(np.min(dt)), "dt_max": float(np.max(dt))}


@dataclass
class PreFitConfig:
    min_points: int = 6
    percentiles: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    eps: float = 1e-12


@dataclass
class PreFitResult:
    features: Dict[str, float]
    diagnostics: Dict[str, Any]


def build_pre_fit_features(
    t: Array1D,
    y: Array1D,
    *,
    config: Optional[PreFitConfig] = None,
    prefix: str = "pre_",
) -> PreFitResult:
    """
    Build pre-fit features from (t, y).

    Returns:
      PreFitResult(features=<flat dict>, diagnostics=<flat dict>)
    """
    cfg = config or PreFitConfig()

    t0 = _as_1d(t)
    y0 = _as_1d(y)
    if len(t0) != len(y0):
        raise ValueError("t and y must have same length.")

    n_raw = int(len(t0))
    m = _finite_mask(t0, y0)
    t1 = t0[m]
    y1 = y0[m]
    n_used = int(len(t1))

    diag: Dict[str, Any] = {
        "n_raw": n_raw,
        "n_used": n_used,
        "dropped_na": int(n_raw - n_used),
        "too_few_points": n_used < int(cfg.min_points),
    }

    if n_used == 0:
        feats = {f"{prefix}ok": 0.0, f"{prefix}reason_code": 1.0}
        return PreFitResult(features=feats, diagnostics=diag)

    # Sort
    idx = np.argsort(t1)
    t1 = t1[idx]
    y1 = y1[idx]

    # Basic stats
    y_min = float(np.min(y1))
    y_max = float(np.max(y1))
    y_rng = float(y_max - y_min)
    y_mean = float(np.mean(y1))
    y_std = float(np.std(y1))
    y_cv = float(_safe_div(y_std, abs(y_mean), default=np.nan))

    # Trend
    a, b, r2, resid_std = _linear_fit(t1, y1)

    # Diff noise
    dy = np.diff(y1) if len(y1) >= 2 else np.array([], dtype=float)
    dy_std = float(np.std(dy)) if dy.size > 0 else np.nan
    dy_mean = float(np.mean(dy)) if dy.size > 0 else np.nan

    # Timing
    pct_times: Dict[str, float] = {}
    if y_rng > cfg.eps:
        for pct in cfg.percentiles:
            pct_times[f"{prefix}t_p{int(round(pct*100)):02d}"] = _percentile_time(t1, y1, float(pct))
    else:
        for pct in cfg.percentiles:
            pct_times[f"{prefix}t_p{int(round(pct*100)):02d}"] = np.nan

    # dt regularity
    dts = _dt_stats(t1)

    # shape proxies
    mono = _monotonicity_score(y1)
    signc = _sign_consistency(y1)
    curv = _curvature_proxy(y1)

    # crossings of mid-range
    mid = y_min + 0.5 * y_rng
    crosses = np.nan
    if y_rng > cfg.eps and len(y1) >= 2:
        s = np.sign(y1 - mid)
        crosses = float(np.sum(np.diff(s) != 0))

    diag["constant_series"] = bool(y_rng <= cfg.eps)
    diag["time_span"] = float(t1[-1] - t1[0]) if len(t1) >= 2 else np.nan

    # Fit-ability heuristic (simple)
    fit_ok = 1.0
    reason_code = 0.0
    if diag["too_few_points"]:
        fit_ok = 0.0
        reason_code = 2.0
    elif diag["constant_series"]:
        fit_ok = 0.0
        reason_code = 3.0
    elif not np.isfinite(diag["time_span"]) or diag["time_span"] <= 0:
        fit_ok = 0.0
        reason_code = 4.0

    feats: Dict[str, float] = {
        f"{prefix}ok": float(fit_ok),
        f"{prefix}reason_code": float(reason_code),
        f"{prefix}n_obs": float(n_used),
        f"{prefix}y_start": float(y1[0]),
        f"{prefix}y_end": float(y1[-1]),
        f"{prefix}y_min": float(y_min),
        f"{prefix}y_max": float(y_max),
        f"{prefix}y_range": float(y_rng),
        f"{prefix}y_mean": float(y_mean),
        f"{prefix}y_std": float(y_std),
        f"{prefix}y_cv": float(y_cv),
        f"{prefix}slope": float(b),
        f"{prefix}intercept": float(a),
        f"{prefix}r2": float(r2),
        f"{prefix}lin_resid_std": float(resid_std),
        f"{prefix}dy_mean": float(dy_mean),
        f"{prefix}dy_std": float(dy_std),
        f"{prefix}curvature": float(curv),
        f"{prefix}monotonicity": float(mono),
        f"{prefix}sign_consistency": float(signc),
        f"{prefix}mid_crossings": float(crosses),
        f"{prefix}time_span": float(diag["time_span"]),
        f"{prefix}dt_mean": float(dts["dt_mean"]),
        f"{prefix}dt_std": float(dts["dt_std"]),
        f"{prefix}dt_cv": float(dts["dt_cv"]),
        f"{prefix}dt_min": float(dts["dt_min"]),
        f"{prefix}dt_max": float(dts["dt_max"]),
    }
    feats.update({k: float(v) for k, v in pct_times.items()})

    return PreFitResult(features=feats, diagnostics=diag)