# scurve/features/fallback.py
"""
Fallback feature builders for S-Curve model.

Purpose
- Provide deterministic, robust features when curve fitting fails (or is skipped).
- Avoid hard failures caused by sparse/noisy data, NaNs, constant series, etc.
- Return a consistent feature dict + diagnostics.

These features are intentionally simple and numerically stable:
- level / scale
- linear slope + R^2
- curvature proxy (second-diff)
- drawdown / recovery proxies
- monotonicity + sign consistency
- normalized AUC (area under curve) proxy
- percentile-times (t at which y crosses x% of range), when possible

No dependencies beyond numpy/pandas.
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


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(np.clip(x, 0.0, 1.0))


def _nan_if_bad(x: float) -> float:
    return float(x) if np.isfinite(x) else float("nan")


def _linear_fit(t: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    """
    Simple OLS y = a + b t.
    Returns (intercept a, slope b, r2).
    """
    if len(t) < 2:
        return (np.nan, np.nan, np.nan)

    t0 = t - np.mean(t)
    y0 = y - np.mean(y)
    denom = np.sum(t0 * t0)
    if denom <= 0:
        return (float(np.mean(y)), 0.0, 0.0)

    b = np.sum(t0 * y0) / denom
    a = float(np.mean(y) - b * np.mean(t))

    yhat = a + b * t
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return (float(a), float(b), float(r2))


def _percentile_time(t: np.ndarray, y: np.ndarray, pct: float) -> float:
    """
    Time when y crosses y_min + pct*(y_max - y_min).
    Returns NaN if not bracketed.
    Uses linear interpolation between first crossing pair.
    """
    if len(t) < 2:
        return np.nan
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    rng = y_max - y_min
    if rng <= 0:
        return np.nan

    target = y_min + pct * rng
    # Find first index where y >= target (assuming increasing-ish; still works as "first crossing")
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


def _auc_norm(t: np.ndarray, y: np.ndarray) -> float:
    """
    Normalized area under curve:
      AUC / (T * y_max)  (if y_max>0)
    """
    if len(t) < 2:
        return np.nan
    y_max = float(np.max(y))
    T = float(t[-1] - t[0])
    if y_max <= 0 or T <= 0:
        return np.nan
    trapz_fn = getattr(np, "trapezoid", None)
    if trapz_fn is None:
        trapz_fn = getattr(np, "trapz")
    auc = float(trapz_fn(y, t))
    return _safe_div(auc, T * y_max, default=np.nan)


def _monotonicity_score(y: np.ndarray) -> float:
    """
    Fraction of consecutive differences that are non-negative.
    (1.0 means monotone non-decreasing.)
    """
    if len(y) < 2:
        return np.nan
    dy = np.diff(y)
    return float(np.mean(dy >= 0))


def _sign_consistency(y: np.ndarray) -> float:
    """
    Fraction of consecutive differences that have the dominant sign.
    Useful when series oscillates: returns near 0.5.
    """
    if len(y) < 3:
        return np.nan
    dy = np.diff(y)
    pos = np.sum(dy > 0)
    neg = np.sum(dy < 0)
    dom = max(pos, neg)
    return float(dom / max(1, len(dy)))


def _curvature_proxy(y: np.ndarray) -> float:
    """
    Mean absolute second difference scaled by range.
    """
    if len(y) < 3:
        return np.nan
    d2 = np.diff(y, n=2)
    rng = float(np.max(y) - np.min(y))
    if rng <= 0:
        return 0.0
    return float(np.mean(np.abs(d2)) / rng)


def _drawdown_stats(y: np.ndarray) -> Tuple[float, float]:
    """
    Returns (max_drawdown, dd_end) on the raw series.
    drawdown defined relative to running max: (y - peak)/peak.
    """
    if len(y) < 2:
        return (np.nan, np.nan)
    peak = np.maximum.accumulate(y)
    # avoid divide by zero
    denom = np.where(peak == 0, np.nan, peak)
    dd = (y - peak) / denom
    finite_dd = dd[np.isfinite(dd)]
    if finite_dd.size == 0:
        return (np.nan, np.nan)
    max_dd = float(np.min(finite_dd))
    dd_end = float(dd[-1]) if np.isfinite(dd[-1]) else np.nan
    return (max_dd, dd_end)


@dataclass
class FallbackConfig:
    """
    Settings controlling fallback feature behavior.
    """

    min_points: int = 6
    percentiles: Tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    eps: float = 1e-12


@dataclass
class FallbackResult:
    features: Dict[str, float]
    diagnostics: Dict[str, Any]


def build_fallback_features(
    t: Array1D,
    y: Array1D,
    *,
    config: Optional[FallbackConfig] = None,
    prefix: str = "fb_",
) -> FallbackResult:
    """
    Build deterministic fallback features from (t, y).

    Returns:
      FallbackResult(features=<flat dict>, diagnostics=<flat dict>)

    Diagnostics includes:
      - n_raw, n_used
      - dropped_na
      - reason flags (too_few_points, constant_series, nonpositive_range, etc.)
    """
    cfg = config or FallbackConfig()

    t0 = _as_1d(t)
    y0 = _as_1d(y)
    n_raw = int(len(t0))

    if n_raw != int(len(y0)):
        raise ValueError("t and y must have same length.")

    m = _finite_mask(t0, y0)
    t1 = t0[m]
    y1 = y0[m]
    n_used = int(len(t1))
    dropped = int(n_raw - n_used)

    diag: Dict[str, Any] = {
        "n_raw": n_raw,
        "n_used": n_used,
        "dropped_na": dropped,
        "too_few_points": n_used < int(cfg.min_points),
    }

    # If too few points, return mostly NaNs but still include basic last/first when possible.
    if n_used == 0:
        feats = {
            f"{prefix}ok": 0.0,
            f"{prefix}reason_code": 1.0,  # 1: no usable data
        }
        return FallbackResult(features=feats, diagnostics=diag)

    # Sort by t for determinism
    idx = np.argsort(t1)
    t1 = t1[idx]
    y1 = y1[idx]

    y_min = float(np.min(y1))
    y_max = float(np.max(y1))
    y_last = float(y1[-1])
    y_first = float(y1[0])
    rng = y_max - y_min
    diag["constant_series"] = bool(rng <= cfg.eps)
    diag["range"] = float(rng)

    # core stats
    a, b, r2 = _linear_fit(t1, y1)
    max_dd, dd_end = _drawdown_stats(y1)
    growth_total = _safe_div(y_last - y_first, abs(y_first), default=np.nan)
    growth_per_step = _safe_div(y_last - y_first, max(1, n_used - 1), default=np.nan)
    growth_accel = np.nan
    if n_used >= 3:
        last_step = float(y1[-1] - y1[-2])
        prev_step = float(y1[-2] - y1[-3])
        growth_accel = float(last_step - prev_step)
    growth_strength_01 = np.nan
    if np.isfinite(growth_total):
        growth_strength_01 = _clip01(0.5 + 0.5 * np.tanh(2.0 * growth_total))

    # time normalization
    T = float(t1[-1] - t1[0])
    if T <= 0:
        T = np.nan
    diag["time_span"] = _nan_if_bad(T)

    # percentiles times
    pct_times: Dict[str, float] = {}
    if rng > cfg.eps:
        for pct in cfg.percentiles:
            pct_times[f"{prefix}t_p{int(round(pct*100)):02d}"] = _percentile_time(t1, y1, float(pct))
    else:
        for pct in cfg.percentiles:
            pct_times[f"{prefix}t_p{int(round(pct*100)):02d}"] = np.nan

    # build features
    feats: Dict[str, float] = {
        f"{prefix}ok": 0.0 if diag["too_few_points"] else 1.0,
        f"{prefix}reason_code": 2.0 if diag["too_few_points"] else (3.0 if diag["constant_series"] else 0.0),
        f"{prefix}y_first": float(y_first),
        f"{prefix}y_last": float(y_last),
        f"{prefix}y_min": float(y_min),
        f"{prefix}y_max": float(y_max),
        f"{prefix}range": float(rng),
        f"{prefix}slope": float(b),
        f"{prefix}intercept": float(a),
        f"{prefix}r2": float(r2),
        f"{prefix}curvature": float(_curvature_proxy(y1)),
        f"{prefix}auc_norm": float(_auc_norm(t1, y1)),
        f"{prefix}monotonicity": float(_monotonicity_score(y1)),
        f"{prefix}sign_consistency": float(_sign_consistency(y1)),
        f"{prefix}n_used": float(n_used),
        f"{prefix}max_drawdown": float(max_dd),
        f"{prefix}drawdown_end": float(dd_end),
        f"{prefix}level_pos_frac": float(np.mean(y1 > 0)) if n_used > 0 else np.nan,
        f"{prefix}growth_total": float(growth_total),
        f"{prefix}growth_per_step": float(growth_per_step),
        f"{prefix}growth_accel": float(growth_accel),
        f"{prefix}growth_strength_01": float(growth_strength_01),
        f"{prefix}growth_pos_flag": 1.0 if np.isfinite(growth_total) and growth_total > 0 else 0.0,
    }
    feats.update({k: float(v) for k, v in pct_times.items()})

    # If too few points, keep features but mark ok=0
    if diag["too_few_points"]:
        feats[f"{prefix}ok"] = 0.0

    return FallbackResult(features=feats, diagnostics=diag)
