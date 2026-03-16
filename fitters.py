# scurve/fit/fitters.py
"""
Curve fitters for the S-Curve model.

Purpose
- Provide a single, deterministic interface to fit supported curve families.
- Use dependency-light, derivative-free fitters (grid + refinement) implemented
  inside model modules:
    - Gompertz: scurve.models.gompertz.GompertzFitter
    - Bass:     scurve.models.bass.BassFitter
- (Optional) Logistic is a classifier and not an S-curve fit; not used here.

This module also:
- runs diagnostics (scurve.fit.diagnostics)
- returns standardized FitResult objects used by gates/pipeline

Design goals
- Deterministic behavior with stable defaults
- Graceful failure: never crash the pipeline; return status + reason

NOTE
- If you later add SciPy TRF/least_squares, this is the place to plug it in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from scurve.models.gompertz import GompertzCurve, GompertzFitter, GompertzFitConfig
from scurve.models.bass import BassCurve, BassFitter, BassFitConfig
from scurve.fit.diagnostics import DiagnosticsConfig, FitDiagnostics, compute_diagnostics

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


@dataclass
class FitResult:
    """
    Standardized fit result for one curve family.
    """

    model: str  # "gompertz" or "bass"
    ok: bool
    curve: Optional[Union[GompertzCurve, BassCurve]]
    diagnostics: Optional[FitDiagnostics]
    params: Optional[Dict[str, Any]]
    fit_info: Optional[Dict[str, Any]]
    reason: Optional[str] = None


@dataclass
class FittersConfig:
    """
    High-level configuration for fitters.
    """

    # which families to attempt, in order
    try_gompertz: bool = True
    try_bass: bool = True

    # gompertz config
    gompertz: GompertzFitConfig = field(default_factory=GompertzFitConfig)

    # bass config
    bass: BassFitConfig = field(default_factory=BassFitConfig)

    # diagnostics config
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


def fit_gompertz(
    t: Array1D,
    y: Array1D,
    *,
    cfg: Optional[GompertzFitConfig] = None,
    diag_cfg: Optional[DiagnosticsConfig] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    weights: Optional[Array1D] = None,
) -> FitResult:
    """
    Fit Gompertz curve to (t,y).
    """
    t0 = _as_1d(t)
    y0 = _as_1d(y)
    m = _finite_mask(t0, y0)
    t1 = t0[m]
    y1 = y0[m]

    if len(t1) < 5:
        return FitResult(model="gompertz", ok=False, curve=None, diagnostics=None, params=None, fit_info=None, reason="too_few_points")

    fitter = GompertzFitter(cfg or GompertzFitConfig())

    try:
        curve, info = fitter.fit(
            t1,
            y1,
            L_bounds=bounds.get("L") if bounds else None,
            t0_bounds=bounds.get("t0") if bounds else None,
            k_bounds=bounds.get("k") if bounds else None,
            weights=weights,
        )
        diag = compute_diagnostics(curve, t1, y1, cfg=diag_cfg or DiagnosticsConfig(), param_bounds=bounds)
        return FitResult(model="gompertz", ok=True, curve=curve, diagnostics=diag, params=curve.to_dict(), fit_info=info)
    except Exception as e:
        return FitResult(model="gompertz", ok=False, curve=None, diagnostics=None, params=None, fit_info=None, reason=f"exception:{type(e).__name__}")


def fit_bass(
    t: Array1D,
    y: Array1D,
    *,
    cfg: Optional[BassFitConfig] = None,
    diag_cfg: Optional[DiagnosticsConfig] = None,
    bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    weights: Optional[Array1D] = None,
    fit_to: Optional[str] = None,
) -> FitResult:
    """
    Fit Bass diffusion model to (t,y).

    By default, fits to cumulative series.
    If you want to fit to rate series, pass fit_to="rate" and y should be rate.
    """
    t0 = _as_1d(t)
    y0 = _as_1d(y)
    m = _finite_mask(t0, y0)
    t1 = t0[m]
    y1 = y0[m]

    if len(t1) < 6:
        return FitResult(model="bass", ok=False, curve=None, diagnostics=None, params=None, fit_info=None, reason="too_few_points")

    fitter = BassFitter(cfg or BassFitConfig())

    try:
        curve, info = fitter.fit(
            t1,
            y1,
            fit_to=(fit_to or (cfg.fit_to if cfg is not None else "cumulative")),
            p_bounds=bounds.get("p") if bounds else None,
            q_bounds=bounds.get("q") if bounds else None,
            m_bounds=bounds.get("m") if bounds else None,
            weights=weights,
        )
        # diagnostics computed on cumulative
        diag = compute_diagnostics(
            curve,
            t1,
            y1 if (fit_to or (cfg.fit_to if cfg is not None else "cumulative")) == "cumulative" else curve.cumulative(t1),
            cfg=diag_cfg or DiagnosticsConfig(),
            param_bounds=bounds,
        )
        return FitResult(model="bass", ok=True, curve=curve, diagnostics=diag, params=curve.to_dict(), fit_info=info)
    except Exception as e:
        return FitResult(model="bass", ok=False, curve=None, diagnostics=None, params=None, fit_info=None, reason=f"exception:{type(e).__name__}")


def fit_best_curve(
    t: Array1D,
    y: Array1D,
    *,
    cfg: Optional[FittersConfig] = None,
    weights: Optional[Array1D] = None,
    bounds_gompertz: Optional[Dict[str, Tuple[float, float]]] = None,
    bounds_bass: Optional[Dict[str, Tuple[float, float]]] = None,
) -> FitResult:
    """
    Try curve families (Gompertz then Bass by default) and select the best
    by RMSE normalized by range (lower is better).

    Returns a FitResult for the best curve, or ok=False if none succeeded.
    """
    cfg = cfg or FittersConfig()

    results: list[FitResult] = []

    if cfg.try_gompertz:
        results.append(
            fit_gompertz(
                t,
                y,
                cfg=cfg.gompertz,
                diag_cfg=cfg.diagnostics,
                bounds=bounds_gompertz,
                weights=weights,
            )
        )

    if cfg.try_bass:
        results.append(
            fit_bass(
                t,
                y,
                cfg=cfg.bass,
                diag_cfg=cfg.diagnostics,
                bounds=bounds_bass,
                weights=weights,
                fit_to=cfg.bass.fit_to,
            )
        )

    ok_results = [r for r in results if r.ok and r.diagnostics is not None]
    if not ok_results:
        # Choose "best" failure reason to propagate
        reason = "no_fit"
        for r in results:
            if r.reason is not None:
                reason = r.reason
                break
        return FitResult(model="none", ok=False, curve=None, diagnostics=None, params=None, fit_info=None, reason=reason)

    # Choose min rmse_norm_range
    best = None
    best_val = float("inf")
    for r in ok_results:
        val = r.diagnostics.metrics.get("rmse_norm_range", np.nan)
        if np.isfinite(val) and val < best_val:
            best_val = float(val)
            best = r

    # If rmse_norm_range missing for all, fall back to rmse
    if best is None:
        best = min(ok_results, key=lambda rr: rr.diagnostics.metrics.get("rmse", float("inf")))

    return best