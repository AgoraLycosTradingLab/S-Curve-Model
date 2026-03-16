# scurve/fit/gates.py
"""
Fit gating + fallback routing for the S-Curve model.

Purpose
- Decide whether a curve fit is "good enough" to trust for downstream stage/score.
- Provide deterministic pass/fail logic with transparent reasons.
- If failed, route to fallback feature computation.

This module does NOT perform fitting itself. It consumes:
- FitResult from scurve.fit.fitters
- Pre-fit diagnostics/features from scurve.features.pre_fit
- (Optionally) fallback features can be computed by caller when gate fails.

Outputs
- GateResult with:
    - pass_fit (bool)
    - use_fallback (bool)
    - reasons (list[str])
    - quality_score (0..1) [simple heuristic]
    - chosen_model (str)

Philosophy
- Be conservative: only pass when fit quality is acceptable and series looks
  S-curve-like (monotone-ish).
- Never crash: missing fields reduce quality and usually fail.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from scurve.fit.fitters import FitResult


def _get(d: Dict[str, Any], k: str, default: float = np.nan) -> float:
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _isfinite(x: float) -> bool:
    return bool(np.isfinite(x))


def _clip01(x: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(np.clip(x, 0.0, 1.0))


@dataclass
class GateConfig:
    """
    Thresholds for fit acceptance.
    """

    # Minimum observations
    min_points: int = 6

    # Fit quality
    min_r2: float = 0.10
    max_rmse_norm_range: float = 0.50

    # Shape sanity
    min_monotonicity_yhat: float = 0.80

    # Outlier sanity
    max_outlier_rate: float = 0.25  # fraction of residuals beyond z*std

    # Bound sanity
    fail_if_any_param_near_bound: bool = False  # if True, treat near-bound as fail
    allow_any_param_near_bound: bool = True     # if False, treat near-bound as fail

    # Simple quality score weighting (0..1)
    w_r2: float = 0.40
    w_rmse: float = 0.40
    w_mono: float = 0.20


@dataclass
class GateResult:
    pass_fit: bool
    use_fallback: bool
    chosen_model: str
    quality_score: float
    reasons: List[str]
    diagnostics: Dict[str, Any]


class FitGate:
    """
    Deterministic gate for FitResult.
    """

    def __init__(self, config: Optional[GateConfig] = None):
        self.config = config or GateConfig()

    def evaluate(
        self,
        fit: FitResult,
        *,
        pre_features: Optional[Dict[str, Any]] = None,
    ) -> GateResult:
        c = self.config
        reasons: List[str] = []

        if fit is None or not isinstance(fit, FitResult):
            return GateResult(
                pass_fit=False,
                use_fallback=True,
                chosen_model="none",
                quality_score=0.0,
                reasons=["no_fit_result"],
                diagnostics={},
            )

        if not fit.ok or fit.diagnostics is None:
            reasons.append(f"fit_failed:{fit.reason or 'unknown'}")
            return GateResult(
                pass_fit=False,
                use_fallback=True,
                chosen_model=fit.model,
                quality_score=0.0,
                reasons=reasons,
                diagnostics={"fit_reason": fit.reason, "model": fit.model},
            )

        m = fit.diagnostics.metrics
        f = fit.diagnostics.flags

        n_obs = _get(m, "n_obs", np.nan)
        r2 = _get(m, "r2", np.nan)
        rmse_nr = _get(m, "rmse_norm_range", np.nan)
        mono_yhat = _get(m, "monotonicity_yhat", np.nan)
        outlier_rate = _get(f, "outlier_rate", np.nan)
        any_near_bound = bool(f.get("any_param_near_bound", False))
        has_nan_pred = bool(f.get("has_nan_pred", False))
        has_nan_resid = bool(f.get("has_nan_resid", False))

        # Basic checks
        if _isfinite(n_obs) and int(round(n_obs)) < int(c.min_points):
            reasons.append("too_few_points")

        if has_nan_pred or has_nan_resid:
            reasons.append("nan_in_fit")

        # Fit quality checks
        if _isfinite(r2) and r2 < float(c.min_r2):
            reasons.append("low_r2")
        if _isfinite(rmse_nr) and rmse_nr > float(c.max_rmse_norm_range):
            reasons.append("high_rmse_norm_range")
        if not _isfinite(r2) and not _isfinite(rmse_nr):
            reasons.append("missing_fit_quality")

        # Shape checks
        if _isfinite(mono_yhat) and mono_yhat < float(c.min_monotonicity_yhat):
            reasons.append("low_monotonicity_yhat")
        if not _isfinite(mono_yhat):
            reasons.append("missing_monotonicity_yhat")

        # Outliers
        if _isfinite(outlier_rate) and outlier_rate > float(c.max_outlier_rate):
            reasons.append("high_outlier_rate")

        # Bounds
        if any_near_bound:
            if c.fail_if_any_param_near_bound or (not c.allow_any_param_near_bound):
                reasons.append("param_near_bound")

        # Optional pre-fit sanity (if provided)
        if pre_features is not None:
            pre_ok = _get(pre_features, "pre_ok", np.nan)
            if _isfinite(pre_ok) and pre_ok < 0.5:
                reasons.append("prefit_not_ok")

        # Quality score (0..1): higher is better
        # r2 component
        r2_c = np.nan
        if _isfinite(r2):
            r2_c = _clip01(np.clip(r2, 0.0, 1.0))
        # rmse component: invert and clip
        rmse_c = np.nan
        if _isfinite(rmse_nr):
            rmse_c = _clip01(1.0 - np.clip(rmse_nr, 0.0, 1.0))
        # monotonicity component
        mono_c = np.nan
        if _isfinite(mono_yhat):
            mono_c = _clip01(mono_yhat)  # already 0..1

        comps = []
        weights = []
        if _isfinite(r2_c):
            comps.append(float(r2_c)); weights.append(float(c.w_r2))
        if _isfinite(rmse_c):
            comps.append(float(rmse_c)); weights.append(float(c.w_rmse))
        if _isfinite(mono_c):
            comps.append(float(mono_c)); weights.append(float(c.w_mono))

        if len(comps) == 0:
            quality = 0.0
        else:
            wsum = float(np.sum(weights))
            if wsum <= 0:
                wsum = float(len(weights))
                weights = [1.0] * len(weights)
            quality = float(np.dot(comps, np.array(weights) / wsum))
            quality = float(np.clip(quality, 0.0, 1.0))

        pass_fit = len(reasons) == 0
        use_fallback = not pass_fit

        diag = {
            "model": fit.model,
            "n_obs": n_obs,
            "r2": r2,
            "rmse_norm_range": rmse_nr,
            "monotonicity_yhat": mono_yhat,
            "outlier_rate": outlier_rate,
            "any_param_near_bound": any_near_bound,
            "has_nan_pred": has_nan_pred,
            "has_nan_resid": has_nan_resid,
            "quality_components": {"r2": r2_c, "rmse": rmse_c, "mono": mono_c},
        }

        return GateResult(
            pass_fit=bool(pass_fit),
            use_fallback=bool(use_fallback),
            chosen_model=fit.model,
            quality_score=float(quality),
            reasons=reasons,
            diagnostics=diag,
        )