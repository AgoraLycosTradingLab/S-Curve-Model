# scurve/score/stage.py
"""
Stage labeling logic for the S-Curve model.

Purpose
- Convert curve-fit + feature signals into a deterministic "stage" label.
- Provide both:
  1) discrete stage label (string)
  2) stage score components (0..1) that can feed composite scoring

This module is intentionally rule-based and dependency-light.

Expected inputs
- A flat feature dict, typically produced by:
  - scurve.features.pre_fit.build_pre_fit_features(...)
  - scurve.features.post_fit.build_post_fit_features(...)
  - scurve.features.fallback.build_fallback_features(...)
- The feature dict should include either Gompertz-derived keys OR Bass-derived keys.
  You can prefix keys before passing them in (e.g., "g_", "b_"), but the default
  StageScorer below expects the *unprefixed* keys created by post_fit adapters:
    Gompertz: L, t0, k, peak_slope, inflection_t, progress_end, ...
    Bass:     p, q, m, peak_rate, peak_rate_t, progress_end, ...

Design approach
- Use "progress" fraction (end / cap) to infer maturity.
- Use slope (peak slope/peak rate) and recent slope proxy to infer growth phase.
- Use monotonicity and fit_quality (r2 / rmse_norm_range) to downweight.
- Provide deterministic fallback behavior if some keys are missing.

Stages
- "early"      : low penetration, accelerating growth
- "growth"     : mid penetration, high slope
- "mature"     : high penetration, slowing slope
- "decline"    : negative trend or drawdown dominance (proxy)
- "unknown"    : insufficient information

NOTE
- This is a first-pass stage system meant to be stable.
- You can later tune thresholds in config without changing the pipeline API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np


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


def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return float(default)
    return float(a / b)


@dataclass
class StageConfig:
    """
    Threshold configuration for stage labeling.

    progress_* thresholds are fractions of cap (0..1).

    early_max:
        progress_end <= early_max => early candidate (if slope decent)
    growth_max:
        early_max < progress_end <= growth_max => growth candidate
    mature_min:
        progress_end >= mature_min => mature candidate (unless decline)
    """

    early_max: float = 0.25
    growth_max: float = 0.70
    mature_min: float = 0.70

    # Slope normalization:
    # We normalize slope by (cap / time_span) which is a crude "average slope to reach cap".
    # growth_slope_min is relative to that.
    growth_slope_min: float = 0.75
    early_slope_min: float = 0.25
    mature_slope_max: float = 0.60  # below this suggests slowing

    # Decline logic using observed y trend proxies (from pre_/fb_ keys)
    decline_slope_max: float = -1e-12  # negative slope => decline candidate
    decline_drawdown_min: float = -0.25  # max_drawdown <= -25% => decline candidate

    # Fit quality gating
    min_r2: float = 0.10
    max_rmse_norm_range: float = 0.50  # if too large, fit is poor

    # Confidence shaping
    min_monotonicity: float = 0.55  # below suggests noisy / not S-curve-like


@dataclass
class StageResult:
    stage: str
    confidence: float
    components: Dict[str, float]
    diagnostics: Dict[str, Any]


class StageScorer:
    """
    Deterministic rule-based stage scorer.

    Usage:
        scorer = StageScorer()
        res = scorer.score(features)
    """

    def __init__(self, config: Optional[StageConfig] = None):
        self.config = config or StageConfig()

    def score(self, feats: Dict[str, Any]) -> StageResult:
        c = self.config

        # Prefer post-fit keys, fall back to pre_/fb_ keys when needed.
        progress = _get(feats, "progress_end", np.nan)

        # Fit quality (from post_fit.py)
        r2 = _get(feats, "r2", np.nan)
        rmse_nr = _get(feats, "rmse_norm_range", np.nan)

        # Data shape / noise proxies (from pre_fit or fallback)
        mono = _get(feats, "pre_monotonicity", _get(feats, "fb_monotonicity", np.nan))
        pre_slope = _get(feats, "pre_slope", _get(feats, "fb_slope", np.nan))
        fb_growth = _get(feats, "fb_growth_total", np.nan)
        fb_growth_strength = _get(feats, "fb_growth_strength_01", np.nan)
        fb_n_used = _get(feats, "fb_n_used", _get(feats, "pre_n_obs", np.nan))
        max_dd = _get(feats, "fb_max_drawdown", np.nan)  # fallback has it
        if not _isfinite(max_dd):
            # Some pipelines may include this without fb_ prefix.
            max_dd = _get(feats, "max_drawdown", np.nan)

        time_span = _get(feats, "time_span", _get(feats, "pre_time_span", np.nan))
        y_end = _get(feats, "y_end", _get(feats, "pre_y_end", _get(feats, "fb_y_last", np.nan)))

        # Cap estimate (Gompertz: L, Bass: m)
        cap = _get(feats, "L", np.nan)
        if not _isfinite(cap):
            cap = _get(feats, "m", np.nan)

        # Peak slope proxy from post-fit
        peak_slope = _get(feats, "peak_slope", np.nan)  # Gompertz
        if not _isfinite(peak_slope):
            peak_slope = _get(feats, "peak_rate", np.nan)  # Bass
        if not _isfinite(peak_slope):
            peak_slope = _get(feats, "peak_rate", np.nan)

        # Normalize slope by cap/time_span if possible
        avg_slope_to_cap = _safe_div(cap, time_span, default=np.nan)
        slope_rel = _safe_div(peak_slope, avg_slope_to_cap, default=np.nan)

        # If we don't have peak slope, use pre_slope as crude proxy
        if not _isfinite(slope_rel) and _isfinite(pre_slope) and _isfinite(avg_slope_to_cap):
            slope_rel = _safe_div(pre_slope, avg_slope_to_cap, default=np.nan)
        if not _isfinite(slope_rel) and _isfinite(fb_growth_strength):
            slope_rel = float(2.0 * fb_growth_strength)

        # Fit quality gate
        fit_ok = True
        fit_bad_reasons = []
        if _isfinite(r2) and r2 < c.min_r2:
            fit_ok = False
            fit_bad_reasons.append("low_r2")
        if _isfinite(rmse_nr) and rmse_nr > c.max_rmse_norm_range:
            fit_ok = False
            fit_bad_reasons.append("high_rmse_norm_range")

        # If fit is missing entirely, don't fail; just reduce confidence.
        has_fit_metrics = _isfinite(r2) or _isfinite(rmse_nr)

        # Decline detection (uses observed trend proxies)
        decline = False
        if _isfinite(pre_slope) and pre_slope < c.decline_slope_max:
            decline = True
        if _isfinite(max_dd) and max_dd <= c.decline_drawdown_min:
            decline = True

        # Stage candidate logic based on progress + slope
        stage = "unknown"
        if decline:
            stage = "decline"
        else:
            if _isfinite(progress):
                if progress <= c.early_max:
                    # early requires some positive slope
                    if (not _isfinite(slope_rel)) or slope_rel >= c.early_slope_min:
                        stage = "early"
                    else:
                        stage = "unknown"
                elif progress <= c.growth_max:
                    # growth requires stronger slope
                    if (not _isfinite(slope_rel)) or slope_rel >= c.growth_slope_min:
                        stage = "growth"
                    else:
                        # low slope in mid progress looks like "mature-ish"
                        stage = "mature"
                else:
                    # high progress => mature unless slope still very high
                    if _isfinite(slope_rel) and slope_rel >= c.growth_slope_min:
                        stage = "growth"
                    else:
                        stage = "mature"
            else:
                # No progress: infer from slope sign and monotonicity
                if _isfinite(fb_growth) and fb_growth > 0 and _isfinite(fb_n_used) and fb_n_used <= 2:
                    if _isfinite(fb_growth_strength) and fb_growth_strength >= 0.85:
                        stage = "early"
                    elif _isfinite(fb_growth_strength) and fb_growth_strength >= 0.55:
                        stage = "growth"
                    else:
                        stage = "mature"
                elif _isfinite(pre_slope) and pre_slope > 0 and (_isfinite(mono) and mono >= c.min_monotonicity):
                    if _isfinite(fb_n_used) and fb_n_used <= 2:
                        if _isfinite(fb_growth_strength) and fb_growth_strength >= 0.85:
                            stage = "early"
                        elif _isfinite(fb_growth_strength) and fb_growth_strength >= 0.55:
                            stage = "growth"
                        else:
                            stage = "mature"
                    else:
                        stage = "growth"
                elif _isfinite(pre_slope) and pre_slope > 0:
                    stage = "growth"
                elif _isfinite(fb_growth) and fb_growth > 0:
                    if _isfinite(fb_n_used) and fb_n_used <= 2:
                        if _isfinite(fb_growth_strength) and fb_growth_strength >= 0.85:
                            stage = "early"
                        elif _isfinite(fb_growth_strength) and fb_growth_strength >= 0.55:
                            stage = "growth"
                        else:
                            stage = "mature"
                    else:
                        stage = "growth"
                elif _isfinite(pre_slope) and pre_slope < 0:
                    stage = "decline"
                else:
                    stage = "unknown"

        # Confidence score (0..1), conservative
        conf = 0.50

        # boost if we have progress + cap
        if _isfinite(progress) and _isfinite(cap) and cap > 0:
            conf += 0.15

        # boost if monotone-ish
        if _isfinite(mono):
            conf += 0.20 * float(np.clip((mono - 0.5) / 0.5, 0.0, 1.0))  # 0 at 0.5, 1 at 1.0

        # boost for good fit metrics
        if has_fit_metrics and fit_ok:
            conf += 0.15
        elif has_fit_metrics and not fit_ok:
            conf -= 0.20

        # small penalty if key drivers missing
        missing_drivers = 0
        for key in ("progress_end", "L", "m", "peak_slope", "peak_rate", "pre_slope"):
            if not _isfinite(_get(feats, key, np.nan)):
                missing_drivers += 1
        conf -= 0.03 * min(missing_drivers, 5)

        conf = float(np.clip(conf, 0.0, 1.0))

        # Component scores (0..1) to feed composite
        # These are interpretable:
        # - maturity: progress
        # - growth_strength: slope_rel clipped
        # - fit_quality: based on r2 and rmse_norm_range
        maturity = _clip01(progress)
        if not _isfinite(maturity) and _isfinite(fb_growth):
            if fb_growth > 0:
                if _isfinite(fb_n_used) and fb_n_used <= 2 and _isfinite(fb_growth_strength):
                    if fb_growth_strength >= 0.85:
                        maturity = 0.20
                    elif fb_growth_strength >= 0.55:
                        maturity = 0.50
                    else:
                        maturity = 0.80
                else:
                    maturity = 0.40
            else:
                maturity = 0.80

        growth_strength = _clip01(_safe_div(slope_rel, 2.0, default=np.nan))  # slope_rel=2 => 1.0
        if not _isfinite(growth_strength) and _isfinite(fb_growth_strength):
            growth_strength = _clip01(fb_growth_strength)

        fit_quality = np.nan
        if _isfinite(r2) and _isfinite(rmse_nr):
            fit_quality = float(np.clip(0.5 * (np.clip(r2, 0.0, 1.0)) + 0.5 * (1.0 - np.clip(rmse_nr, 0.0, 1.0)), 0.0, 1.0))
        elif _isfinite(r2):
            fit_quality = float(np.clip(r2, 0.0, 1.0))
        elif _isfinite(rmse_nr):
            fit_quality = float(np.clip(1.0 - np.clip(rmse_nr, 0.0, 1.0), 0.0, 1.0))
        elif _isfinite(_get(feats, "fb_r2", np.nan)):
            fit_quality = float(np.clip(_get(feats, "fb_r2", np.nan), 0.0, 1.0))

        # decline risk proxy: based on negative slope and drawdown
        decline_risk = 0.0
        if _isfinite(pre_slope) and pre_slope < 0:
            decline_risk += 0.5
        if _isfinite(max_dd) and max_dd <= c.decline_drawdown_min:
            decline_risk += 0.5
        decline_risk = float(np.clip(decline_risk, 0.0, 1.0))

        components = {
            "maturity": float(maturity) if _isfinite(maturity) else np.nan,
            "growth_strength": float(growth_strength) if _isfinite(growth_strength) else np.nan,
            "fit_quality": float(fit_quality) if _isfinite(fit_quality) else np.nan,
            "decline_risk": float(decline_risk),
        }

        diagnostics = {
            "fit_ok": bool(fit_ok) if has_fit_metrics else None,
            "fit_bad_reasons": fit_bad_reasons,
            "progress": progress,
            "cap": cap,
            "peak_slope": peak_slope,
            "slope_rel": slope_rel,
            "r2": r2,
            "rmse_norm_range": rmse_nr,
            "monotonicity": mono,
            "pre_slope": pre_slope,
            "fb_growth_total": fb_growth,
            "fb_growth_strength": fb_growth_strength,
            "max_drawdown": max_dd,
            "time_span": time_span,
            "y_end": y_end,
        }

        return StageResult(stage=stage, confidence=conf, components=components, diagnostics=diagnostics)
