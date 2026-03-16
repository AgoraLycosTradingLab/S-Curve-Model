# scurve/score/composite.py
"""
Composite scoring for the S-Curve model.

Purpose
- Convert stage components + pre/post/fallback features into:
  - a deterministic 0..100 composite score
  - interpretable subscores (0..100)
  - gating flags (fit_ok, data_ok, etc.)

Overlay modes (driven by cfg['overlays'] via score_name adapter)
- A: Revenue-only (core structural S-curve components only)
- B: Revenue + EPS revisions
- C: Revenue + EPS revisions + breakout confirmation
- D: Full charter overlays (same as C unless extended)

Important
- CompositeScorer consumes a flat feature dict plus a StageResult.
- run.py in this codebase calls score_name(...), so score_name is the
  integration/compat layer that:
    1) reads overlay toggles from cfg
    2) builds CompositeScorer config
    3) converts StageFeatures/FitResult into a StageResult-like object
    4) returns scurve.core.types.ScoreResult (legacy interface)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from scurve.score.stage import StageResult


def _get(d: Dict[str, Any], k: str, default: float = np.nan) -> float:
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return float(default)


def _isfinite(x: float) -> bool:
    return bool(np.isfinite(x))


def _clip(x: float, lo: float, hi: float) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(np.clip(x, lo, hi))


def _clip01(x: float) -> float:
    return _clip(x, 0.0, 1.0)


def _to_0_100(x01: float) -> float:
    if not np.isfinite(x01):
        return float("nan")
    return float(np.clip(x01, 0.0, 1.0) * 100.0)


def _tanh01(z: float) -> float:
    """
    Smooth mapping from z-score-ish values to 0..1.
    - z=0 -> 0.5
    - large positive -> ~1
    - large negative -> ~0
    """
    if not np.isfinite(z):
        return float("nan")
    return float(np.clip(0.5 + 0.5 * np.tanh(z), 0.0, 1.0))


@dataclass
class CompositeConfig:
    """
    Composite scoring weights.

    Each subscore is 0..1 internally, then scaled to 0..100.
    Weights are automatically renormalized across enabled / available blocks.

    Core blocks (always considered):
      - growth
      - maturity
      - quality
      - risk (penalty)

    Optional overlays (only if enabled):
      - revision (EPS revisions / expectations confirmation)
      - breakout (price breakout / relative strength confirmation)
    """

    # ---- core weights ----
    w_growth: float = 0.35
    w_maturity: float = 0.20
    w_quality: float = 0.15
    w_risk: float = 0.10  # subtractive

    # ---- overlay weights ----
    w_revision: float = 0.10
    w_breakout: float = 0.10
    w_valuation: float = 0.10

    # ---- overlay toggles ----
    use_revision: bool = False
    use_breakout: bool = False
    use_valuation: bool = True

    # ---- maturity preference band ----
    maturity_center: float = 0.50
    maturity_width: float = 0.35

    # ---- penalties / boosts ----
    penalty_missing: float = 0.05  # per missing enabled block (capped)
    penalty_decline_stage: float = 0.30
    boost_growth_stage: float = 0.05

    # ---- data quality thresholds ----
    min_data_points: int = 6
    min_monotonicity: float = 0.55

    # ---- feature keys (override upstream if needed) ----
    key_rev_z_3m: str = "rev_zscore_3m"
    key_breakout_flag: str = "breakout_flag"
    key_rel_strength_12m: str = "rel_strength_12m"
    key_fcf_yield: str = "fcf_yield"
    key_gross_margin: str = "gross_margin"
    key_operating_margin: str = "operating_margin"
    key_pe_ratio: str = "pe_ratio"
    key_ev_to_fcf: str = "ev_to_fcf"


@dataclass
class CompositeScoreResult:
    score: float
    subscores: Dict[str, float]
    flags: Dict[str, Any]
    diagnostics: Dict[str, Any]


class CompositeScorer:
    """
    Deterministic composite scorer.

    Usage:
        scorer = CompositeScorer(CompositeConfig(use_revision=True, use_breakout=False))
        out = scorer.score(features, stage_result)
    """

    def __init__(self, config: Optional[CompositeConfig] = None):
        self.config = config or CompositeConfig()

    def score(self, feats: Dict[str, Any], stage: StageResult) -> CompositeScoreResult:
        cfg = self.config

        # ---- Pull stage components (0..1) ----
        maturity = _get(stage.components, "maturity", np.nan)
        growth_strength = _get(stage.components, "growth_strength", np.nan)
        fit_quality = _get(stage.components, "fit_quality", np.nan)
        decline_risk = _get(stage.components, "decline_risk", np.nan)

        # ---- Additional proxies from features ----
        mono = _get(feats, "pre_monotonicity", _get(feats, "fb_monotonicity", np.nan))
        n_obs = _get(feats, "pre_n_obs", _get(feats, "fb_n_used", _get(feats, "n_obs", np.nan)))
        if not _isfinite(n_obs):
            n_obs = _get(feats, "pre_n_obs", np.nan)

        # ---- Subscore: growth (0..1) ----
        growth = _clip01(growth_strength)
        if not _isfinite(growth):
            growth = _clip01(_get(feats, "fb_growth_strength_01", np.nan))

        # ---- Subscore: maturity preference (0..1) ----
        if _isfinite(maturity):
            d = abs(maturity - cfg.maturity_center)
            maturity_pref = 1.0 - (d / max(1e-12, cfg.maturity_width)) ** 2
            maturity_pref = float(np.clip(maturity_pref, 0.0, 1.0))
        else:
            maturity_pref = np.nan

        # ---- Subscore: quality (0..1) ----
        q_parts = []
        if _isfinite(fit_quality):
            q_parts.append(_clip01(fit_quality))
        if _isfinite(mono):
            q_parts.append(float(np.clip((mono - 0.5) / 0.5, 0.0, 1.0)))
        fb_r2 = _get(feats, "fb_r2", np.nan)
        if _isfinite(fb_r2):
            q_parts.append(_clip01(fb_r2))
        fb_sign = _get(feats, "fb_sign_consistency", np.nan)
        if _isfinite(fb_sign):
            q_parts.append(_clip01(fb_sign))
        quality = float(np.mean(q_parts)) if q_parts else np.nan

        # ---- Subscore: risk penalty (0..1) ----
        risk = _clip01(decline_risk)

        # ---- Overlay: EPS revisions (0..1) ----
        revision = np.nan
        if cfg.use_revision:
            rev_z = _get(feats, cfg.key_rev_z_3m, np.nan)
            if _isfinite(rev_z):
                revision = _tanh01(rev_z)

        # ---- Overlay: breakout confirmation (0..1) ----
        breakout = np.nan
        if cfg.use_breakout:
            brk_flag = _get(feats, cfg.key_breakout_flag, np.nan)
            rs_12m = _get(feats, cfg.key_rel_strength_12m, np.nan)

            b_parts = []
            if _isfinite(brk_flag):
                b_parts.append(_clip01(brk_flag))
            if _isfinite(rs_12m):
                b_parts.append(_tanh01(rs_12m))

            if b_parts:
                breakout = float(np.mean(b_parts))

        # ---- Overlay: valuation quality (0..1) ----
        valuation = np.nan
        if cfg.use_valuation:
            v_parts = []
            fcf_yield = _get(feats, cfg.key_fcf_yield, np.nan)
            if _isfinite(fcf_yield):
                v_parts.append(_clip01(0.5 + 0.5 * np.tanh(10.0 * fcf_yield)))

            gross_margin = _get(feats, cfg.key_gross_margin, np.nan)
            if _isfinite(gross_margin):
                v_parts.append(_clip01((gross_margin - 0.20) / 0.50))

            op_margin = _get(feats, cfg.key_operating_margin, np.nan)
            if _isfinite(op_margin):
                v_parts.append(_clip01((op_margin - 0.05) / 0.30))

            pe_ratio = _get(feats, cfg.key_pe_ratio, np.nan)
            if _isfinite(pe_ratio) and pe_ratio > 0:
                v_parts.append(_clip01(1.0 - (np.log(pe_ratio) - np.log(15.0)) / np.log(4.0)))

            ev_to_fcf = _get(feats, cfg.key_ev_to_fcf, np.nan)
            if _isfinite(ev_to_fcf) and ev_to_fcf > 0:
                v_parts.append(_clip01(1.0 - (np.log(ev_to_fcf) - np.log(12.0)) / np.log(5.0)))

            if v_parts:
                valuation = float(np.mean(v_parts))

        # ---- Build parts/weights dynamically and renormalize ----
        parts: Dict[str, float] = {
            "growth": growth,
            "maturity": maturity_pref,
            "quality": quality,
            "risk": risk,  # subtractive
        }
        weights: Dict[str, float] = {
            "growth": cfg.w_growth,
            "maturity": cfg.w_maturity,
            "quality": cfg.w_quality,
            "risk": cfg.w_risk,
        }

        if cfg.use_revision:
            parts["revision"] = revision
            weights["revision"] = cfg.w_revision
        if cfg.use_breakout:
            parts["breakout"] = breakout
            weights["breakout"] = cfg.w_breakout
        if cfg.use_valuation:
            parts["valuation"] = valuation
            weights["valuation"] = cfg.w_valuation

        keys = list(parts.keys())
        vals = np.array([parts[k] for k in keys], dtype=float)
        w = np.array([weights[k] for k in keys], dtype=float)

        # If risk missing, treat as zero penalty
        for i, k in enumerate(keys):
            if k == "risk" and not np.isfinite(vals[i]):
                vals[i] = 0.0

        valid = np.isfinite(vals)
        w_eff = w.copy()
        w_eff[~valid] = 0.0

        w_sum = float(np.sum(np.abs(w_eff)))
        if w_sum <= 0.0:
            base = 0.0
            w_norm = w_eff
        else:
            w_norm = w_eff / w_sum
            base = 0.0
            for i, k in enumerate(keys):
                if not valid[i]:
                    continue
                if k == "risk":
                    base -= float(w_norm[i] * vals[i])
                else:
                    base += float(w_norm[i] * vals[i])

        base = float(np.clip(base, 0.0, 1.0))

        # ---- Missingness penalty ----
        missing_blocks = 0
        if not _isfinite(growth_strength):
            missing_blocks += 1
        if not _isfinite(maturity):
            missing_blocks += 1
        if not _isfinite(fit_quality):
            missing_blocks += 1
        if not _isfinite(mono) and not _isfinite(fb_sign):
            missing_blocks += 1
        if cfg.use_revision and not _isfinite(revision):
            missing_blocks += 1
        if cfg.use_breakout and not _isfinite(breakout):
            missing_blocks += 1
        if cfg.use_valuation and not _isfinite(valuation):
            missing_blocks += 1

        miss_pen = float(np.clip(missing_blocks * cfg.penalty_missing, 0.0, 0.25))
        score01 = float(np.clip(base - miss_pen, 0.0, 1.0))

        # ---- Stage-based adjustments ----
        if stage.stage == "decline":
            score01 = float(np.clip(score01 - cfg.penalty_decline_stage, 0.0, 1.0))
        elif stage.stage == "growth":
            score01 = float(np.clip(score01 + cfg.boost_growth_stage, 0.0, 1.0))

        # ---- Data quality flags ----
        data_ok = True
        if _isfinite(n_obs) and int(round(n_obs)) < int(cfg.min_data_points):
            data_ok = False

        flags = {
            "data_ok": bool(data_ok),
            "stage": stage.stage,
            "stage_confidence": float(stage.confidence),
            "use_revision": bool(cfg.use_revision),
            "use_breakout": bool(cfg.use_breakout),
            "use_valuation": bool(cfg.use_valuation),
        }

        subscores: Dict[str, float] = {
            "growth_0_100": _to_0_100(growth),
            "maturity_0_100": _to_0_100(maturity_pref),
            "quality_0_100": _to_0_100(quality),
            "risk_0_100": _to_0_100(risk),
        }
        if cfg.use_revision:
            subscores["revision_0_100"] = _to_0_100(revision)
        if cfg.use_breakout:
            subscores["breakout_0_100"] = _to_0_100(breakout)
        if cfg.use_valuation:
            subscores["valuation_0_100"] = _to_0_100(valuation)

        diagnostics = {
            "base_score_0_1": base,
            "missing_blocks": int(missing_blocks),
            "missing_penalty_0_1": miss_pen,
            "weights_effective": {k: float(w_norm[i]) for i, k in enumerate(keys)},
            "components_0_1": {
                "maturity": float(maturity) if _isfinite(maturity) else np.nan,
                "growth_strength": float(growth_strength) if _isfinite(growth_strength) else np.nan,
                "fit_quality": float(fit_quality) if _isfinite(fit_quality) else np.nan,
                "decline_risk": float(decline_risk) if _isfinite(decline_risk) else np.nan,
                "monotonicity": float(mono) if _isfinite(mono) else np.nan,
                "n_obs": float(n_obs) if _isfinite(n_obs) else np.nan,
                "revision": float(revision) if _isfinite(revision) else np.nan,
                "breakout": float(breakout) if _isfinite(breakout) else np.nan,
                "valuation": float(valuation) if _isfinite(valuation) else np.nan,
            },
            "feature_keys": {
                "rev_z_3m": cfg.key_rev_z_3m,
                "breakout_flag": cfg.key_breakout_flag,
                "rel_strength_12m": cfg.key_rel_strength_12m,
                "fcf_yield": cfg.key_fcf_yield,
                "gross_margin": cfg.key_gross_margin,
                "operating_margin": cfg.key_operating_margin,
                "pe_ratio": cfg.key_pe_ratio,
                "ev_to_fcf": cfg.key_ev_to_fcf,
            },
        }

        return CompositeScoreResult(
            score=float(np.clip(score01 * 100.0, 0.0, 100.0)),
            subscores=subscores,
            flags=flags,
            diagnostics=diagnostics,
        )


# -----------------------------------------------------------------------------
# Compatibility adapter for scurve/run.py
# -----------------------------------------------------------------------------

from scurve.core.types import ScoreResult


def _stage_label_from_maturity(maturity_ratio: float) -> str:
    # Keep this mapping simple and deterministic.
    if not np.isfinite(maturity_ratio):
        return "Unknown"
    if maturity_ratio < 0.33:
        return "early"
    if maturity_ratio < 0.75:
        return "growth"
    return "mature"


def score_name(
    cfg: Dict[str, Any],
    ticker: str,
    series: Any,
    stage_feats: Any,
    fit: Any,
    mkt_overlays: Optional[Dict[str, Any]] = None,
    logger: Any = None,
) -> ScoreResult:
    """
    Adapter used by scurve/run.py.

    Reads overlay toggles from cfg and returns scurve.core.types.ScoreResult.

    Parameters
    ----------
    cfg : dict
        Loaded config dict (from scurve.core.config.load_config)
    stage_feats : StageFeatures-like
        Expected to have: maturity_ratio, normalized_slope, curvature, accel_flag
        (see scurve.core.types.StageFeatures)
    fit : FitResult-like or None
        In this codebase, scurve.core.types.FitResult has fields:
        ok, family, params, nrmse, sse, k_on_upper_bound, message
    mkt_overlays : dict | None
        Optional dict of overlay features (e.g., rev_zscore_3m, breakout_flag, rel_strength_12m).
    """

    # 1) Build overlay-aware composite config
    overlays = cfg.get("overlays", {}) if isinstance(cfg, dict) else {}
    comp_cfg = CompositeConfig(
        use_revision=bool(overlays.get("eps_revisions", {}).get("enabled", False)),
        use_breakout=bool(overlays.get("breakout", {}).get("enabled", False)),
        # you can also override maturity_center/width/weights from cfg here later
    )
    scorer = CompositeScorer(comp_cfg)

    # 2) Build feature dict (only what composite needs)
    feats: Dict[str, Any] = {}
    if isinstance(mkt_overlays, dict):
        feats.update(mkt_overlays)

    # 3) Convert StageFeatures -> StageResult expected by CompositeScorer
    maturity_ratio = float(getattr(stage_feats, "maturity_ratio", np.nan))
    normalized_slope = float(getattr(stage_feats, "normalized_slope", np.nan))

    fit_ok = bool(getattr(fit, "ok", False)) if fit is not None else False
    nrmse = float(getattr(fit, "nrmse", np.nan)) if fit is not None else np.nan

    # Map nrmse to a 0..1 fit_quality (simple monotone mapping)
    # - nrmse ~ 0 -> ~1
    # - nrmse >= 1 -> ~0
    fit_quality = np.nan
    if np.isfinite(nrmse):
        fit_quality = float(np.clip(1.0 - np.clip(nrmse, 0.0, 1.0), 0.0, 1.0))
    elif fit_ok:
        fit_quality = 0.6  # fallback if ok but nrmse missing

    stage_label = _stage_label_from_maturity(maturity_ratio)

    stage_result = StageResult(
        stage=stage_label,
        confidence=0.75 if fit_ok else 0.40,
        components={
            "maturity": float(np.clip(maturity_ratio, 0.0, 1.0)) if np.isfinite(maturity_ratio) else np.nan,
            "growth_strength": float(np.clip(normalized_slope, 0.0, 1.0)) if np.isfinite(normalized_slope) else np.nan,
            "fit_quality": float(fit_quality) if np.isfinite(fit_quality) else np.nan,
            "decline_risk": 0.0,  # this pipeline doesn't provide decline proxies yet
        },
    )

    comp_out = scorer.score(feats, stage_result)

    # 4) Return legacy ScoreResult expected by run.py
    return ScoreResult(
        score_total=float(comp_out.score),
        stage_label=str(stage_result.stage),
        score_stage=float(comp_out.subscores.get("growth_0_100", 0.0)),
        score_slope=float(comp_out.subscores.get("maturity_0_100", 0.0)),
        score_accel=float(comp_out.subscores.get("quality_0_100", 0.0)),
        fit_used=bool(fit_ok),
    )
