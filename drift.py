# scurve/report/drift.py
"""
Drift monitoring for the S-Curve model.

Purpose
- Compare current run diagnostics/scores/features to a baseline snapshot.
- Flag large changes ("drift") that may indicate:
    - data source changes
    - universe definition changes
    - model behavior changes
    - regime shifts that break assumptions

This is NOT a statistical guarantee. It is a deterministic monitoring layer.

Typical workflow
- Save a baseline JSON/CSV snapshot from a known-good run:
    - summary metrics (fit pass rate, avg score, stage distribution)
    - selected feature distributions (mean/std/quantiles)
    - portfolio summary (n holdings, turnover, sector exposure)
- On each new run, compute a "current snapshot" with the same schema
  and call compare_snapshots(...).

Inputs
- snapshots are dictionaries (serializable) produced by helpers below
  OR loaded from disk by your higher-level report module.

Outputs
- DriftReport with:
    - drift_score 0..100 (higher = more drift)
    - flags list[str]
    - per-metric diffs

No external deps beyond numpy/pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return float(default)
    return float(a / b)


def _nan_to_num(x: float, default: float = 0.0) -> float:
    return float(x) if np.isfinite(x) else float(default)


def _abs_pct_change(curr: float, base: float) -> float:
    if not (np.isfinite(curr) and np.isfinite(base)):
        return np.nan
    denom = abs(base) if abs(base) > 1e-12 else np.nan
    return float(abs(curr - base) / denom) if np.isfinite(denom) else np.nan


def _abs_change(curr: float, base: float) -> float:
    if not (np.isfinite(curr) and np.isfinite(base)):
        return np.nan
    return float(abs(curr - base))


def _quantiles(x: pd.Series, qs: Tuple[float, ...] = (0.1, 0.5, 0.9)) -> Dict[str, float]:
    x = pd.to_numeric(x, errors="coerce")
    out = {}
    for q in qs:
        out[f"q{int(q*100):02d}"] = _safe_float(x.quantile(q))
    return out


@dataclass
class DriftConfig:
    """
    Config for drift detection.

    drift_score is built from:
      - weighted sum of normalized diffs
      - plus flags for stage distribution shifts

    Thresholds are deliberately simple defaults.
    """

    # scalar thresholds
    score_mean_abs: float = 10.0          # avg score shift > 10 points
    score_std_abs: float = 10.0           # score std shift > 10 points
    fit_pass_rate_abs: float = 0.15       # pass rate shift > 15 percentage points
    turnover_abs: float = 0.25            # turnover shift > 0.25

    # stage distribution L1 shift threshold
    stage_l1_abs: float = 0.40            # sum(|p_i - q_i|) > 0.40

    # feature distribution thresholds (mean abs change in z-score units)
    feat_mean_z: float = 1.0              # |Δmean| / base_std > 1.0

    # weights for drift score
    w_score_mean: float = 0.25
    w_fit_pass: float = 0.20
    w_stage: float = 0.20
    w_turnover: float = 0.15
    w_score_std: float = 0.10
    w_features: float = 0.10

    # cap
    max_score: float = 100.0


@dataclass
class DriftReport:
    drift_score: float
    flags: List[str]
    diffs: Dict[str, Any]


def make_run_snapshot(
    results_df: pd.DataFrame,
    *,
    score_col: str = "score",
    stage_col: str = "stage",
    fit_pass_col: str = "fit_pass",
    selected_features: Optional[List[str]] = None,
    portfolio_summary: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a serializable snapshot from a results DataFrame.

    results_df should be at the per-asset level (one row per ticker/id) and include:
      - score
      - stage
      - fit_pass (bool/int)

    selected_features:
      optional list of numeric columns to track distribution stats
    portfolio_summary:
      optional dict like {"n_holdings":..., "turnover":..., "sector_top":...}
    """
    df = results_df.copy()

    snap: Dict[str, Any] = {"summary": {}, "stage_dist": {}, "features": {}, "portfolio": portfolio_summary or {}}

    # score stats
    if score_col in df.columns:
        s = pd.to_numeric(df[score_col], errors="coerce")
        snap["summary"]["score_mean"] = _safe_float(s.mean())
        snap["summary"]["score_std"] = _safe_float(s.std())
        snap["summary"]["score_q"] = _quantiles(s)
    else:
        snap["summary"]["score_mean"] = np.nan
        snap["summary"]["score_std"] = np.nan
        snap["summary"]["score_q"] = {}

    # fit pass rate
    if fit_pass_col in df.columns:
        fp = pd.to_numeric(df[fit_pass_col], errors="coerce")
        snap["summary"]["fit_pass_rate"] = _safe_float(fp.mean())
    else:
        snap["summary"]["fit_pass_rate"] = np.nan

    # stage distribution
    if stage_col in df.columns:
        st = df[stage_col].fillna("unknown").astype(str)
        dist = st.value_counts(normalize=True).to_dict()
        snap["stage_dist"] = {str(k): float(v) for k, v in dist.items()}
    else:
        snap["stage_dist"] = {}

    # selected feature distributions
    if selected_features:
        for col in selected_features:
            if col in df.columns:
                x = pd.to_numeric(df[col], errors="coerce")
                snap["features"][col] = {
                    "mean": _safe_float(x.mean()),
                    "std": _safe_float(x.std()),
                    **_quantiles(x),
                }

    return snap


def _stage_l1(curr: Dict[str, float], base: Dict[str, float]) -> float:
    keys = set(curr.keys()) | set(base.keys())
    return float(sum(abs(curr.get(k, 0.0) - base.get(k, 0.0)) for k in keys))


def compare_snapshots(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    *,
    cfg: Optional[DriftConfig] = None,
) -> DriftReport:
    """
    Compare current vs baseline snapshot.

    Returns DriftReport.
    """
    cfg = cfg or DriftConfig()

    flags: List[str] = []
    diffs: Dict[str, Any] = {"summary": {}, "stage": {}, "features": {}, "portfolio": {}}

    # ---- summary diffs ----
    csum = current.get("summary", {})
    bsum = baseline.get("summary", {})

    for k in ("score_mean", "score_std", "fit_pass_rate"):
        curr_v = _safe_float(csum.get(k, np.nan))
        base_v = _safe_float(bsum.get(k, np.nan))
        diffs["summary"][k] = {"current": curr_v, "baseline": base_v, "abs_change": _abs_change(curr_v, base_v)}

    # flags
    if np.isfinite(diffs["summary"]["score_mean"]["abs_change"]) and diffs["summary"]["score_mean"]["abs_change"] > cfg.score_mean_abs:
        flags.append("score_mean_shift")
    if np.isfinite(diffs["summary"]["score_std"]["abs_change"]) and diffs["summary"]["score_std"]["abs_change"] > cfg.score_std_abs:
        flags.append("score_std_shift")
    if np.isfinite(diffs["summary"]["fit_pass_rate"]["abs_change"]) and diffs["summary"]["fit_pass_rate"]["abs_change"] > cfg.fit_pass_rate_abs:
        flags.append("fit_pass_rate_shift")

    # ---- stage distribution drift ----
    cst = current.get("stage_dist", {}) or {}
    bst = baseline.get("stage_dist", {}) or {}
    l1 = _stage_l1({k: float(v) for k, v in cst.items()}, {k: float(v) for k, v in bst.items()})
    diffs["stage"] = {"l1": float(l1), "current": cst, "baseline": bst}
    if np.isfinite(l1) and l1 > cfg.stage_l1_abs:
        flags.append("stage_dist_shift")

    # ---- portfolio summary drift (if present) ----
    cport = current.get("portfolio", {}) or {}
    bport = baseline.get("portfolio", {}) or {}
    # common keys
    for k in ("turnover", "n_holdings", "gross", "net"):
        if k in cport or k in bport:
            curr_v = _safe_float(cport.get(k, np.nan))
            base_v = _safe_float(bport.get(k, np.nan))
            diffs["portfolio"][k] = {"current": curr_v, "baseline": base_v, "abs_change": _abs_change(curr_v, base_v)}

    if "turnover" in diffs["portfolio"]:
        if np.isfinite(diffs["portfolio"]["turnover"]["abs_change"]) and diffs["portfolio"]["turnover"]["abs_change"] > cfg.turnover_abs:
            flags.append("turnover_shift")

    # ---- feature distribution drift ----
    cfeat = current.get("features", {}) or {}
    bfeat = baseline.get("features", {}) or {}
    feat_scores = []

    for col, cs in cfeat.items():
        if col not in bfeat:
            continue
        bs = bfeat[col]
        cmean = _safe_float(cs.get("mean", np.nan))
        bmean = _safe_float(bs.get("mean", np.nan))
        bstd = _safe_float(bs.get("std", np.nan))
        z = _safe_div(abs(cmean - bmean), bstd, default=np.nan)
        diffs["features"][col] = {"mean_current": cmean, "mean_baseline": bmean, "baseline_std": bstd, "mean_z": z}
        if np.isfinite(z):
            feat_scores.append(min(3.0, float(z)))  # cap each feature contribution
        if np.isfinite(z) and z > cfg.feat_mean_z:
            flags.append(f"feature_mean_shift:{col}")

    # ---- compute drift_score 0..100 ----
    # Normalize components (rough heuristics)
    score_mean_n = _nan_to_num(_safe_div(diffs["summary"]["score_mean"]["abs_change"], cfg.score_mean_abs, default=0.0))
    score_std_n = _nan_to_num(_safe_div(diffs["summary"]["score_std"]["abs_change"], cfg.score_std_abs, default=0.0))
    fit_pass_n = _nan_to_num(_safe_div(diffs["summary"]["fit_pass_rate"]["abs_change"], cfg.fit_pass_rate_abs, default=0.0))
    stage_n = _nan_to_num(_safe_div(l1, cfg.stage_l1_abs, default=0.0))
    turnover_n = 0.0
    if "turnover" in diffs["portfolio"]:
        turnover_n = _nan_to_num(_safe_div(diffs["portfolio"]["turnover"]["abs_change"], cfg.turnover_abs, default=0.0))

    feat_n = 0.0
    if len(feat_scores) > 0:
        feat_n = float(np.clip(np.mean(feat_scores) / cfg.feat_mean_z, 0.0, 2.0))

    # Weighted sum, then map to 0..100
    wsum = float(cfg.w_score_mean + cfg.w_fit_pass + cfg.w_stage + cfg.w_turnover + cfg.w_score_std + cfg.w_features)
    if wsum <= 0:
        wsum = 1.0

    drift_0_1 = (
        cfg.w_score_mean * score_mean_n
        + cfg.w_fit_pass * fit_pass_n
        + cfg.w_stage * stage_n
        + cfg.w_turnover * turnover_n
        + cfg.w_score_std * score_std_n
        + cfg.w_features * feat_n
    ) / wsum

    drift_0_1 = float(np.clip(drift_0_1, 0.0, 2.0))  # allow >1 for "very large" drift
    drift_score = float(np.clip(drift_0_1 * 50.0, 0.0, float(cfg.max_score)))  # 1.0 -> 50, 2.0 -> 100

    diffs["normalized"] = {
        "score_mean_n": score_mean_n,
        "score_std_n": score_std_n,
        "fit_pass_n": fit_pass_n,
        "stage_n": stage_n,
        "turnover_n": turnover_n,
        "feat_n": feat_n,
    }

    return DriftReport(drift_score=drift_score, flags=flags, diffs=diffs)