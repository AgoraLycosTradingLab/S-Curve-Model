# scurve/report/summary.py
"""
Summary reporting utilities for the S-Curve model.

Purpose
- Create deterministic run summaries from per-asset results + portfolio outputs.
- Provide:
  - headline metrics (fit pass rate, mean/median score, stage distribution)
  - top/bottom lists
  - portfolio summary (turnover, concentration, sector exposure)
  - convenience export helpers (to dict / JSON-ready)

This module is intentionally lightweight and does not write files directly.
File writing lives in scurve/report/writer.py (or caller).

Expected inputs
- results_df: one row per asset with columns like:
    ticker, score, stage, stage_confidence, fit_pass, chosen_model
  plus any diagnostics columns you include.
- weights_df: portfolio holdings with columns:
    ticker, weight, score, sector (optional)
- prev_weights_df (optional): previous portfolio for turnover stats

No external deps beyond numpy/pandas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _as_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _safe_float(x: Any, default: float = np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _turnover(curr: pd.Series, prev: pd.Series) -> float:
    """
    Simple turnover = sum(|w_new - w_prev|).
    Aligns by index. Missing treated as 0.
    """
    curr = curr.fillna(0.0)
    prev = prev.fillna(0.0)
    idx = curr.index.union(prev.index)
    d = curr.reindex(idx).fillna(0.0) - prev.reindex(idx).fillna(0.0)
    return float(np.sum(np.abs(d)))


def _hhi(w: pd.Series) -> float:
    """
    Herfindahl-Hirschman Index on weights (sum of squares).
    """
    w = w.fillna(0.0).astype(float)
    return float(np.sum(w * w))


def _top_n(df: pd.DataFrame, col: str, n: int, asc: bool = False) -> pd.DataFrame:
    d = df.copy()
    d[col] = _as_numeric(d[col])
    d = d.sort_values([col], ascending=[asc]).head(int(n))
    return d


@dataclass
class SummaryConfig:
    ticker_col: str = "ticker"
    score_col: str = "score"
    stage_col: str = "stage"
    stage_conf_col: str = "stage_confidence"
    fit_pass_col: str = "fit_pass"
    model_col: str = "chosen_model"

    # lists
    top_k: int = 20
    bottom_k: int = 20

    # portfolio
    weight_col: str = "weight"
    sector_col: str = "sector"
    sector_top_k: int = 10


@dataclass
class RunSummary:
    headline: Dict[str, Any]
    stage_dist: Dict[str, float]
    model_dist: Dict[str, float]
    top: List[Dict[str, Any]]
    bottom: List[Dict[str, Any]]
    portfolio: Dict[str, Any]
    diagnostics: Dict[str, Any]


def summarize_run(
    results_df: pd.DataFrame,
    *,
    weights_df: Optional[pd.DataFrame] = None,
    prev_weights_df: Optional[pd.DataFrame] = None,
    cfg: Optional[SummaryConfig] = None,
) -> RunSummary:
    """
    Build a run summary from per-asset results and optional portfolio.

    Returns RunSummary which is JSON-friendly (dict/list scalars).
    """
    cfg = cfg or SummaryConfig()

    df = results_df.copy()
    if cfg.ticker_col not in df.columns:
        raise ValueError(f"results_df missing '{cfg.ticker_col}'")

    # numeric score
    if cfg.score_col in df.columns:
        df[cfg.score_col] = _as_numeric(df[cfg.score_col])

    # Headline metrics
    n = int(len(df))
    fit_pass_rate = np.nan
    if cfg.fit_pass_col in df.columns:
        fp = _as_numeric(df[cfg.fit_pass_col])
        fit_pass_rate = _safe_float(fp.mean())

    headline = {
        "n_assets": n,
        "fit_pass_rate": float(fit_pass_rate) if np.isfinite(fit_pass_rate) else None,
        "score_mean": float(_safe_float(df[cfg.score_col].mean())) if cfg.score_col in df.columns else None,
        "score_median": float(_safe_float(df[cfg.score_col].median())) if cfg.score_col in df.columns else None,
        "score_std": float(_safe_float(df[cfg.score_col].std())) if cfg.score_col in df.columns else None,
    }

    # Stage distribution
    stage_dist: Dict[str, float] = {}
    if cfg.stage_col in df.columns:
        st = df[cfg.stage_col].fillna("unknown").astype(str)
        stage_dist = {str(k): float(v) for k, v in st.value_counts(normalize=True).to_dict().items()}

    # Model distribution
    model_dist: Dict[str, float] = {}
    if cfg.model_col in df.columns:
        md = df[cfg.model_col].fillna("none").astype(str)
        model_dist = {str(k): float(v) for k, v in md.value_counts(normalize=True).to_dict().items()}

    # Top/bottom lists
    top_rows: List[Dict[str, Any]] = []
    bot_rows: List[Dict[str, Any]] = []
    if cfg.score_col in df.columns:
        keep_cols = [c for c in [cfg.ticker_col, cfg.score_col, cfg.stage_col, cfg.stage_conf_col, cfg.fit_pass_col, cfg.model_col] if c in df.columns]

        top_df = _top_n(df[keep_cols], cfg.score_col, cfg.top_k, asc=False)
        bot_df = _top_n(df[keep_cols], cfg.score_col, cfg.bottom_k, asc=True)

        top_rows = top_df.to_dict(orient="records")
        bot_rows = bot_df.to_dict(orient="records")

    # Portfolio summary
    portfolio: Dict[str, Any] = {}
    if weights_df is not None and isinstance(weights_df, pd.DataFrame) and len(weights_df) > 0:
        wdf = weights_df.copy()
        if cfg.ticker_col not in wdf.columns or cfg.weight_col not in wdf.columns:
            raise ValueError(f"weights_df must contain '{cfg.ticker_col}' and '{cfg.weight_col}'")

        wdf[cfg.weight_col] = _as_numeric(wdf[cfg.weight_col]).fillna(0.0)
        w = pd.Series(wdf[cfg.weight_col].to_numpy(dtype=float), index=wdf[cfg.ticker_col].astype(str))

        portfolio = {
            "n_holdings": int((np.abs(w) > 0).sum()),
            "gross": float(np.sum(np.abs(w))),
            "net": float(np.sum(w)),
            "hhi": float(_hhi(w)),
            "top_weight": float(_safe_float(w.max())),
            "top5_weight": float(_safe_float(np.sort(w.values)[-5:].sum())) if len(w) >= 5 else float(_safe_float(w.sum())),
        }

        # Turnover (if previous provided)
        if prev_weights_df is not None and isinstance(prev_weights_df, pd.DataFrame) and len(prev_weights_df) > 0:
            pw = pd.Series(
                _as_numeric(prev_weights_df[cfg.weight_col]).fillna(0.0).to_numpy(dtype=float),
                index=prev_weights_df[cfg.ticker_col].astype(str),
            )
            portfolio["turnover"] = float(_turnover(w, pw))
        else:
            portfolio["turnover"] = None

        # Sector exposure
        if cfg.sector_col in wdf.columns:
            sec = wdf[cfg.sector_col].fillna("UNKNOWN").astype(str)
            sec_w = w.groupby(sec).sum().sort_values(ascending=False)
            portfolio["sector_exposure_top"] = [
                {"sector": str(k), "weight": float(v)} for k, v in sec_w.head(cfg.sector_top_k).items()
            ]

    diagnostics = {
        "stage_dist": stage_dist,
        "model_dist": model_dist,
    }

    return RunSummary(
        headline=headline,
        stage_dist=stage_dist,
        model_dist=model_dist,
        top=top_rows,
        bottom=bot_rows,
        portfolio=portfolio,
        diagnostics=diagnostics,
    )


def summary_to_dict(summary: RunSummary) -> Dict[str, Any]:
    """
    Convert RunSummary into a JSON-serializable dict.
    """
    return {
        "headline": summary.headline,
        "stage_dist": summary.stage_dist,
        "model_dist": summary.model_dist,
        "top": summary.top,
        "bottom": summary.bottom,
        "portfolio": summary.portfolio,
        "diagnostics": summary.diagnostics,
    }