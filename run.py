from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# Allow direct script execution: python SCurve/scurve/run.py ...
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scurve.core.config import load_config, save_config_snapshot
from scurve.core.logging import init_logger, log_kv, make_log_path, switch_file_handler, write_run_metadata
from scurve.core.types import ScoreResult, StageFeatures
from scurve.core.utils import make_run_dir, parse_asof
from scurve.data.fundamentals import FundamentalsSeries, load_revenue_ttm_pit
from scurve.data.market import MarketSnapshot, load_market_overlays
from scurve.data.universe import build_universe
from scurve.features.fallback import build_fallback_features
from scurve.features.post_fit import build_post_fit_features
from scurve.features.pre_fit import build_pre_fit_features
from scurve.fit.fitters import FitResult, FittersConfig, fit_best_curve
from scurve.fit.gates import FitGate, GateConfig
from scurve.portfolio.construct import ConstructConfig, construct_portfolio
from scurve.portfolio.risk import RiskConfig, apply_risk_overlays
from scurve.report.summary import SummaryConfig, summarize_run, summary_to_dict
from scurve.score.composite import CompositeConfig, CompositeScorer
from scurve.score.stage import StageConfig, StageScorer


def _flatten_score(score: ScoreResult) -> Dict[str, Any]:
    return {
        "score_total": float(score.score_total),
        "stage_label": str(score.stage_label),
        "score_stage": float(score.score_stage),
        "score_slope": float(score.score_slope),
        "score_accel": float(score.score_accel),
        "fit_used": bool(score.fit_used),
    }


def _flatten_fit(fit: Optional[FitResult]) -> Dict[str, Any]:
    if fit is None:
        return {
            "fit_ok": False,
            "fit_family": "none",
            "fit_reason": "missing",
            "fit_r2": None,
            "fit_rmse": None,
            "fit_rmse_norm_range": None,
        }

    metrics = fit.diagnostics.metrics if fit.diagnostics is not None else {}
    return {
        "fit_ok": bool(fit.ok),
        "fit_family": str(fit.model),
        "fit_reason": fit.reason,
        "fit_r2": float(metrics.get("r2")) if fit.diagnostics and metrics.get("r2") is not None else None,
        "fit_rmse": float(metrics.get("rmse")) if fit.diagnostics and metrics.get("rmse") is not None else None,
        "fit_rmse_norm_range": float(metrics.get("rmse_norm_range")) if fit.diagnostics and metrics.get("rmse_norm_range") is not None else None,
    }


def _flatten_stage(stage: Optional[StageFeatures]) -> Dict[str, Any]:
    if stage is None:
        return {
            "maturity_ratio": None,
            "normalized_slope": None,
            "curvature": None,
            "accel_flag": None,
        }
    return {
        "maturity_ratio": float(stage.maturity_ratio),
        "normalized_slope": float(stage.normalized_slope),
        "curvature": float(stage.curvature),
        "accel_flag": int(stage.accel_flag),
    }


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _series_to_xy(series: FundamentalsSeries) -> tuple[np.ndarray, np.ndarray]:
    df = series.df.sort_values("period_end").reset_index(drop=True)
    y = df["value"].astype(float).to_numpy()
    t = np.arange(len(y), dtype=float)
    return t, y


def _market_to_feats(ms: Optional[MarketSnapshot]) -> Dict[str, Any]:
    if ms is None:
        return {}
    return {
        "breakout_flag": 1.0 if bool(ms.breakout_6m) else 0.0,
        "rel_strength_12m": ms.mom_12m,
        "adv_usd": ms.adv_dollar_3m,
        "vol": ms.vol_60d_ann,
    }


def _build_valuation_map(
    cfg: Dict[str, Any],
    universe: list[str],
    logger=None,
) -> Dict[str, Dict[str, Any]]:
    val_cfg = cfg.get("valuation", {}) if isinstance(cfg.get("valuation", {}), dict) else {}
    if not bool(val_cfg.get("enabled", True)):
        return {}
    if str(val_cfg.get("provider", "yfinance")).strip().lower() not in {"yfinance", "yf"}:
        return {}
    max_tickers = int(val_cfg.get("max_tickers", 0))
    tickers = universe[:max_tickers] if max_tickers > 0 else universe
    out: Dict[str, Dict[str, Any]] = {}
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        if logger:
            logger.warning("valuation_disabled reason=yfinance_not_available")
        return {}

    for tkr in tickers:
        try:
            info = yf.Ticker(tkr).get_info()
        except Exception:
            continue
        if not isinstance(info, dict) or not info:
            continue

        fcf = info.get("freeCashflow")
        market_cap = info.get("marketCap")
        enterprise_value = info.get("enterpriseValue")

        fcf_yield = None
        if market_cap not in (None, 0) and fcf is not None:
            try:
                fcf_yield = float(fcf) / float(market_cap)
            except Exception:
                fcf_yield = None

        ev_to_fcf = None
        if fcf not in (None, 0) and enterprise_value is not None:
            try:
                ev_to_fcf = float(enterprise_value) / float(fcf)
            except Exception:
                ev_to_fcf = None

        out[tkr] = {
            "pe_ratio": info.get("trailingPE", info.get("forwardPE")),
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "fcf_yield": fcf_yield,
            "ev_to_fcf": ev_to_fcf,
            "country": info.get("country"),
            "exchange": info.get("exchange"),
            "industry_name": info.get("industry"),
            "sector_name": info.get("sector"),
            "business_summary": info.get("longBusinessSummary"),
        }

    if logger:
        logger.info("valuation_loaded tickers_requested=%d tickers_with_data=%d", len(tickers), len(out))
    return out


def _apply_investability_policy(
    scores_df: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Apply mandate/investability filters before portfolio construction.
    """
    policy = cfg.get("policy", {}) if isinstance(cfg.get("policy", {}), dict) else {}
    if scores_df.empty:
        return scores_df

    df = scores_df.copy()

    # defaults: strict US-only and no crypto/mining-type businesses
    us_only = bool(policy.get("us_only", True))
    allowed_countries = {str(x).strip().upper() for x in policy.get("allowed_countries", ["UNITED STATES", "USA", "US"])}
    if us_only and "country" in df.columns:
        c = df["country"].astype(str).str.upper()
        df = df[c.isin(allowed_countries)]

    exclude_tickers = {str(x).strip().upper() for x in policy.get("exclude_tickers", [])}
    if exclude_tickers:
        df = df[~df["ticker"].astype(str).str.upper().isin(exclude_tickers)]

    exclude_kw = [str(x).strip().upper() for x in policy.get("exclude_business_keywords", ["CRYPTO", "BITCOIN", "DIGITAL ASSET", "MINING"])]
    if exclude_kw:
        industry = df.get("industry_name", pd.Series("", index=df.index)).astype(str).str.upper()
        summary = df.get("business_summary", pd.Series("", index=df.index)).astype(str).str.upper()
        mask = pd.Series(False, index=df.index)
        for kw in exclude_kw:
            if not kw:
                continue
            mask = mask | industry.str.contains(kw, na=False) | summary.str.contains(kw, na=False)
        df = df[~mask]

    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asof", required=True, help="As-of date YYYY-MM-DD")
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    asof = parse_asof(args.asof)

    cfg = load_config(args.config)
    run_dir = make_run_dir(cfg, asof=asof)

    logger = init_logger(cfg, run_dir)
    switch_file_handler(logger, make_log_path(cfg, asof=asof))

    save_config_snapshot(cfg, Path(run_dir) / "config_snapshot.yaml")
    write_run_metadata(cfg, run_dir, asof=asof, logger=logger)

    logger.info("run_start asof=%s run_dir=%s", asof, str(run_dir))

    universe = build_universe(cfg, asof=asof, logger=logger)
    rev_map = load_revenue_ttm_pit(cfg, universe, asof=asof, logger=logger)
    mkt_map = load_market_overlays(cfg, universe, asof=asof, logger=logger)
    valuation_map = _build_valuation_map(cfg, universe, logger=logger)

    rows_scores: list[Dict[str, Any]] = []
    rows_diag: list[Dict[str, Any]] = []

    n_total = 0
    n_fit_ok = 0
    strict_parametric_only = bool(cfg.get("fit", {}).get("strict_parametric_only", False))

    fit_cfg = FittersConfig(
        try_gompertz=("gompertz" in cfg.get("fit", {}).get("models", [])),
        try_bass=("bass" in cfg.get("fit", {}).get("models", [])),
    )
    gate = FitGate(GateConfig())
    stage_scorer = StageScorer(StageConfig())

    overlays = cfg.get("overlays", {}) if isinstance(cfg, dict) else {}
    comp_cfg = CompositeConfig(
        use_revision=bool(overlays.get("eps_revisions", {}).get("enabled", False)),
        use_breakout=bool(overlays.get("breakout", {}).get("enabled", False)),
        use_valuation=bool(cfg.get("valuation", {}).get("enabled", True)),
    )
    comp_scorer = CompositeScorer(comp_cfg)

    for ticker in universe:
        n_total += 1
        series = rev_map.get(ticker)
        if series is None or series.df.empty:
            continue

        t, y = _series_to_xy(series)

        pre = build_pre_fit_features(t, y)
        fit = fit_best_curve(t, y, cfg=fit_cfg)
        gate_res = gate.evaluate(fit, pre_features=pre.features)

        if gate_res.pass_fit and fit.curve is not None:
            n_fit_ok += 1
            post = build_post_fit_features(fit.curve, t, y, prefix="")
            feats = {**pre.features, **post.features}
        else:
            if strict_parametric_only:
                # In strict mode, only names with accepted Gompertz/Bass fits can be selected.
                diag_row = {
                    "asof": asof,
                    "ticker": ticker,
                    **_flatten_fit(fit),
                    "gate_pass": int(gate_res.pass_fit),
                    "gate_quality": float(gate_res.quality_score),
                    "gate_reasons": "|".join(gate_res.reasons) if gate_res.reasons else "",
                }
                rows_diag.append(diag_row)
                continue
            fb = build_fallback_features(t, y)
            feats = {**pre.features, **fb.features}

        mkt_feats = _market_to_feats(mkt_map.get(ticker))
        feats.update(mkt_feats)
        feats.update(valuation_map.get(ticker, {}))

        stage_result = stage_scorer.score(feats)
        comp_out = comp_scorer.score(feats, stage_result)

        curvature = float(feats.get("pre_curvature", feats.get("fb_curvature", np.nan)))
        stage_feats = StageFeatures(
            maturity_ratio=float(stage_result.components.get("maturity", np.nan)),
            normalized_slope=float(stage_result.components.get("growth_strength", np.nan)),
            curvature=curvature,
            accel_flag=1 if np.isfinite(curvature) and curvature > 0 else 0,
        )

        score = ScoreResult(
            score_total=float(comp_out.score),
            stage_label=str(stage_result.stage),
            score_stage=float(comp_out.subscores.get("growth_0_100", 0.0)),
            score_slope=float(comp_out.subscores.get("maturity_0_100", 0.0)),
            score_accel=float(comp_out.subscores.get("quality_0_100", 0.0)),
            fit_used=bool(gate_res.pass_fit),
        )

        score_row = {
            "asof": asof,
            "ticker": ticker,
            **_flatten_score(score),
            **_flatten_stage(stage_feats),
            "available_date": series.available_date_latest,
            "stage_confidence": float(stage_result.confidence),
            "fit_pass": int(gate_res.pass_fit),
            "chosen_model": str(fit.model),
            "score": float(score.score_total),
            "stage": str(stage_result.stage),
            "adv_usd": mkt_feats.get("adv_usd"),
            "vol": mkt_feats.get("vol"),
            "pe_ratio": feats.get("pe_ratio"),
            "gross_margin": feats.get("gross_margin"),
            "operating_margin": feats.get("operating_margin"),
            "fcf_yield": feats.get("fcf_yield"),
            "ev_to_fcf": feats.get("ev_to_fcf"),
            "country": feats.get("country"),
            "exchange": feats.get("exchange"),
            "industry_name": feats.get("industry_name"),
        }
        rows_scores.append(score_row)

        diag_row = {
            "asof": asof,
            "ticker": ticker,
            **_flatten_fit(fit),
            "gate_pass": int(gate_res.pass_fit),
            "gate_quality": float(gate_res.quality_score),
            "gate_reasons": "|".join(gate_res.reasons) if gate_res.reasons else "",
        }
        rows_diag.append(diag_row)

    scores_df = pd.DataFrame(rows_scores)
    diag_df = pd.DataFrame(rows_diag)

    if not scores_df.empty:
        scores_df = scores_df.sort_values(["score_total", "ticker"], ascending=[False, True]).reset_index(drop=True)
        scores_df["rank"] = range(1, len(scores_df) + 1)

    _write_csv(scores_df, Path(run_dir) / "scores.csv")
    _write_csv(diag_df, Path(run_dir) / "fit_diagnostics.csv")

    log_kv(
        logger,
        logging.INFO,
        "fit_summary",
        asof=asof,
        universe_size=len(universe),
        names_scored=int(scores_df["ticker"].nunique()) if not scores_df.empty else 0,
        fit_ok=int(n_fit_ok),
        fit_ok_rate=(n_fit_ok / n_total) if n_total else None,
        strict_parametric_only=strict_parametric_only,
    )

    policy_filtered_df = _apply_investability_policy(scores_df, cfg)
    scored_for_port = (
        policy_filtered_df[["ticker", "score", "adv_usd"]].copy()
        if not policy_filtered_df.empty
        else pd.DataFrame(columns=["ticker", "score", "adv_usd"])
    )

    top_pct = float(cfg.get("ranking", {}).get("top_percentile", 0.10))
    c_cfg = ConstructConfig(
        top_quantile=top_pct,
        top_n=max(1, int(len(scored_for_port) * top_pct)) if len(scored_for_port) > 0 else 1,
        min_adv_usd=float(cfg.get("data", {}).get("adv_dollar_min_usd", 0.0)) or None,
        col_score="score",
        position_cap=float(cfg.get("risk", {}).get("max_position_weight", 0.05)),
        sector_cap=float(cfg.get("risk", {}).get("sector_cap", 0.35)),
    )

    try:
        portfolio_df, _ = construct_portfolio(scored_for_port, config=c_cfg)
    except Exception as e:
        logger.warning("portfolio_construct_failed reason=%s", str(e))
        portfolio_df = pd.DataFrame(columns=["ticker", "weight", "score", "rank"])

    vol_map = {t: m.vol_60d_ann for t, m in mkt_map.items() if m is not None and m.vol_60d_ann is not None}
    risk_cfg = RiskConfig(
        target_vol=float(cfg.get("risk", {}).get("vol_target_annual", 0.18)) if bool(cfg.get("risk", {}).get("use_vol_targeting", False)) else None,
        gross_cap=1.0,
        renormalize=True,
    )
    if not portfolio_df.empty:
        portfolio_df, _ = apply_risk_overlays(portfolio_df, config=risk_cfg, vol=vol_map)

    _write_csv(portfolio_df, Path(run_dir) / "portfolio_weights.csv")

    summary = summarize_run(
        scores_df[["ticker", "score", "stage", "stage_confidence", "fit_pass", "chosen_model"]].copy()
        if not scores_df.empty
        else pd.DataFrame(columns=["ticker", "score", "stage", "stage_confidence", "fit_pass", "chosen_model"]),
        weights_df=portfolio_df,
        cfg=SummaryConfig(),
    )
    summary_dict = summary_to_dict(summary)
    if not scores_df.empty:
        top_score = float(scores_df["score"].max())
        top_tie_count = int((scores_df["score"] == top_score).sum())
        score_unique = int(scores_df["score"].nunique(dropna=True))
        stage_norm = scores_df["stage"].astype(str).str.lower()
        early_rate = float((stage_norm == "early").mean())
        growth_rate = float((stage_norm == "growth").mean())
        early_growth_rate = float(((stage_norm == "early") | (stage_norm == "growth")).mean())
        summary_dict["objective_diagnostics"] = {
            "fit_pass_rate": float(scores_df["fit_pass"].mean()) if "fit_pass" in scores_df.columns else None,
            "early_rate": early_rate,
            "growth_rate": growth_rate,
            "early_or_growth_rate": early_growth_rate,
            "score_unique_count": score_unique,
            "top_score": top_score,
            "top_score_tie_count": top_tie_count,
            "valuation_coverage_rate": float(scores_df["pe_ratio"].notna().mean()) if "pe_ratio" in scores_df.columns else None,
            "fcf_yield_mean": float(pd.to_numeric(scores_df["fcf_yield"], errors="coerce").mean()) if "fcf_yield" in scores_df.columns else None,
            "policy_input_names": int(len(scores_df)),
            "policy_output_names": int(len(policy_filtered_df)),
            "strict_parametric_only": bool(strict_parametric_only),
        }

    with (Path(run_dir) / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2)

    logger.info("run_complete asof=%s outputs=%s", asof, str(run_dir))


if __name__ == "__main__":
    main()
