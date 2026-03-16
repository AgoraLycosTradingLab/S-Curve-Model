# tests/test_pipeline_smoke.py
"""
Smoke tests for an end-to-end S-Curve pipeline slice.

Goal
- Validate that the core building blocks interoperate:
    pre_fit -> fit_best_curve -> gate -> post_fit/fallback -> stage -> composite
    -> portfolio construct -> risk overlay -> summary

This is NOT a full backtest; it is a minimal integration test.
It uses synthetic curves (Gompertz and Bass) to produce stable behavior.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scurve.features.pre_fit import build_pre_fit_features
from scurve.features.post_fit import build_post_fit_features
from scurve.features.fallback import build_fallback_features
from scurve.fit.fitters import fit_best_curve, FittersConfig
from scurve.fit.gates import FitGate, GateConfig
from scurve.score.stage import StageScorer, StageConfig
from scurve.score.composite import CompositeScorer, CompositeConfig
from scurve.portfolio.construct import construct_portfolio, ConstructConfig
from scurve.portfolio.risk import apply_risk_overlays, RiskConfig
from scurve.report.summary import summarize_run, SummaryConfig

from scurve.models.gompertz import GompertzCurve, GompertzParams
from scurve.models.bass import BassCurve, BassParams


def _merge_dicts(*ds):
    out = {}
    for d in ds:
        if d:
            out.update(d)
    return out


def test_pipeline_smoke_runs_and_returns_outputs():
    # --- synthetic universe: 2 gompertz, 2 bass, 1 noisy/flat (forces fallback) ---
    t = np.linspace(0, 30, 61)

    g1 = GompertzCurve(GompertzParams(L=120.0, t0=10.0, k=0.25))
    g2 = GompertzCurve(GompertzParams(L=180.0, t0=14.0, k=0.20))
    b1 = BassCurve(BassParams(p=0.02, q=0.5, m=200.0))
    b2 = BassCurve(BassParams(p=0.015, q=0.4, m=260.0))

    y = {
        "G1": g1.predict(t),
        "G2": g2.predict(t),
        "B1": b1.cumulative(t),
        "B2": b2.cumulative(t),
        "FLAT": np.zeros_like(t) + 5.0,  # constant series => should fail pre-fit and use fallback
    }

    rows = []
    for ticker, series in y.items():
        # pre-fit
        pre = build_pre_fit_features(t, series)
        pre_feats = pre.features

        # fit
        fit_cfg = FittersConfig()  # defaults
        fit = fit_best_curve(t, series, cfg=fit_cfg)

        # gate
        gate = FitGate(GateConfig())
        gate_res = gate.evaluate(fit, pre_features=pre_feats)

        # features
        if gate_res.pass_fit and fit.curve is not None:
            post = build_post_fit_features(fit.curve, t, series)
            feats = _merge_dicts(pre_feats, post.features)
            fit_pass = 1
        else:
            fb = build_fallback_features(t, series)
            feats = _merge_dicts(pre_feats, fb.features)
            fit_pass = 0

        # stage + composite
        stage = StageScorer(StageConfig()).score(feats)
        comp = CompositeScorer(CompositeConfig()).score(feats, stage)

        rows.append(
            {
                "ticker": ticker,
                "score": comp.score,
                "stage": stage.stage,
                "stage_confidence": stage.confidence,
                "fit_pass": fit_pass,
                "chosen_model": fit.model,
            }
        )

    results_df = pd.DataFrame(rows)

    assert len(results_df) == 5
    assert results_df["score"].between(0, 100).all()
    assert results_df["stage"].notna().all()
    assert results_df["fit_pass"].isin([0, 1]).all()

    # portfolio construct
    wdf, pdiag = construct_portfolio(
        results_df.rename(columns={"ticker": "ticker", "score": "score"}),
        config=ConstructConfig(top_n=3, method="equal", position_cap=0.8, sector_cap=None),
    )
    assert len(wdf) == 3
    assert abs(wdf["weight"].sum() - 1.0) < 1e-9

    # risk overlay (no vol info, should skip vol target gracefully)
    wdf2, rdiag = apply_risk_overlays(
        wdf,
        config=RiskConfig(target_vol=None, gross_cap=1.0, renormalize=True),
        ticker_col="ticker",
        weight_col="weight",
    )
    assert abs(wdf2["weight"].sum() - 1.0) < 1e-9
    assert "vol_target_skipped" in rdiag and rdiag["vol_target_skipped"] is True

    # summary
    summary = summarize_run(results_df, weights_df=wdf2, cfg=SummaryConfig(top_k=3, bottom_k=3))
    assert summary.headline["n_assets"] == 5
    assert "score_mean" in summary.headline
    assert summary.portfolio.get("n_holdings", 0) == 3