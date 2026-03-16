# tests/test_scoring.py
"""
Unit tests for scurve/score (stage + composite).

These tests validate:
- stage scorer returns a valid label and bounded confidence
- composite scorer returns a 0..100 score and subscores
- behavior is stable on simple synthetic feature sets (fit-based and fallback)
"""

from __future__ import annotations

import numpy as np

from scurve.score.stage import StageScorer, StageConfig
from scurve.score.composite import CompositeScorer, CompositeConfig
from scurve.features.pre_fit import build_pre_fit_features
from scurve.features.fallback import build_fallback_features
from scurve.models.gompertz import GompertzCurve, GompertzParams
from scurve.features.post_fit import build_post_fit_features


def test_stage_scorer_outputs_valid_stage_and_confidence():
    scorer = StageScorer(StageConfig())

    # Minimal feature set (fallback-like)
    feats = {
        "pre_monotonicity": 0.9,
        "pre_slope": 1.0,
        "pre_time_span": 10.0,
        "fb_max_drawdown": -0.05,
    }
    res = scorer.score(feats)
    assert isinstance(res.stage, str)
    assert res.stage in {"early", "growth", "mature", "decline", "unknown"}
    assert 0.0 <= res.confidence <= 1.0
    assert isinstance(res.components, dict)
    assert "maturity" in res.components


def test_stage_decline_detects_negative_trend():
    scorer = StageScorer(StageConfig())
    feats = {
        "pre_slope": -1.0,
        "pre_monotonicity": 0.6,
        "fb_max_drawdown": -0.30,  # deep dd
    }
    res = scorer.score(feats)
    assert res.stage == "decline"
    assert 0.0 <= res.confidence <= 1.0


def test_composite_score_bounded_and_subscores_present():
    stage_scorer = StageScorer(StageConfig())
    comp_scorer = CompositeScorer(CompositeConfig())

    feats = {
        "progress_end": 0.5,
        "r2": 0.8,
        "rmse_norm_range": 0.2,
        "pre_monotonicity": 0.9,
        "pre_n_obs": 20.0,
        "pre_time_span": 10.0,
        "peak_slope": 5.0,
        "L": 100.0,
    }
    stage = stage_scorer.score(feats)
    out = comp_scorer.score(feats, stage)

    assert 0.0 <= out.score <= 100.0
    assert "growth_0_100" in out.subscores
    assert "maturity_0_100" in out.subscores
    assert "quality_0_100" in out.subscores
    assert "risk_0_100" in out.subscores
    assert isinstance(out.flags, dict)
    assert out.flags["stage"] == stage.stage


def test_scoring_on_realistic_synthetic_gompertz_features():
    """
    Build features from a fitted Gompertz curve and ensure scoring is stable.
    """
    t = np.linspace(0, 30, 61)
    curve = GompertzCurve(GompertzParams(L=150.0, t0=12.0, k=0.2))
    y = curve.predict(t)

    pre = build_pre_fit_features(t, y)
    post = build_post_fit_features(curve, t, y)

    feats = {}
    feats.update(pre.features)
    feats.update(post.features)

    stage = StageScorer(StageConfig()).score(feats)
    comp = CompositeScorer(CompositeConfig()).score(feats, stage)

    assert stage.stage in {"early", "growth", "mature", "decline", "unknown"}
    assert 0.0 <= stage.confidence <= 1.0
    assert 0.0 <= comp.score <= 100.0


def test_scoring_on_fallback_features_runs():
    """
    If fit fails / is skipped, fallback features should still score.
    """
    t = np.linspace(0, 10, 11)
    y = np.ones_like(t) * 5.0  # constant series

    pre = build_pre_fit_features(t, y)
    fb = build_fallback_features(t, y)

    feats = {}
    feats.update(pre.features)
    feats.update(fb.features)

    stage = StageScorer(StageConfig()).score(feats)
    comp = CompositeScorer(CompositeConfig()).score(feats, stage)

    assert 0.0 <= stage.confidence <= 1.0
    assert 0.0 <= comp.score <= 100.0