# tests/test_models.py
"""
Unit tests for scurve/models.

These tests are designed to be:
- deterministic
- lightweight (no external data)
- robust to minor numerical differences

They validate:
- basic monotonic properties of Gompertz/Bass cumulative curves
- parameter validation
- fitter "works" on synthetic data (recover roughly correct params)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scurve.models.gompertz import GompertzCurve, GompertzParams, GompertzFitter, GompertzFitConfig
from scurve.models.bass import BassCurve, BassParams, BassFitter, BassFitConfig


def test_gompertz_params_validation():
    with pytest.raises(ValueError):
        GompertzParams(L=-1.0, t0=10.0, k=0.5).validate()
    with pytest.raises(ValueError):
        GompertzParams(L=100.0, t0=10.0, k=0.0).validate()
    # ok
    GompertzParams(L=100.0, t0=10.0, k=0.2).validate()


def test_bass_params_validation():
    with pytest.raises(ValueError):
        BassParams(p=0.0, q=0.5, m=100.0).validate()
    with pytest.raises(ValueError):
        BassParams(p=0.01, q=-0.1, m=100.0).validate()
    with pytest.raises(ValueError):
        BassParams(p=0.01, q=0.1, m=0.0).validate()
    # ok
    BassParams(p=0.01, q=0.5, m=100.0).validate()


def test_gompertz_curve_monotone_increasing():
    curve = GompertzCurve(GompertzParams(L=100.0, t0=10.0, k=0.25))
    t = np.linspace(0, 30, 301)
    y = curve.predict(t)
    # non-decreasing
    dy = np.diff(y)
    assert np.all(dy >= -1e-10)
    # bounded by [0, L]
    assert np.min(y) >= -1e-6
    assert np.max(y) <= 100.0 + 1e-6


def test_bass_curve_monotone_increasing():
    curve = BassCurve(BassParams(p=0.02, q=0.4, m=200.0))
    t = np.linspace(0, 30, 301)
    y = curve.cumulative(t)
    dy = np.diff(y)
    assert np.all(dy >= -1e-10)
    assert np.min(y) >= -1e-6
    assert np.max(y) <= 200.0 + 1e-6


def test_bass_rate_nonnegative():
    curve = BassCurve(BassParams(p=0.02, q=0.6, m=200.0))
    t = np.linspace(0, 30, 301)
    r = curve.rate(t)
    assert np.min(r) >= -1e-10


def test_gompertz_fitter_recovers_synthetic_params_roughly():
    """
    Generate synthetic Gompertz data (no noise) and confirm fit is close.
    """
    true = GompertzParams(L=150.0, t0=12.0, k=0.18)
    curve = GompertzCurve(true)
    t = np.linspace(0, 30, 61)
    y = curve.predict(t)

    cfg = GompertzFitConfig(
        grid_sizes=(18, 18, 18),
        refine_steps=5,
        refine_shrink=0.60,
    )
    fitter = GompertzFitter(cfg)
    fit_curve, info = fitter.fit(
        t,
        y,
        L_bounds=(50.0, 300.0),
        t0_bounds=(0.0, 30.0),
        k_bounds=(0.01, 1.0),
    )
    p = fit_curve.params

    assert abs(p.L - true.L) / true.L < 0.10
    assert abs(p.t0 - true.t0) / true.t0 < 0.15
    assert abs(p.k - true.k) / true.k < 0.25
    assert math.isfinite(info["loss"])


def test_bass_fitter_recovers_synthetic_params_roughly():
    """
    Generate synthetic Bass cumulative data (no noise) and confirm fit is close.
    Bass fits can be less stable than Gompertz, so tolerances are looser.
    """
    true = BassParams(p=0.015, q=0.45, m=500.0)
    curve = BassCurve(true)
    t = np.linspace(0, 40, 81)
    y = curve.cumulative(t)

    cfg = BassFitConfig(
        fit_to="cumulative",
        grid_sizes=(18, 22, 16),
        refine_steps=5,
        refine_shrink=0.60,
        p_bounds=(1e-4, 0.2),
        q_bounds=(0.0, 2.0),
        m_bounds_mult=(0.8, 1.25),
    )
    fitter = BassFitter(cfg)

    fit_curve, info = fitter.fit(
        t,
        y,
        fit_to="cumulative",
        p_bounds=(1e-4, 0.2),
        q_bounds=(0.0, 2.0),
        m_bounds=(300.0, 900.0),
    )

    p = fit_curve.params
    assert abs(p.m - true.m) / true.m < 0.15
    # p and q can be correlated; just check they're in the right ballpark
    assert abs(p.p - true.p) / true.p < 0.50
    assert abs(p.q - true.q) / true.q < 0.50
    assert math.isfinite(info["loss"])