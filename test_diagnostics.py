# tests/test_diagnostics.py
"""
Unit tests for scurve/fit/diagnostics.py

These tests validate:
- diagnostics metrics are computed and finite on simple synthetic data
- monotonicity_yhat behaves as expected
- bound flagging works when bounds are supplied
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from scurve.models.gompertz import GompertzCurve, GompertzParams
from scurve.models.bass import BassCurve, BassParams
from scurve.fit.diagnostics import DiagnosticsConfig, compute_diagnostics


def test_compute_diagnostics_gompertz_basic_metrics():
    true = GompertzParams(L=100.0, t0=10.0, k=0.25)
    curve = GompertzCurve(true)
    t = np.linspace(0, 25, 51)
    y = curve.predict(t)

    diag = compute_diagnostics(curve, t, y, cfg=DiagnosticsConfig(outlier_z=3.0))

    # Metrics existence
    for k in ("mse", "rmse", "mae", "r2", "rmse_norm_range", "monotonicity_yhat", "n_obs", "time_span"):
        assert k in diag.metrics

    assert diag.metrics["fit_ok"] == 1.0
    assert math.isfinite(diag.metrics["rmse"])
    assert diag.metrics["rmse"] < 1e-6  # perfect synthetic fit
    assert diag.metrics["r2"] >= 0.999999
    assert 0.95 <= diag.metrics["monotonicity_yhat"] <= 1.0

    # Flags
    assert diag.flags["curve_type"] == "GompertzCurve"
    assert diag.flags["has_nan_pred"] is False
    assert diag.flags["has_nan_resid"] is False
    assert 0.0 <= diag.flags["outlier_rate"] <= 1.0


def test_compute_diagnostics_bass_basic_metrics():
    true = BassParams(p=0.02, q=0.4, m=200.0)
    curve = BassCurve(true)
    t = np.linspace(0, 30, 61)
    y = curve.cumulative(t)

    diag = compute_diagnostics(curve, t, y)

    assert diag.metrics["fit_ok"] == 1.0
    assert diag.metrics["rmse"] < 1e-6
    assert diag.metrics["r2"] >= 0.999999
    assert 0.95 <= diag.metrics["monotonicity_yhat"] <= 1.0
    assert diag.flags["curve_type"] == "BassCurve"


def test_compute_diagnostics_flags_near_bound():
    """
    If parameter is near the provided bound, *_near_bound should be True.
    """
    p = GompertzParams(L=100.0, t0=10.0, k=0.25)
    curve = GompertzCurve(p)
    t = np.linspace(0, 25, 51)
    y = curve.predict(t)

    bounds = {"L": (100.0, 200.0), "t0": (0.0, 25.0), "k": (0.01, 1.0)}  # L exactly at lower bound
    diag = compute_diagnostics(curve, t, y, param_bounds=bounds, cfg=DiagnosticsConfig(bound_epsilon=1e-6))

    assert "L_near_bound" in diag.flags
    assert diag.flags["L_near_bound"] is True
    assert diag.flags["any_param_near_bound"] is True


def test_compute_diagnostics_handles_missing_data_gracefully():
    curve = GompertzCurve(GompertzParams(L=100.0, t0=10.0, k=0.25))
    t = np.array([0.0, 1.0, np.nan, 3.0, 4.0])
    y = np.array([0.0, 1.0, 2.0, np.nan, 5.0])

    diag = compute_diagnostics(curve, t, y)

    assert "n_obs" in diag.metrics
    # should drop NaNs and still compute
    assert diag.metrics["n_obs"] >= 2
    assert diag.flags["no_data"] is False


def test_compute_diagnostics_no_usable_data():
    curve = GompertzCurve(GompertzParams(L=100.0, t0=10.0, k=0.25))
    t = np.array([np.nan, np.nan])
    y = np.array([np.nan, np.nan])

    diag = compute_diagnostics(curve, t, y)
    assert diag.flags["no_data"] is True
    assert diag.metrics["fit_ok"] == 0.0