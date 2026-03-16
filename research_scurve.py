# research_scurve.py
"""
Interactive Research Mode runner for the S-Curve model.

Where to place:
- Put this file in the PROJECT ROOT (same level as pyproject.toml), not inside the scurve/ package.

What it does:
- Runs a single time series through:
    pre_fit -> fit_best_curve -> gate -> post_fit/fallback -> stage -> composite
- Prints key artifacts for inspection.

How to run:
    python research_scurve.py

How to adapt to real data:
- Replace the synthetic series in `load_example_series()` with your own loader.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Dict, Tuple, Optional

import numpy as np

from scurve import (
    build_pre_fit_features,
    build_post_fit_features,
    build_fallback_features,
    fit_best_curve,
    FitGate,
    GateConfig,
    StageScorer,
    StageConfig,
    CompositeScorer,
    CompositeConfig,
)


def load_example_series(kind: str = "gompertz") -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    """
    Returns (t, y, meta) for a synthetic example.

    kind:
      - "gompertz": clean S-curve
      - "bass": diffusion cumulative
      - "flat": constant series (forces fallback)
      - "noisy": noisy increasing series
    """
    t = np.linspace(0.0, 30.0, 61)

    if kind == "gompertz":
        # y = L * exp(-exp(-k*(t-t0)))
        L, t0, k = 150.0, 12.0, 0.20
        y = L * np.exp(-np.exp(-k * (t - t0)))
        meta = {"id": "SYN_GOMPERTZ", "kind": kind}

    elif kind == "bass":
        # simple Bass-like cumulative proxy (not using fitter internals here)
        # create a smooth cumulative S-curve using logistic as a proxy then scale
        m, t0, k = 240.0, 12.0, 0.35
        y = m / (1.0 + np.exp(-k * (t - t0)))
        meta = {"id": "SYN_BASS_PROXY", "kind": kind}

    elif kind == "flat":
        y = np.zeros_like(t) + 5.0
        meta = {"id": "SYN_FLAT", "kind": kind}

    elif kind == "noisy":
        rng = np.random.default_rng(123)  # deterministic noise
        base = 120.0 * np.exp(-np.exp(-0.22 * (t - 11.0)))
        noise = rng.normal(0.0, 1.0, size=t.shape)
        y = np.maximum.accumulate(base + noise)  # keep mostly non-decreasing
        meta = {"id": "SYN_NOISY", "kind": kind}

    else:
        raise ValueError("kind must be one of: gompertz, bass, flat, noisy")

    return t, y, meta


def run_single_series(
    t: np.ndarray,
    y: np.ndarray,
    *,
    gate_cfg: Optional[GateConfig] = None,
    stage_cfg: Optional[StageConfig] = None,
    comp_cfg: Optional[CompositeConfig] = None,
) -> Dict[str, object]:
    """
    Runs the full research pipeline on a single series and returns a dict
    of key artifacts for interactive inspection.
    """
    # 1) Pre-fit
    pre = build_pre_fit_features(t, y)

    # 2) Fit
    fit = fit_best_curve(t, y)

    # 3) Gate
    gate = FitGate(gate_cfg or GateConfig())
    gate_res = gate.evaluate(fit, pre_features=pre.features)

    # 4) Features
    if gate_res.pass_fit and fit.curve is not None:
        post = build_post_fit_features(fit.curve, t, y)
        feats = {**pre.features, **post.features}
        fit_pass = True
        feature_mode = "post_fit"
    else:
        fb = build_fallback_features(t, y)
        feats = {**pre.features, **fb.features}
        fit_pass = False
        feature_mode = "fallback"

    # 5) Stage
    stage = StageScorer(stage_cfg or StageConfig()).score(feats)

    # 6) Composite
    comp = CompositeScorer(comp_cfg or CompositeConfig()).score(feats, stage)

    return {
        "fit_pass": fit_pass,
        "feature_mode": feature_mode,
        "pre_features": pre.features,
        "fit_model": fit.model,
        "fit_reason": fit.reason,
        "fit_params": fit.params,
        "fit_metrics": (fit.diagnostics.metrics if fit.diagnostics is not None else None),
        "fit_flags": (fit.diagnostics.flags if fit.diagnostics is not None else None),
        "gate": asdict(gate_res),
        "stage": {
            "stage": stage.stage,
            "confidence": stage.confidence,
            "components": stage.components,
            "diagnostics": stage.diagnostics,
        },
        "composite": {
            "score": comp.score,
            "subscores": comp.subscores,
            "flags": comp.flags,
            "diagnostics": comp.diagnostics,
        },
    }


def _print_block(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> None:
    # Choose one: "gompertz" | "bass" | "flat" | "noisy"
    kind = "gompertz"

    t, y, meta = load_example_series(kind=kind)

    _print_block("S-CURVE INTERACTIVE RESEARCH MODE")
    print(f"Series: {meta['id']} (kind={meta['kind']})")
    print(f"n={len(t)}  t_span=[{t[0]:.2f},{t[-1]:.2f}]")

    out = run_single_series(t, y)

    _print_block("FIT")
    print("fit_pass:", out["fit_pass"])
    print("feature_mode:", out["feature_mode"])
    print("fit_model:", out["fit_model"])
    print("fit_reason:", out["fit_reason"])
    print("fit_params:", out["fit_params"])

    _print_block("DIAGNOSTICS (metrics)")
    fm = out["fit_metrics"] or {}
    # print a compact subset
    keys = ["r2", "rmse", "rmse_norm_range", "rmse_norm_scale", "monotonicity_yhat", "n_obs", "time_span"]
    for k in keys:
        if k in fm:
            print(f"{k:>18}: {fm[k]}")

    _print_block("GATE")
    gate = out["gate"]
    print("pass_fit:", gate["pass_fit"])
    print("quality_score:", gate["quality_score"])
    print("reasons:", gate["reasons"])

    _print_block("STAGE")
    st = out["stage"]
    print("stage:", st["stage"])
    print("confidence:", st["confidence"])
    print("components:", st["components"])

    _print_block("COMPOSITE SCORE")
    cs = out["composite"]
    print("score:", cs["score"])
    print("subscores:", cs["subscores"])
    print("flags:", cs["flags"])

    _print_block("NEXT STEPS")
    print("1) Replace load_example_series() with your real series loader.")
    print("2) Change `kind` in main() or wire it to CLI args.")
    print("3) Inspect out['fit_metrics'], out['stage']['diagnostics'], out['composite']['diagnostics'].")


if __name__ == "__main__":
    main()