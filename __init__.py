# scurve/__init__.py
"""
S-Curve Model Package
=====================

Agora Lycos Trading Lab — S-Curve Research Framework

This package implements a full S-curve lifecycle pipeline:

    data → fit → diagnostics → gates → features → stage → composite score
        → portfolio construction → risk overlays → reporting

Core components
---------------

Fit Layer
- scurve.fit.fitters.fit_best_curve
- scurve.fit.gates.FitGate
- scurve.fit.diagnostics.compute_diagnostics

Feature Layer
- scurve.features.pre_fit.build_pre_fit_features
- scurve.features.post_fit.build_post_fit_features
- scurve.features.fallback.build_fallback_features

Scoring Layer
- scurve.score.stage.StageScorer
- scurve.score.composite.CompositeScorer

Portfolio Layer
- scurve.portfolio.construct.construct_portfolio
- scurve.portfolio.risk.apply_risk_overlays

Reporting
- scurve.report.summary.summarize_run
- scurve.report.drift.compare_snapshots

Design Philosophy
-----------------
- Deterministic
- Dependency-light
- Transparent thresholds
- Modular and replaceable

All submodules are safe to import independently.
"""

__version__ = "0.1.0"

# ---- Fit Layer ----
from scurve.fit.fitters import (
    FitResult,
    FittersConfig,
    fit_gompertz,
    fit_bass,
    fit_best_curve,
)

from scurve.fit.gates import (
    FitGate,
    GateConfig,
    GateResult,
)

from scurve.fit.diagnostics import (
    FitDiagnostics,
    DiagnosticsConfig,
    compute_diagnostics,
)

# ---- Feature Layer ----
from scurve.features.pre_fit import (
    PreFitConfig,
    PreFitResult,
    build_pre_fit_features,
)

from scurve.features.post_fit import (
    PostFitResult,
    build_post_fit_features,
)

from scurve.features.fallback import (
    FallbackConfig,
    FallbackResult,
    build_fallback_features,
)

# ---- Scoring Layer ----
from scurve.score.stage import (
    StageScorer,
    StageConfig,
    StageResult,
)

from scurve.score.composite import (
    CompositeScorer,
    CompositeConfig,
    CompositeScoreResult,
)

# ---- Portfolio Layer ----
from scurve.portfolio.construct import (
    ConstructConfig,
    construct_portfolio,
)

from scurve.portfolio.risk import (
    RiskConfig,
    apply_risk_overlays,
)

# ---- Reporting ----
from scurve.report.summary import (
    RunSummary,
    SummaryConfig,
    summarize_run,
    summary_to_dict,
)

from scurve.report.drift import (
    DriftConfig,
    DriftReport,
    make_run_snapshot,
    compare_snapshots,
)

__all__ = [
    # version
    "__version__",

    # fit
    "FitResult",
    "FittersConfig",
    "fit_gompertz",
    "fit_bass",
    "fit_best_curve",
    "FitGate",
    "GateConfig",
    "GateResult",
    "FitDiagnostics",
    "DiagnosticsConfig",
    "compute_diagnostics",

    # features
    "PreFitConfig",
    "PreFitResult",
    "build_pre_fit_features",
    "PostFitResult",
    "build_post_fit_features",
    "FallbackConfig",
    "FallbackResult",
    "build_fallback_features",

    # scoring
    "StageScorer",
    "StageConfig",
    "StageResult",
    "CompositeScorer",
    "CompositeConfig",
    "CompositeScoreResult",

    # portfolio
    "ConstructConfig",
    "construct_portfolio",
    "RiskConfig",
    "apply_risk_overlays",

    # reporting
    "RunSummary",
    "SummaryConfig",
    "summarize_run",
    "summary_to_dict",
    "DriftConfig",
    "DriftReport",
    "make_run_snapshot",
    "compare_snapshots",
]