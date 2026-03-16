# scurve/core/types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Literal

ModelFamily = Literal["logistic", "gompertz"]

@dataclass(frozen=True)
class FitResult:
    ok: bool
    family: ModelFamily
    params: Dict[str, float]          # K, r, t0
    nrmse: float
    sse: float
    k_on_upper_bound: bool
    message: str = ""

@dataclass(frozen=True)
class StageFeatures:
    maturity_ratio: float
    normalized_slope: float
    curvature: float
    accel_flag: int                   # 1 if curvature > 0 else 0

@dataclass(frozen=True)
class ScoreResult:
    score_total: float                # 0..100
    stage_label: str                  # Early/Expansion/Mature/Unknown
    score_stage: float
    score_slope: float
    score_accel: float
    fit_used: bool                    # True if parametric features used

@dataclass(frozen=True)
class PipelineRow:
    ticker: str
    asof: str
    available_date: str
    fit: Optional[FitResult]
    stage: Optional[StageFeatures]
    score: ScoreResult