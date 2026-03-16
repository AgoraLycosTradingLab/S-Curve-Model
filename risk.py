# scurve/portfolio/risk.py
"""
Risk controls for S-Curve portfolio outputs.

Purpose
- Provide lightweight, deterministic risk overlays that can be applied
  after initial portfolio construction:
    - volatility targeting (single scalar target vol)
    - exposure caps (gross / net)
    - turnover / weight change limits (basic)
    - optional risk scaling by instrument volatility

This module is intentionally not a full optimizer. It focuses on
transparent transformations on an existing weights vector.

Inputs
- weights DataFrame from scurve.portfolio.construct.construct_portfolio()
  expected columns: ticker, weight
- optional series/dicts for:
    - vol (annualized or per-period; consistent units required)
    - previous weights (for turnover / max change constraints)

Outputs
- updated weights DataFrame + diagnostics dict

Notes
- Vol targeting requires vol estimates; if missing, overlay is skipped safely.
- All operations are deterministic and stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd


def _to_series(x: Union[pd.Series, Dict[str, float], None], index: pd.Index) -> Optional[pd.Series]:
    if x is None:
        return None
    if isinstance(x, pd.Series):
        return x.reindex(index)
    if isinstance(x, dict):
        return pd.Series(x).reindex(index)
    raise TypeError("Expected vol/prev_weights as pd.Series, dict, or None.")


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    if not (np.isfinite(a) and np.isfinite(b)) or b == 0:
        return float(default)
    return float(a / b)


@dataclass
class RiskConfig:
    # Volatility targeting
    target_vol: Optional[float] = None  # if None, skip
    vol_col: str = "vol"  # if vol is provided in df
    min_scale: float = 0.0
    max_scale: float = 2.0

    # Exposure control (post-scale)
    gross_cap: Optional[float] = 1.0  # sum(|w|) cap, None disables
    net_cap: Optional[float] = None   # |sum(w)| cap, None disables

    # Turnover / weight change controls
    max_turnover: Optional[float] = None  # sum(|w_new - w_prev|) <= max_turnover
    max_weight_change: Optional[float] = None  # per-name cap on |delta w|

    # Weight floor / cleanup
    min_weight_abs: float = 0.0   # drop tiny weights to zero
    renormalize: bool = True      # renormalize after transformations


def estimate_portfolio_vol(
    weights: pd.Series,
    vol: pd.Series,
    *,
    corr: Optional[np.ndarray] = None,
) -> float:
    """
    Estimate portfolio volatility.

    If corr is None:
      assumes zero correlation and returns sqrt(sum((w*vol)^2))

    If corr is provided:
      uses vol vector and correlation matrix:
        cov = diag(vol) @ corr @ diag(vol)
        vol_p = sqrt(w^T cov w)

    vol units are whatever you provide (must be consistent with target_vol).
    """
    w = weights.to_numpy(dtype=float)
    s = vol.to_numpy(dtype=float)

    if corr is None:
        return float(np.sqrt(np.sum((w * s) ** 2)))

    corr = np.asarray(corr, dtype=float)
    if corr.shape != (len(w), len(w)):
        raise ValueError("corr must be square with shape (n,n) matching weights.")
    # Guard: ensure corr diagonal is 1-ish
    # (we don't force it; just proceed)
    cov = (s[:, None] * corr) * s[None, :]
    v = float(w.T @ cov @ w)
    return float(np.sqrt(max(v, 0.0)))


def apply_vol_target(
    weights: pd.Series,
    vol: pd.Series,
    target_vol: float,
    *,
    corr: Optional[np.ndarray] = None,
    min_scale: float = 0.0,
    max_scale: float = 2.0,
) -> Dict[str, Any]:
    """
    Returns dict with:
      - scaled_weights: pd.Series
      - scale: float
      - est_vol_before: float
      - est_vol_after: float
    """
    est_before = estimate_portfolio_vol(weights, vol, corr=corr)
    scale = _safe_div(float(target_vol), est_before, default=np.nan)
    if not np.isfinite(scale):
        scale = 1.0
    scale = _clip(scale, float(min_scale), float(max_scale))

    w2 = weights * scale
    est_after = estimate_portfolio_vol(w2, vol, corr=corr) if np.isfinite(est_before) else np.nan

    return {
        "scaled_weights": w2,
        "scale": float(scale),
        "est_vol_before": float(est_before),
        "est_vol_after": float(est_after),
    }


def apply_turnover_cap(
    w_new: pd.Series,
    w_prev: pd.Series,
    max_turnover: float,
) -> Dict[str, Any]:
    """
    Cap turnover by scaling changes toward previous weights.

    If turnover = sum(|w_new - w_prev|) > cap:
      w_adj = w_prev + alpha*(w_new - w_prev)
    where alpha = cap / turnover.

    Deterministic and convex.
    """
    d = w_new - w_prev
    turnover = float(np.sum(np.abs(d)))
    if turnover <= float(max_turnover) + 1e-12:
        return {"weights": w_new, "turnover": turnover, "alpha": 1.0, "capped": False}

    alpha = float(max_turnover) / max(1e-12, turnover)
    w_adj = w_prev + alpha * d
    return {"weights": w_adj, "turnover": turnover, "alpha": float(alpha), "capped": True}


def apply_max_weight_change(
    w_new: pd.Series,
    w_prev: pd.Series,
    max_change: float,
) -> Dict[str, Any]:
    """
    Cap per-name absolute weight change:
      w_adj_i = w_prev_i + clip(w_new_i - w_prev_i, -max_change, +max_change)
    """
    d = w_new - w_prev
    d_clip = d.clip(lower=-float(max_change), upper=float(max_change))
    w_adj = w_prev + d_clip
    changed = bool(np.any(np.abs(d) > float(max_change) + 1e-12))
    return {"weights": w_adj, "capped": changed}


def enforce_exposure_caps(
    w: pd.Series,
    *,
    gross_cap: Optional[float],
    net_cap: Optional[float],
) -> Dict[str, Any]:
    """
    Enforce gross and net exposure caps by uniform scaling.
    """
    w2 = w.copy()
    gross = float(np.sum(np.abs(w2)))
    net = float(np.sum(w2))

    scale = 1.0
    if gross_cap is not None and gross > float(gross_cap) + 1e-12:
        scale = min(scale, float(gross_cap) / max(1e-12, gross))
    if net_cap is not None and abs(net) > float(net_cap) + 1e-12:
        scale = min(scale, float(net_cap) / max(1e-12, abs(net)))

    if scale != 1.0:
        w2 = w2 * scale

    return {
        "weights": w2,
        "gross_before": gross,
        "net_before": net,
        "scale": float(scale),
        "gross_after": float(np.sum(np.abs(w2))),
        "net_after": float(np.sum(w2)),
    }


def apply_risk_overlays(
    weights_df: pd.DataFrame,
    *,
    config: Optional[RiskConfig] = None,
    vol: Union[pd.Series, Dict[str, float], None] = None,
    prev_weights: Union[pd.Series, Dict[str, float], None] = None,
    corr: Optional[np.ndarray] = None,
    ticker_col: str = "ticker",
    weight_col: str = "weight",
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply risk overlays to a weights DataFrame.

    Parameters
    ----------
    weights_df:
        DataFrame with at least ticker, weight.
    config:
        RiskConfig.
    vol:
        Vol estimates keyed by ticker (Series or dict). If None, will look for config.vol_col in df.
    prev_weights:
        Previous weights keyed by ticker (Series or dict) for turnover controls.
    corr:
        Optional correlation matrix aligned to the final ticker ordering.

    Returns
    -------
    (new_weights_df, diagnostics)
    """
    cfg = config or RiskConfig()
    if ticker_col not in weights_df.columns or weight_col not in weights_df.columns:
        raise ValueError(f"weights_df must contain '{ticker_col}' and '{weight_col}'.")

    df = weights_df.copy()
    df[ticker_col] = df[ticker_col].astype(str)
    df[weight_col] = pd.to_numeric(df[weight_col], errors="coerce").fillna(0.0)

    idx = df[ticker_col]
    w = pd.Series(df[weight_col].to_numpy(dtype=float), index=idx, dtype=float)

    diag: Dict[str, Any] = {"n": int(len(df))}

    # ---- Vol targeting ----
    vol_series = None
    if vol is not None:
        vol_series = _to_series(vol, idx)
    elif cfg.target_vol is not None and cfg.vol_col in df.columns:
        vol_series = pd.Series(pd.to_numeric(df[cfg.vol_col], errors="coerce").to_numpy(dtype=float), index=idx)

    if cfg.target_vol is not None and vol_series is not None:
        vol_series = vol_series.fillna(np.nan)
        # If too many missing, skip
        if np.isfinite(vol_series).sum() >= max(3, int(0.6 * len(vol_series))):
            # fill missing with median for stability
            med = float(np.nanmedian(vol_series.to_numpy(dtype=float)))
            if not np.isfinite(med) or med <= 0:
                diag["vol_target_skipped"] = True
            else:
                vol_used = vol_series.fillna(med).clip(lower=1e-12)
                out = apply_vol_target(
                    w,
                    vol_used,
                    float(cfg.target_vol),
                    corr=corr,
                    min_scale=float(cfg.min_scale),
                    max_scale=float(cfg.max_scale),
                )
                w = out["scaled_weights"]
                diag["vol_target"] = {
                    "target_vol": float(cfg.target_vol),
                    "scale": out["scale"],
                    "est_vol_before": out["est_vol_before"],
                    "est_vol_after": out["est_vol_after"],
                }
        else:
            diag["vol_target_skipped"] = True
    else:
        diag["vol_target_skipped"] = True

    # ---- Turnover controls ----
    prev = _to_series(prev_weights, idx) if prev_weights is not None else None
    if prev is not None:
        prev = prev.fillna(0.0).astype(float)

        if cfg.max_weight_change is not None:
            out = apply_max_weight_change(w, prev, float(cfg.max_weight_change))
            w = out["weights"]
            diag["max_weight_change"] = {"cap": float(cfg.max_weight_change), "capped": out["capped"]}

        if cfg.max_turnover is not None:
            out = apply_turnover_cap(w, prev, float(cfg.max_turnover))
            w = out["weights"]
            diag["turnover"] = {
                "cap": float(cfg.max_turnover),
                "turnover_before": out["turnover"],
                "alpha": out["alpha"],
                "capped": out["capped"],
            }

    # ---- Exposure caps ----
    exp = enforce_exposure_caps(w, gross_cap=cfg.gross_cap, net_cap=cfg.net_cap)
    w = exp["weights"]
    diag["exposure_caps"] = exp

    # ---- Drop tiny weights ----
    if cfg.min_weight_abs is not None and float(cfg.min_weight_abs) > 0:
        before_nnz = int((np.abs(w) > 0).sum())
        w = w.where(np.abs(w) >= float(cfg.min_weight_abs), 0.0)
        after_nnz = int((np.abs(w) > 0).sum())
        diag["min_weight_abs"] = {"threshold": float(cfg.min_weight_abs), "nnz_before": before_nnz, "nnz_after": after_nnz}

    # ---- Renormalize (common for long-only) ----
    if cfg.renormalize:
        s = float(w.sum())
        if s > 0:
            w = w / s
        diag["renormalized"] = True
        diag["weight_sum_after"] = float(w.sum())
    else:
        diag["renormalized"] = False
        diag["weight_sum_after"] = float(w.sum())

    df[weight_col] = w.values
    return df, diag