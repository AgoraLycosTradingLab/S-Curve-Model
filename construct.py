# scurve/portfolio/construct.py
"""
Portfolio construction for the S-Curve model.

Purpose
- Convert scored universe into a deterministic portfolio:
  - select top names by score (or top quantile)
  - apply liquidity filters (already expected upstream, but can re-check)
  - apply caps (position cap, sector cap)
  - optional volatility targeting (stub hook; full vol targeting lives in portfolio/risk.py)

Design goals
- Deterministic and transparent (no optimizer required).
- Works on a pandas DataFrame containing at least:
    - ticker (or id)
    - score (0..100)
    - optional: sector, adv (avg daily $ volume), price, vol, etc.

Outputs
- weights DataFrame with columns:
    ticker, weight, score, sector, rank
- diagnostics dict describing selection and any caps that bound.

Notes
- This is a simple constructor suitable for research and walk-forward backtests.
- For production-grade constraints (turnover, cost-aware, etc.), extend later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns


def _clip(x: float, lo: float, hi: float) -> float:
    return float(np.clip(float(x), float(lo), float(hi)))


@dataclass
class ConstructConfig:
    # Selection
    top_n: int = 25
    top_quantile: Optional[float] = None  # e.g. 0.10 means top decile; overrides top_n if provided

    # Column names
    col_ticker: str = "ticker"
    col_score: str = "score"
    col_sector: str = "sector"

    # Liquidity filter (optional)
    min_adv_usd: Optional[float] = None
    col_adv_usd: str = "adv_usd"

    # Weighting
    method: str = "equal"  # "equal" or "score"
    score_floor: float = 0.0  # for score weighting, clip negatives to this
    score_power: float = 1.0  # score^power for score weighting

    # Caps
    position_cap: float = 0.10  # max single name weight
    sector_cap: Optional[float] = 0.30  # max sector weight, None disables

    # Minimum holdings
    min_holdings: int = 5


def _apply_position_cap(w: pd.Series, cap: float) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Iteratively enforce position caps by redistributing excess pro-rata to uncapped names.
    Deterministic because it uses stable ordering.
    """
    cap = float(cap)
    diag: Dict[str, Any] = {"position_cap": cap, "bound_names": []}

    if cap <= 0:
        # degenerate
        w[:] = 0.0
        return w, diag

    # Loop until no cap violations or no recipients
    for _ in range(50):
        over = w[w > cap]
        if over.empty:
            break

        # amount to redistribute
        excess = float((over - cap).sum())
        w.loc[over.index] = cap
        diag["bound_names"] = list(sorted(set(diag["bound_names"] + list(over.index))))

        recipients = w[w < cap - 1e-12]
        if recipients.empty:
            break

        # redistribute excess proportional to current recipient weights
        total_rec = float(recipients.sum())
        if total_rec <= 0:
            # if all recipients are 0, distribute equally among them
            add_each = excess / float(len(recipients))
            w.loc[recipients.index] = w.loc[recipients.index] + add_each
        else:
            w.loc[recipients.index] = w.loc[recipients.index] + excess * (recipients / total_rec)

        # normalize to 1 (numerical drift)
        s = float(w.sum())
        if s > 0:
            w = w / s

    return w, diag


def _apply_sector_cap(
    df: pd.DataFrame, w: pd.Series, sector_col: str, cap: float
) -> Tuple[pd.Series, Dict[str, Any]]:
    """
    Enforce sector caps by trimming sectors above cap and redistributing to others.
    Deterministic, iterative, pro-rata within/among sectors.
    """
    cap = float(cap)
    diag: Dict[str, Any] = {"sector_cap": cap, "bound_sectors": []}

    if cap <= 0:
        w[:] = 0.0
        return w, diag

    sectors = df[sector_col].fillna("UNKNOWN").astype(str)

    for _ in range(50):
        sector_w = w.groupby(sectors).sum()
        over = sector_w[sector_w > cap + 1e-12]
        if over.empty:
            break

        excess_total = 0.0
        for sec, sec_w in over.items():
            diag["bound_sectors"].append(sec)
            idx = sectors[sectors == sec].index
            # trim within sector proportionally
            if sec_w <= 0:
                continue
            trim_ratio = cap / float(sec_w)
            before = float(w.loc[idx].sum())
            w.loc[idx] = w.loc[idx] * trim_ratio
            after = float(w.loc[idx].sum())
            excess_total += max(0.0, before - after)

        recipients = sector_w[sector_w < cap - 1e-12]
        if recipients.empty or excess_total <= 0:
            break

        # distribute across sectors proportional to their current weight (or equally if zero)
        rec_sector_w = w.groupby(sectors).sum()
        rec_sector_w = rec_sector_w.loc[recipients.index]
        total_rec = float(rec_sector_w.sum())
        if total_rec <= 0:
            # equal across recipient sectors, then within sector proportional
            add_per_sec = excess_total / float(len(rec_sector_w))
            for sec in rec_sector_w.index:
                idx = sectors[sectors == sec].index
                sw = float(w.loc[idx].sum())
                if sw <= 0:
                    # equal within sector if all zero
                    w.loc[idx] = w.loc[idx] + add_per_sec / float(len(idx))
                else:
                    w.loc[idx] = w.loc[idx] + add_per_sec * (w.loc[idx] / sw)
        else:
            for sec in rec_sector_w.index:
                idx = sectors[sectors == sec].index
                sec_add = excess_total * (float(rec_sector_w.loc[sec]) / total_rec)
                sw = float(w.loc[idx].sum())
                if sw <= 0:
                    w.loc[idx] = w.loc[idx] + sec_add / float(len(idx))
                else:
                    w.loc[idx] = w.loc[idx] + sec_add * (w.loc[idx] / sw)

        # renormalize
        s = float(w.sum())
        if s > 0:
            w = w / s

    diag["bound_sectors"] = list(sorted(set(diag["bound_sectors"])))
    return w, diag


def construct_portfolio(
    scored: pd.DataFrame,
    *,
    config: Optional[ConstructConfig] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Build a portfolio from a scored universe.

    Parameters
    ----------
    scored:
        DataFrame with at least [ticker, score].
        Optional: sector, adv_usd.
    config:
        ConstructConfig.

    Returns
    -------
    (weights_df, diagnostics)
    """
    cfg = config or ConstructConfig()

    if not isinstance(scored, pd.DataFrame):
        raise TypeError("scored must be a pandas DataFrame.")

    if not _col(scored, cfg.col_ticker):
        raise ValueError(f"Missing required column '{cfg.col_ticker}'.")
    if not _col(scored, cfg.col_score):
        raise ValueError(f"Missing required column '{cfg.col_score}'.")

    df = scored.copy()

    # Clean essentials
    df[cfg.col_ticker] = df[cfg.col_ticker].astype(str)
    df[cfg.col_score] = pd.to_numeric(df[cfg.col_score], errors="coerce")

    # Liquidity filter (optional)
    diag: Dict[str, Any] = {"n_in": int(len(df))}
    if cfg.min_adv_usd is not None:
        if not _col(df, cfg.col_adv_usd):
            raise ValueError(f"min_adv_usd set but missing column '{cfg.col_adv_usd}'.")
        df[cfg.col_adv_usd] = pd.to_numeric(df[cfg.col_adv_usd], errors="coerce")
        before = len(df)
        df = df[df[cfg.col_adv_usd] >= float(cfg.min_adv_usd)].copy()
        diag["liquidity_filter_min_adv_usd"] = float(cfg.min_adv_usd)
        diag["n_after_liquidity"] = int(len(df))
        diag["dropped_liquidity"] = int(before - len(df))

    # Drop missing score/ticker
    df = df.dropna(subset=[cfg.col_ticker, cfg.col_score]).copy()

    # Sort by score desc, stable tie-breaker by ticker
    df = df.sort_values([cfg.col_score, cfg.col_ticker], ascending=[False, True]).reset_index(drop=True)

    # Selection
    if cfg.top_quantile is not None:
        q = float(cfg.top_quantile)
        if not (0.0 < q <= 1.0):
            raise ValueError("top_quantile must be in (0,1].")
        n_sel = max(cfg.min_holdings, int(np.ceil(len(df) * q)))
    else:
        n_sel = max(cfg.min_holdings, int(cfg.top_n))

    selected = df.head(n_sel).copy()
    diag["n_selected"] = int(len(selected))
    if len(selected) < cfg.min_holdings:
        diag["warning"] = f"Selected {len(selected)} < min_holdings={cfg.min_holdings}."

    # Weighting
    method = (cfg.method or "equal").lower().strip()
    if method not in {"equal", "score"}:
        raise ValueError("method must be 'equal' or 'score'.")

    if method == "equal":
        w = np.full(len(selected), 1.0 / max(1, len(selected)), dtype=float)
    else:
        s = selected[cfg.col_score].to_numpy(dtype=float)
        s = np.maximum(s, float(cfg.score_floor))
        s = np.power(s, float(cfg.score_power))
        tot = float(np.sum(s))
        if tot <= 0:
            w = np.full(len(selected), 1.0 / max(1, len(selected)), dtype=float)
        else:
            w = s / tot

    w = pd.Series(w, index=selected.index, dtype=float)

    # Apply caps
    pos_cap_diag = {}
    sec_cap_diag = {}
    if cfg.position_cap is not None:
        w, pos_cap_diag = _apply_position_cap(w, float(cfg.position_cap))

    if cfg.sector_cap is not None:
        if _col(selected, cfg.col_sector):
            w, sec_cap_diag = _apply_sector_cap(selected, w, cfg.col_sector, float(cfg.sector_cap))
        else:
            diag["sector_cap_skipped"] = True

    # Final normalization
    s = float(w.sum())
    if s > 0:
        w = w / s
    else:
        w[:] = 0.0

    selected["weight"] = w.values
    selected["rank"] = np.arange(1, len(selected) + 1, dtype=int)

    # Output columns
    out_cols = [cfg.col_ticker, "weight", cfg.col_score, "rank"]
    if _col(selected, cfg.col_sector):
        out_cols.insert(3, cfg.col_sector)
    if cfg.min_adv_usd is not None and _col(selected, cfg.col_adv_usd):
        out_cols.append(cfg.col_adv_usd)

    weights_df = selected[out_cols].copy()

    diag.update(
        {
            "weighting_method": method,
            "position_cap": float(cfg.position_cap) if cfg.position_cap is not None else None,
            "sector_cap": float(cfg.sector_cap) if cfg.sector_cap is not None else None,
            "position_cap_diag": pos_cap_diag,
            "sector_cap_diag": sec_cap_diag,
            "weight_sum": float(weights_df["weight"].sum()),
        }
    )

    return weights_df, diag