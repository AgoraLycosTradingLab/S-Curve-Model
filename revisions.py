"""
scurve.data.revisions

Analyst revisions (consensus EPS) adapter + point-in-time snapshot overlays
for Agora Lycos S-Curve model.

Purpose
-------
Provide an optional overlay signal: "earnings revision momentum".
This is not required for the model to run, but if present it can be used in:
- scoring (confirmation signal)
- validation (separate the curve-fit signal from Street expectations)

Adapter approach
---------------
This module is adapter-friendly. Current implementation supports a CSV file.

Expected CSV columns (case-insensitive mapping supported):
  - date            (the date the consensus value is observed)
  - ticker
  - eps_consensus   (consensus EPS; numeric)

Optional:
  - fy              (fiscal year label)
  - period          (e.g., "FY1", "NTM", etc.)

PIT policy
---------
For revisions, the "date" is assumed to be the availability date (i.e., you only
use rows with date <= asof). If you want an additional lag, set:
  cfg['data']['revisions_lag_days']

Outputs
-------
load_revisions_overlays(...) returns dict[ticker] -> RevisionSnapshot with:
- eps_last
- rev_1m (change over ~21 trading days)
- rev_3m (change over ~63 trading days)
- rev_1m_pct, rev_3m_pct (percent changes)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from scurve.core.utils import parse_asof


@dataclass(frozen=True)
class RevisionSnapshot:
    ticker: str
    asof: str
    eps_last: Optional[float] = None
    rev_1m: Optional[float] = None
    rev_3m: Optional[float] = None
    rev_1m_pct: Optional[float] = None
    rev_3m_pct: Optional[float] = None
    n_obs_used: int = 0


def load_revisions_overlays(
    cfg: Dict[str, Any],
    universe: Iterable[str],
    *,
    asof: str,
    logger=None,
) -> Dict[str, RevisionSnapshot]:
    """
    Load analyst revisions overlays as-of date.

    Config keys (optional):
      cfg['data']['revisions_csv'] : path to CSV (if missing, returns {})
      cfg['data']['revisions_lag_days'] : int lag days applied to 'date' before PIT filter
      cfg['data']['rev_1m_days'] : window for 1m change (default 21)
      cfg['data']['rev_3m_days'] : window for 3m change (default 63)

    Returns dict[ticker] -> RevisionSnapshot.
    """
    asof = parse_asof(asof)
    data_cfg = cfg.get("data", {})

    path = data_cfg.get("revisions_csv")
    if not path:
        # Optional module: if not configured, return empty overlays
        if logger:
            logger.info("revisions_skipped reason=missing_revisions_csv")
        return {}

    lag_days = int(data_cfg.get("revisions_lag_days", 0))
    w_1m = int(data_cfg.get("rev_1m_days", 21))
    w_3m = int(data_cfg.get("rev_3m_days", 63))

    df = _read_revisions_csv(path)
    if df.empty:
        if logger:
            logger.warning("revisions_empty path=%s", path)
        return {}

    df["ticker"] = df["ticker"].astype(str).str.upper()
    uni = {str(t).upper() for t in universe}
    df = df[df["ticker"].isin(uni)].copy()

    if df.empty:
        if logger:
            logger.info("revisions_empty_after_universe universe=%d", len(uni))
        return {}

    # PIT filter: available_date = date + lag_days
    df["available_date"] = df["date"] + pd.to_timedelta(lag_days, unit="D")
    asof_dt = pd.Timestamp(asof)
    df = df[df["available_date"] <= asof_dt].copy()

    out: Dict[str, RevisionSnapshot] = {}

    for tkr, g in df.groupby("ticker", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        if g.empty:
            continue

        # Latest EPS
        eps_last = float(g["eps_consensus"].iloc[-1])

        # Compute changes over trailing windows (calendar days approximation)
        # We use "nearest earlier observation at or before (last_date - window_days)".
        last_date = pd.Timestamp(g["date"].iloc[-1])

        rev_1m, rev_1m_pct = _change_from_window(g, last_date, w_1m)
        rev_3m, rev_3m_pct = _change_from_window(g, last_date, w_3m)

        out[tkr] = RevisionSnapshot(
            ticker=tkr,
            asof=asof,
            eps_last=eps_last,
            rev_1m=rev_1m,
            rev_3m=rev_3m,
            rev_1m_pct=rev_1m_pct,
            rev_3m_pct=rev_3m_pct,
            n_obs_used=int(len(g)),
        )

    if logger:
        logger.info("revisions_loaded asof=%s universe_in=%d overlays_out=%d lag_days=%d", asof, len(uni), len(out), lag_days)

    return out


# -------------------------
# Internals
# -------------------------

_COL_ALIASES = {
    "date": {"date", "asof", "obs_date", "timestamp"},
    "ticker": {"ticker", "symbol"},
    "eps_consensus": {"eps_consensus", "consensus_eps", "eps", "mean_eps", "estimate_eps"},
}


def _read_revisions_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"revisions_csv not found: {p}")

    df = pd.read_csv(p)
    if df.empty:
        return df

    df = _standardize_columns(df)

    required = ["date", "ticker", "eps_consensus"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"revisions_csv missing required columns: {missing}. Have columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["eps_consensus"] = pd.to_numeric(df["eps_consensus"], errors="coerce")

    df = df.dropna(subset=["date", "ticker", "eps_consensus"]).copy()
    return df


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    rename_map = {}
    for canon, aliases in _COL_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                rename_map[a] = canon
                break

    if rename_map:
        df = df.rename(columns=rename_map)

    return df


def _change_from_window(g: pd.DataFrame, last_date: pd.Timestamp, window_days: int) -> tuple[Optional[float], Optional[float]]:
    """
    Compute absolute and percent change in eps_consensus from an observation
    window_days before last_date (nearest earlier).
    """
    if g.empty or window_days <= 0:
        return None, None

    target = last_date - pd.Timedelta(days=int(window_days))
    # Find last row with date <= target
    hist = g[g["date"] <= target]
    if hist.empty:
        return None, None

    eps_then = float(hist["eps_consensus"].iloc[-1])
    eps_last = float(g["eps_consensus"].iloc[-1])
    delta = eps_last - eps_then

    if eps_then == 0.0:
        pct = None
    else:
        pct = (eps_last / eps_then) - 1.0

    return float(delta), (float(pct) if pct is not None else None)