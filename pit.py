"""
scurve.data.pit

Point-in-time (PIT) utilities for Agora Lycos S-Curve model.

Purpose
-------
Provide deterministic, reusable helpers to enforce "available as of date" constraints
for datasets with release/report timestamps (fundamentals, estimates, etc.).

Key concepts
------------
- event_date: the period being measured (e.g., quarter end)
- report_date: the date the market could first know the value (filing/release)
- available_date: report_date + lag_days (your enforcement rule)

This module does NOT fetch data. It only enforces timing logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd

from scurve.core.utils import parse_asof


@dataclass(frozen=True)
class PITSpec:
    """
    Defines PIT enforcement.

    lag_days:
      add a fixed lag to report_date to represent realistic availability.

    allowed_columns:
      optional for validation.
    """
    lag_days: int = 60
    report_date_col: str = "report_date"
    available_date_col: str = "available_date"


def compute_available_date(
    df: pd.DataFrame,
    *,
    report_date_col: str = "report_date",
    available_date_col: str = "available_date",
    lag_days: int = 60,
) -> pd.DataFrame:
    """
    Compute available_date = report_date + lag_days.

    Returns a copy of df with available_date_col.
    """
    if report_date_col not in df.columns:
        raise KeyError(f"Missing report_date column: {report_date_col}")
    out = df.copy()
    out[report_date_col] = pd.to_datetime(out[report_date_col], errors="coerce")
    out[available_date_col] = out[report_date_col] + pd.to_timedelta(int(lag_days), unit="D")
    return out


def filter_asof(
    df: pd.DataFrame,
    *,
    asof: str,
    available_date_col: str = "available_date",
    inclusive: bool = True,
) -> pd.DataFrame:
    """
    Filter rows to those available as-of date.

    inclusive=True keeps rows where available_date <= asof.
    inclusive=False keeps rows where available_date < asof.
    """
    if available_date_col not in df.columns:
        raise KeyError(f"Missing available_date column: {available_date_col}")
    asof_dt = pd.Timestamp(parse_asof(asof))
    out = df.copy()
    out[available_date_col] = pd.to_datetime(out[available_date_col], errors="coerce")
    if inclusive:
        return out.loc[out[available_date_col] <= asof_dt].copy()
    return out.loc[out[available_date_col] < asof_dt].copy()


def latest_available_row(
    df: pd.DataFrame,
    *,
    asof: str,
    available_date_col: str = "available_date",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Return the latest available row per group (or globally if group_col is None)
    for rows with available_date <= asof.

    Useful for getting latest fundamentals snapshot as-of date.

    Returns a DataFrame subset (may be empty).
    """
    filtered = filter_asof(df, asof=asof, available_date_col=available_date_col, inclusive=True)
    if filtered.empty:
        return filtered

    filtered = filtered.sort_values(available_date_col)

    if group_col is None:
        return filtered.tail(1).copy()

    if group_col not in filtered.columns:
        raise KeyError(f"Missing group_col: {group_col}")

    # Latest per group: sort then take last in each group
    return filtered.groupby(group_col, sort=True, as_index=False).tail(1).copy()


def assert_no_lookahead(
    df: pd.DataFrame,
    *,
    asof: str,
    available_date_col: str = "available_date",
    raise_on_fail: bool = True,
) -> bool:
    """
    Sanity check that there are no rows with available_date > asof.
    Returns True if clean, False otherwise (or raises).
    """
    if available_date_col not in df.columns:
        raise KeyError(f"Missing available_date column: {available_date_col}")
    asof_dt = pd.Timestamp(parse_asof(asof))
    s = pd.to_datetime(df[available_date_col], errors="coerce")
    bad = s > asof_dt
    ok = not bool(bad.any())
    if not ok and raise_on_fail:
        n = int(bad.sum())
        max_bad = s[bad].max()
        raise ValueError(f"PIT lookahead violation: {n} rows have {available_date_col} > asof ({asof}). Max={max_bad}")
    return ok


def enforce_min_history(
    df: pd.DataFrame,
    *,
    group_col: str,
    min_rows: int,
) -> pd.DataFrame:
    """
    Drop groups with fewer than min_rows rows.
    """
    if df.empty:
        return df
    if group_col not in df.columns:
        raise KeyError(f"Missing group_col: {group_col}")
    if int(min_rows) <= 0:
        return df

    counts = df.groupby(group_col)[group_col].transform("count")
    return df.loc[counts >= int(min_rows)].copy()