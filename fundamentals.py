"""
scurve.data.fundamentals

Fundamentals ingestion + point-in-time enforcement for the Agora Lycos S-Curve model.

This module is intentionally "adapter-friendly":
- In production you may source fundamentals from a vendor/DB/API.
- For now, this file provides a clean contract and a practical CSV-based adapter.

Core requirements (per charter/process):
- Quarterly fundamentals (revenue) as primary S-curve signal
- Revenue_TTM computed on a quarterly grid
- Point-in-time alignment via "available_date" = report_release_date + lag_days
- Enforce minimum history (min_quarters) before downstream fitting/scoring

Data contract
-------------
We represent each ticker's quarterly series as a FundamentalsSeries object with:
- df: columns ['period_end', 'value', 'report_date', 'available_date']
- value: Revenue_TTM (default), or raw quarterly revenue before transform

Downstream expects:
- series.df sorted by period_end ascending
- series.available_date_latest: latest available_date <= asof
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from scurve.core.utils import parse_asof


# -------------------------
# Data types
# -------------------------

@dataclass(frozen=True)
class FundamentalsSeries:
    """
    Quarterly fundamentals series for one ticker.

    df columns (required):
      - period_end: pandas.Timestamp (quarter end date)
      - value: float (Revenue_TTM by default)
      - report_date: pandas.Timestamp (filing / release date)
      - available_date: pandas.Timestamp (report_date + lag)

    Notes:
    - 'value' is the modeled scalar. For S-Curve we prefer Revenue_TTM.
    - available_date is used for point-in-time enforcement.
    """
    ticker: str
    df: pd.DataFrame

    @property
    def available_date_latest(self) -> Optional[pd.Timestamp]:
        if self.df.empty:
            return None
        return pd.Timestamp(self.df["available_date"].max())

    def asof_slice(self, asof: str) -> "FundamentalsSeries":
        """
        Return a series containing only rows with available_date <= asof.
        """
        asof_dt = pd.Timestamp(parse_asof(asof))
        sub = self.df.loc[self.df["available_date"] <= asof_dt].copy()
        return FundamentalsSeries(self.ticker, sub)


# -------------------------
# Public API
# -------------------------

def load_revenue_ttm_pit(
    cfg: Dict[str, Any],
    universe: Iterable[str],
    *,
    asof: str,
    logger=None,
) -> Dict[str, FundamentalsSeries]:
    """
    Load Revenue_TTM series for tickers in universe with point-in-time enforcement.

    Supported adapters:
      - CSV: cfg['data']['fundamentals_csv']
      - yfinance: cfg['data']['provider'] = "yfinance"

    Required CSV columns (case-insensitive mapping supported):
      - ticker
      - period_end        (quarter end date)
      - revenue           (quarterly revenue; numeric)
      - report_date       (filing / release date)

    This function:
      - parses dates
      - computes available_date = report_date + lag_days
      - computes Revenue_TTM from quarterly revenue (rolling 4 quarters)
      - returns per-ticker FundamentalsSeries filtered to available_date <= asof
      - drops tickers with insufficient history (< min_quarters)

    Returns dict ticker -> FundamentalsSeries
    """
    asof = parse_asof(asof)
    data_cfg = cfg.get("data", {})
    provider = str(data_cfg.get("provider", "auto")).strip().lower()
    path = data_cfg.get("fundamentals_csv")

    lag_days = int(data_cfg.get("lag_days", 60))
    min_quarters = int(data_cfg.get("min_quarters", 16))

    if provider in {"yfinance", "yf"} or (provider == "auto" and not path):
        yf_cfg = data_cfg.get("yfinance", {}) if isinstance(data_cfg.get("yfinance", {}), dict) else {}
        # With PIT lag and Yahoo history limits, usable TTM points can be sparse.
        min_quarters = int(yf_cfg.get("min_quarters", min(min_quarters, 1)))
        df = _read_fundamentals_yfinance(cfg, universe, asof=asof, logger=logger)
    else:
        if not path:
            raise ValueError(
                "Missing cfg['data']['fundamentals_csv']. "
                "Set data.provider='yfinance' to fetch from Yahoo Finance."
            )
        df = _read_fundamentals_csv(path)

    # Filter to universe early
    uni = {str(t).upper() for t in universe}
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df = df[df["ticker"].isin(uni)].copy()

    if df.empty:
        if logger:
            logger.warning("fundamentals_empty universe=%d path=%s", len(uni), path)
        return {}

    df = _compute_available_date(df, lag_days=lag_days)
    df = _compute_revenue_ttm(df)

    # Point-in-time: only include rows available by asof
    asof_dt = pd.Timestamp(asof)
    df = df[df["available_date"] <= asof_dt].copy()

    # Build per-ticker series and enforce min history
    out: Dict[str, FundamentalsSeries] = {}
    for tkr, g in df.groupby("ticker", sort=True):
        g = g.sort_values("period_end").reset_index(drop=True)
        # Keep only rows where TTM exists
        g = g.dropna(subset=["value"]).copy()
        if len(g) < min_quarters:
            continue
        out[tkr] = FundamentalsSeries(ticker=tkr, df=g[["period_end", "value", "report_date", "available_date"]].copy())

    if logger:
        logger.info(
            "fundamentals_loaded asof=%s universe_in=%d series_out=%d lag_days=%d min_quarters=%d",
            asof, len(uni), len(out), lag_days, min_quarters
        )
    return out


# -------------------------
# CSV adapter + transforms
# -------------------------

_COL_ALIASES = {
    "ticker": {"ticker", "symbol"},
    "period_end": {"period_end", "periodend", "quarter_end", "fiscal_period_end", "date"},
    "revenue": {"revenue", "sales", "total_revenue"},
    "report_date": {"report_date", "reportdate", "filing_date", "filingdate", "release_date", "releasedate"},
}


def _read_fundamentals_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"fundamentals_csv not found: {p}")

    df = pd.read_csv(p)
    if df.empty:
        return df

    df = _standardize_columns(df)
    required = ["ticker", "period_end", "revenue", "report_date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"fundamentals_csv missing required columns: {missing}. Have columns: {list(df.columns)}")

    # parse dates
    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    df["report_date"] = pd.to_datetime(df["report_date"], errors="coerce")

    # numeric revenue
    df["revenue"] = pd.to_numeric(df["revenue"], errors="coerce")

    df = df.dropna(subset=["ticker", "period_end", "report_date", "revenue"]).copy()
    return df


def _read_fundamentals_yfinance(
    cfg: Dict[str, Any],
    universe: Iterable[str],
    *,
    asof: str,
    logger=None,
) -> pd.DataFrame:
    """
    Pull quarterly revenue via yfinance.

    Note:
    - Yahoo does not provide an explicit filing date in this endpoint.
    - We approximate report_date as period_end + report_delay_days.
    """
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        raise ImportError("yfinance is required for data.provider='yfinance'.") from e

    data_cfg = cfg.get("data", {})
    yf_cfg = data_cfg.get("yfinance", {}) if isinstance(data_cfg.get("yfinance", {}), dict) else {}

    report_delay_days = int(yf_cfg.get("report_delay_days", 45))
    max_tickers = int(yf_cfg.get("max_tickers", 0))

    tickers = sorted({str(t).upper() for t in universe})
    if max_tickers > 0:
        tickers = tickers[:max_tickers]

    rows: list[Dict[str, Any]] = []
    asof_dt = pd.Timestamp(asof)
    n_ok = 0

    for tkr in tickers:
        try:
            stmt = yf.Ticker(tkr).quarterly_income_stmt
            if stmt is None or stmt.empty:
                continue

            revenue_key = None
            for k in ("Total Revenue", "TotalRevenue", "Revenue"):
                if k in stmt.index:
                    revenue_key = k
                    break
            if revenue_key is None:
                continue

            rev = pd.to_numeric(stmt.loc[revenue_key], errors="coerce").dropna()
            if rev.empty:
                continue

            n_ok += 1
            for period_end, val in rev.items():
                pe = pd.to_datetime(period_end, errors="coerce")
                if pd.isna(pe):
                    continue
                # Keep only potentially usable rows up to asof horizon.
                if pe > asof_dt:
                    continue

                rows.append(
                    {
                        "ticker": tkr,
                        "period_end": pe,
                        "revenue": float(val),
                        "report_date": pe + pd.to_timedelta(report_delay_days, unit="D"),
                    }
                )
        except Exception:
            continue

    out = pd.DataFrame(rows, columns=["ticker", "period_end", "revenue", "report_date"])
    if logger:
        logger.info(
            "fundamentals_yfinance_loaded asof=%s universe_in=%d tickers_with_data=%d rows=%d",
            asof,
            len(tickers),
            n_ok,
            len(out),
        )
    return out


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=cols)

    # map aliases -> canonical
    new_cols = {}
    for canon, aliases in _COL_ALIASES.items():
        for a in aliases:
            if a in df.columns:
                new_cols[a] = canon
                break
    df = df.rename(columns=new_cols)

    return df


def _compute_available_date(df: pd.DataFrame, lag_days: int) -> pd.DataFrame:
    df = df.copy()
    df["available_date"] = df["report_date"] + pd.to_timedelta(int(lag_days), unit="D")
    return df


def _compute_revenue_ttm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Revenue_TTM as rolling sum of last 4 quarterly revenues per ticker.

    Sets output column 'value' to Revenue_TTM.
    """
    df = df.copy()
    df = df.sort_values(["ticker", "period_end"]).reset_index(drop=True)

    df["value"] = (
        df.groupby("ticker", sort=False)["revenue"]
          .rolling(window=4, min_periods=4)
          .sum()
          .reset_index(level=0, drop=True)
    )

    return df
