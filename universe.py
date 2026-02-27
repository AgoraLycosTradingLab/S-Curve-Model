"""
scurve.data.universe

Universe construction + liquidity filters for Agora Lycos S-Curve model.

Responsibilities
----------------
- Load base universe (e.g., S&P500 + Nasdaq list)
- Apply liquidity / size filters:
    - min price
    - min market cap
    - min ADV$
- Return a clean list of tickers for the given as-of date

Design notes
------------
- This module does NOT enforce fundamentals history. That is handled in fundamentals.py.
- This module does NOT compute scores.
- All filters are deterministic and config-driven.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

import pandas as pd

from scurve.core.utils import parse_asof
from filterstemplate import load_tickers_from_csv, normalize_ticker


def build_universe(
    cfg: Dict[str, Any],
    *,
    asof: str,
    logger=None,
) -> List[str]:
    """
    Build filtered universe for the given as-of date.

    Config keys:
      Preferred:
          cfg['data']['sp500_csv']
          cfg['data']['nasdaq_csv']
      Fallback:
          cfg['data']['universe_csv']
      Optional base-universe narrowing:
          cfg['data']['universe_cap_buckets'] = ["Large", "Mid"]
          cfg['data']['universe_sectors'] = ["Technology", "Healthcare"]
          cfg['data']['universe_exclude_industries'] = ["Biotechnology"]
      Optional filter inputs:
          cfg['data']['market_cap_csv']
          cfg['data']['prices_csv'] (for price filter)
          cfg['data']['volume_csv'] (for ADV filter)

    Filter thresholds (config-driven):
      data.price_min
      data.market_cap_min_usd
      data.adv_dollar_min_usd

    Returns:
      sorted list of tickers
    """
    asof = parse_asof(asof)
    data_cfg = cfg.get("data", {})

    sp500_path = data_cfg.get("sp500_csv")
    nasdaq_path = data_cfg.get("nasdaq_csv")
    universe_path = data_cfg.get("universe_csv")

    cap_buckets = _to_upper_set(data_cfg.get("universe_cap_buckets"))
    sectors = _to_upper_set(data_cfg.get("universe_sectors"))
    exclude_industries = _to_upper_set(data_cfg.get("universe_exclude_industries"))

    if sp500_path or nasdaq_path:
        parts = []
        if sp500_path:
            parts.append(
                _read_master_universe_csv(
                    sp500_path,
                    cap_buckets=cap_buckets,
                    sectors=sectors,
                    exclude_industries=exclude_industries,
                )
            )
        if nasdaq_path:
            parts.append(
                _read_master_universe_csv(
                    nasdaq_path,
                    cap_buckets=cap_buckets,
                    sectors=sectors,
                    exclude_industries=exclude_industries,
                )
            )
        df_uni = pd.concat(parts, ignore_index=True).drop_duplicates()
    elif universe_path:
        df_uni = _read_universe_csv(universe_path)
    else:
        raise ValueError(
            "Missing universe config. Provide data.sp500_csv + data.nasdaq_csv "
            "or data.universe_csv."
        )

    tickers: Set[str] = set(df_uni["ticker"].astype(str).str.upper())

    # Apply filters
    tickers = _apply_price_filter(cfg, tickers, asof, logger)
    tickers = _apply_market_cap_filter(cfg, tickers, asof, logger)
    tickers = _apply_adv_filter(cfg, tickers, asof, logger)

    out = sorted(tickers)

    if logger:
        logger.info("universe_built asof=%s size=%d", asof, len(out))

    return out


# -------------------------
# Base universe loader
# -------------------------

def _read_universe_csv(path: str | Path) -> pd.DataFrame:
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"universe_csv not found: {p}")

    tickers = load_tickers_from_csv(p)
    return pd.DataFrame({"ticker": tickers})


def _read_master_universe_csv(
    path: str | Path,
    *,
    cap_buckets: Optional[Set[str]] = None,
    sectors: Optional[Set[str]] = None,
    exclude_industries: Optional[Set[str]] = None,
) -> pd.DataFrame:
    """
    Read master universe CSV (Ticker/CapBucket/Sector aware) using
    filterstemplate-style ticker normalization.
    """
    p = _resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"master universe csv not found: {p}")

    df = pd.read_csv(p)
    if df.empty:
        return pd.DataFrame(columns=["ticker"])

    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    if "ticker" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "ticker"})
    if "ticker" not in df.columns:
        # fallback: first column
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "ticker"})

    df["ticker"] = df["ticker"].astype(str).map(normalize_ticker)
    df = df[df["ticker"].str.len() > 0]

    if cap_buckets and "capbucket" in df.columns:
        col = df["capbucket"].astype(str).str.upper()
        df = df[col.isin(cap_buckets)]

    if sectors and "sector" in df.columns:
        col = df["sector"].astype(str).str.upper()
        df = df[col.isin(sectors)]

    if exclude_industries and "industry" in df.columns:
        col = df["industry"].astype(str).str.upper()
        df = df[~col.isin(exclude_industries)]

    return df[["ticker"]].drop_duplicates().copy()


def _resolve_path(path: str | Path) -> Path:
    """
    Resolve data file paths robustly across different launch CWDs.
    """
    p = Path(path)
    if p.exists() or p.is_absolute():
        return p

    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / p
    if candidate.exists():
        return candidate

    return p


def _to_upper_set(value: Any) -> Optional[Set[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        vals = [v.strip() for v in value.split(",")]
    elif isinstance(value, (list, tuple, set)):
        vals = list(value)
    else:
        return None
    out = {str(v).strip().upper() for v in vals if str(v).strip()}
    return out or None


# -------------------------
# Filters
# -------------------------

def _apply_price_filter(
    cfg: Dict[str, Any],
    tickers: Set[str],
    asof: str,
    logger=None,
) -> Set[str]:
    price_min = float(cfg.get("data", {}).get("price_min", 0.0))
    prices_path = cfg.get("data", {}).get("prices_csv")

    if price_min <= 0.0 or not prices_path:
        return tickers

    df = _read_prices_csv(prices_path)
    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    asof_dt = pd.Timestamp(asof)

    df = df[df["date"] <= asof_dt].copy()

    # latest close per ticker
    latest = (
        df.sort_values("date")
          .groupby("ticker", as_index=False)
          .tail(1)
    )

    price_map = dict(zip(latest["ticker"], latest["close"]))

    filtered = {t for t in tickers if price_map.get(t, 0.0) >= price_min}

    if logger:
        logger.info(
            "universe_price_filter applied min_price=%.2f before=%d after=%d",
            price_min, len(tickers), len(filtered)
        )

    return filtered


def _apply_market_cap_filter(
    cfg: Dict[str, Any],
    tickers: Set[str],
    asof: str,
    logger=None,
) -> Set[str]:
    mcap_min = float(cfg.get("data", {}).get("market_cap_min_usd", 0.0))
    path = cfg.get("data", {}).get("market_cap_csv")

    if mcap_min <= 0.0 or not path:
        return tickers

    df = pd.read_csv(path)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    if "ticker" not in df.columns:
        raise ValueError("market_cap_csv must contain 'ticker' column.")
    if "market_cap" not in df.columns:
        raise ValueError("market_cap_csv must contain 'market_cap' column.")

    df["ticker"] = df["ticker"].astype(str).str.upper()
    df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")

    mcap_map = dict(zip(df["ticker"], df["market_cap"]))

    filtered = {t for t in tickers if mcap_map.get(t, 0.0) >= mcap_min}

    if logger:
        logger.info(
            "universe_mcap_filter applied min_mcap=%.0f before=%d after=%d",
            mcap_min, len(tickers), len(filtered)
        )

    return filtered


def _apply_adv_filter(
    cfg: Dict[str, Any],
    tickers: Set[str],
    asof: str,
    logger=None,
) -> Set[str]:
    adv_min = float(cfg.get("data", {}).get("adv_dollar_min_usd", 0.0))
    prices_path = cfg.get("data", {}).get("prices_csv")
    volume_path = cfg.get("data", {}).get("volume_csv")

    if adv_min <= 0.0 or not prices_path or not volume_path:
        return tickers

    prices = _read_prices_csv(prices_path)
    volume = _read_volume_csv(volume_path)

    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    volume["ticker"] = volume["ticker"].astype(str).str.upper()

    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    volume["date"] = pd.to_datetime(volume["date"], errors="coerce")

    asof_dt = pd.Timestamp(asof)
    prices = prices[prices["date"] <= asof_dt]
    volume = volume[volume["date"] <= asof_dt]

    merged = pd.merge(
        prices[["date", "ticker", "close"]],
        volume[["date", "ticker", "volume"]],
        on=["date", "ticker"],
        how="inner",
    )

    merged["dollar_vol"] = merged["close"].astype(float) * merged["volume"].astype(float)

    # trailing 63 trading days default
    window = int(cfg.get("data", {}).get("adv_window_days", 63))

    adv_map = {}
    for tkr, g in merged.groupby("ticker"):
        g = g.sort_values("date")
        s = g["dollar_vol"]
        if len(s) >= window:
            adv_map[tkr] = float(s.iloc[-window:].mean())
        elif len(s) > 0:
            adv_map[tkr] = float(s.mean())
        else:
            adv_map[tkr] = 0.0

    filtered = {t for t in tickers if adv_map.get(t, 0.0) >= adv_min}

    if logger:
        logger.info(
            "universe_adv_filter applied min_adv=%.0f before=%d after=%d",
            adv_min, len(tickers), len(filtered)
        )

    return filtered


# -------------------------
# CSV helpers
# -------------------------

def _read_prices_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"prices_csv not found: {p}")

    df = pd.read_csv(p)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    if "ticker" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "ticker"})
    if "date" not in df.columns:
        raise ValueError("prices_csv must contain 'date' column.")
    if "close" not in df.columns:
        raise ValueError("prices_csv must contain 'close' column.")

    return df


def _read_volume_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"volume_csv not found: {p}")

    df = pd.read_csv(p)
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})

    if "ticker" not in df.columns and "symbol" in df.columns:
        df = df.rename(columns={"symbol": "ticker"})
    if "date" not in df.columns:
        raise ValueError("volume_csv must contain 'date' column.")
    if "volume" not in df.columns:
        raise ValueError("volume_csv must contain 'volume' column.")

    return df
