"""
scurve.data.market

Market data adapter + lightweight overlays for Agora Lycos S-Curve model.

This module is intentionally adapter-friendly:
- In production you may load from a database, vendor, or parquet store.
- For now, we provide a CSV-based adapter that is deterministic and PIT-safe.

Primary responsibilities:
- Load daily close prices (and optionally volume) for tickers.
- Compute overlays used by the S-Curve model and portfolio layer:
    - trailing realized volatility (e.g., 60d)
    - momentum (e.g., 252d, 63d)
    - breakout flags (e.g., close > 6m high)
    - average daily dollar volume (ADV$) for liquidity filters
- Provide an "as-of" snapshot per ticker (single-row dict-like payload).

NOTE:
- This module does NOT enforce point-in-time fundamentals. It is market data.
- It does enforce that overlays only use data <= asof.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from scurve.core.utils import parse_asof


@dataclass(frozen=True)
class MarketSnapshot:
    """
    Per-ticker market overlay snapshot as of a given date.

    Fields are optional depending on what you load.
    """
    ticker: str
    asof: str
    close: Optional[float] = None
    vol_60d_ann: Optional[float] = None
    mom_12m: Optional[float] = None
    mom_3m: Optional[float] = None
    breakout_6m: Optional[bool] = None
    adv_dollar_3m: Optional[float] = None


def load_market_overlays(
    cfg: Dict[str, Any],
    universe: Iterable[str],
    *,
    asof: str,
    logger=None,
) -> Dict[str, MarketSnapshot]:
    """
    Load market overlays for tickers in universe, as-of date.

    Supported adapters:
      - CSV: cfg['data']['prices_csv'] + optional cfg['data']['volume_csv']
      - yfinance: cfg['data']['provider'] = "yfinance"

    Required prices CSV columns (case-insensitive mapping supported):
      - date
      - ticker
      - close

    Optional volume CSV columns:
      - date
      - ticker
      - volume

    Returns:
      dict[ticker] -> MarketSnapshot
    """
    asof = parse_asof(asof)
    data_cfg = cfg.get("data", {})
    provider = str(data_cfg.get("provider", "auto")).strip().lower()

    prices_path = data_cfg.get("prices_csv")
    vol_path = data_cfg.get("volume_csv")  # optional

    # Overlay windows can live in config, but defaults are safe
    vol_window = int(data_cfg.get("vol_window_days", 60))
    mom_12m_days = int(data_cfg.get("mom_12m_days", 252))
    mom_3m_days = int(data_cfg.get("mom_3m_days", 63))
    breakout_6m_days = int(data_cfg.get("breakout_6m_days", 126))
    adv_window = int(data_cfg.get("adv_window_days", 63))

    uni = {str(t).upper() for t in universe}
    asof_dt = pd.Timestamp(asof)
    if provider in {"yfinance", "yf"} or (provider == "auto" and not prices_path):
        prices, vol_df = _read_market_yfinance(cfg, uni, asof=asof, logger=logger)
    else:
        if not prices_path:
            raise ValueError(
                "Missing cfg['data']['prices_csv'] path for market data. "
                "Set data.provider='yfinance' to fetch from Yahoo Finance."
            )
        prices = _read_prices_csv(prices_path)
        prices["ticker"] = prices["ticker"].astype(str).str.upper()
        prices = prices[prices["ticker"].isin(uni)].copy()
        prices = prices[prices["date"] <= asof_dt].copy()

        vol_df = None
        if vol_path:
            vol_df = _read_volume_csv(vol_path)
            vol_df["ticker"] = vol_df["ticker"].astype(str).str.upper()
            vol_df = vol_df[vol_df["ticker"].isin(uni)].copy()
            vol_df = vol_df[vol_df["date"] <= asof_dt].copy()

    if prices.empty:
        if logger:
            logger.warning("market_prices_empty universe=%d asof=%s", len(uni), asof)
        return {}

    out: Dict[str, MarketSnapshot] = {}

    for tkr, g in prices.groupby("ticker", sort=True):
        g = g.sort_values("date").reset_index(drop=True)
        if g.empty:
            continue

        close = float(g["close"].iloc[-1])

        # daily returns for vol/mom (simple returns)
        rets = g["close"].pct_change()

        vol_60d_ann = _realized_vol_ann(rets, window=vol_window)
        mom_12m = _momentum(g["close"], window=mom_12m_days)
        mom_3m = _momentum(g["close"], window=mom_3m_days)
        breakout_6m = _breakout(g["close"], window=breakout_6m_days)

        adv_dollar = None
        if vol_df is not None and not vol_df.empty:
            gv = vol_df[vol_df["ticker"] == tkr].sort_values("date").reset_index(drop=True)
            if not gv.empty:
                # align dates between prices and volume by merging
                merged = pd.merge(g[["date", "close"]], gv[["date", "volume"]], on="date", how="inner")
                adv_dollar = _adv_dollar(merged, window=adv_window)

        out[tkr] = MarketSnapshot(
            ticker=tkr,
            asof=asof,
            close=close,
            vol_60d_ann=vol_60d_ann,
            mom_12m=mom_12m,
            mom_3m=mom_3m,
            breakout_6m=breakout_6m,
            adv_dollar_3m=adv_dollar,
        )

    if logger:
        logger.info("market_overlays_loaded asof=%s universe_in=%d overlays_out=%d", asof, len(uni), len(out))

    return out


# -------------------------
# CSV adapters
# -------------------------

def _read_prices_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"prices_csv not found: {p}")

    df = pd.read_csv(p)
    if df.empty:
        return df

    df = _standardize_columns(df)
    required = ["date", "ticker", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"prices_csv missing required columns: {missing}. Have columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "close"]).copy()
    return df


def _read_volume_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"volume_csv not found: {p}")

    df = pd.read_csv(p)
    if df.empty:
        return df

    df = _standardize_columns(df)
    required = ["date", "ticker", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"volume_csv missing required columns: {missing}. Have columns: {list(df.columns)}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "volume"]).copy()
    return df


def _read_market_yfinance(
    cfg: Dict[str, Any],
    universe: Iterable[str],
    *,
    asof: str,
    logger=None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        import yfinance as yf  # type: ignore
    except Exception as e:
        raise ImportError("yfinance is required for data.provider='yfinance'.") from e

    data_cfg = cfg.get("data", {})
    yf_cfg = data_cfg.get("yfinance", {}) if isinstance(data_cfg.get("yfinance", {}), dict) else {}

    start = str(yf_cfg.get("history_start", "2010-01-01"))
    chunk_size = int(yf_cfg.get("chunk_size", 100))
    max_tickers = int(yf_cfg.get("max_tickers", 0))

    tickers = sorted({str(t).upper() for t in universe})
    if max_tickers > 0:
        tickers = tickers[:max_tickers]

    # yfinance end is exclusive
    end = (pd.Timestamp(asof) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    prices_parts = []
    vol_parts = []

    for i in range(0, len(tickers), max(1, chunk_size)):
        chunk = tickers[i:i + max(1, chunk_size)]
        try:
            raw = yf.download(
                tickers=chunk,
                start=start,
                end=end,
                auto_adjust=False,
                progress=False,
                group_by="column",
                threads=True,
            )
        except Exception:
            continue

        if raw is None or raw.empty:
            continue

        close_long, vol_long = _extract_close_volume_from_yf(raw, chunk)
        if not close_long.empty:
            prices_parts.append(close_long)
        if not vol_long.empty:
            vol_parts.append(vol_long)

    prices = (
        pd.concat(prices_parts, ignore_index=True)
        if prices_parts
        else pd.DataFrame(columns=["date", "ticker", "close"])
    )
    volume = (
        pd.concat(vol_parts, ignore_index=True)
        if vol_parts
        else pd.DataFrame(columns=["date", "ticker", "volume"])
    )

    prices = prices.dropna(subset=["date", "ticker", "close"]).copy()
    prices["date"] = pd.to_datetime(prices["date"], errors="coerce")
    prices["ticker"] = prices["ticker"].astype(str).str.upper()
    prices["close"] = pd.to_numeric(prices["close"], errors="coerce")
    prices = prices.dropna(subset=["date", "ticker", "close"]).copy()

    volume = volume.dropna(subset=["date", "ticker", "volume"]).copy()
    volume["date"] = pd.to_datetime(volume["date"], errors="coerce")
    volume["ticker"] = volume["ticker"].astype(str).str.upper()
    volume["volume"] = pd.to_numeric(volume["volume"], errors="coerce")
    volume = volume.dropna(subset=["date", "ticker", "volume"]).copy()

    if logger:
        logger.info(
            "market_yfinance_loaded asof=%s tickers_in=%d price_rows=%d vol_rows=%d",
            asof,
            len(tickers),
            len(prices),
            len(volume),
        )

    return prices, volume


def _extract_close_volume_from_yf(raw: pd.DataFrame, tickers: list[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = set(raw.columns.get_level_values(0))
        close_df = raw.xs("Close", axis=1, level=0, drop_level=True) if "Close" in lvl0 else pd.DataFrame()
        vol_df = raw.xs("Volume", axis=1, level=0, drop_level=True) if "Volume" in lvl0 else pd.DataFrame()
    else:
        tkr = tickers[0] if tickers else "UNKNOWN"
        close_df = raw[["Close"]].rename(columns={"Close": tkr}) if "Close" in raw.columns else pd.DataFrame()
        vol_df = raw[["Volume"]].rename(columns={"Volume": tkr}) if "Volume" in raw.columns else pd.DataFrame()

    close_long = _panel_to_long(close_df, "close")
    vol_long = _panel_to_long(vol_df, "volume")
    return close_long, vol_long


def _panel_to_long(panel: pd.DataFrame, value_name: str) -> pd.DataFrame:
    if panel is None or panel.empty:
        return pd.DataFrame(columns=["date", "ticker", value_name])

    if isinstance(panel, pd.Series):
        panel = panel.to_frame()

    out = panel.copy()
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.dropna(axis=0, how="all")
    if out.empty:
        return pd.DataFrame(columns=["date", "ticker", value_name])

    long = out.stack().reset_index()
    long.columns = ["date", "ticker", value_name]
    return long


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    # common aliases
    rename_map = {}
    for c in df.columns:
        if c in ("symbol",):
            rename_map[c] = "ticker"
        if c in ("adjclose", "adj_close", "adjusted_close"):
            rename_map[c] = "close"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# -------------------------
# Overlay calculations
# -------------------------

def _realized_vol_ann(rets: pd.Series, window: int) -> Optional[float]:
    """
    Annualized realized volatility using simple daily returns.
    """
    if rets is None or rets.dropna().empty:
        return None
    r = rets.dropna()
    if len(r) < max(5, window // 4):
        return None
    sub = r.iloc[-window:] if len(r) >= window else r
    vol = float(sub.std(ddof=1) * np.sqrt(252.0))
    return vol


def _momentum(close: pd.Series, window: int) -> Optional[float]:
    """
    Simple momentum: close / close[-window] - 1
    """
    if close is None or close.dropna().empty:
        return None
    s = close.dropna()
    if len(s) <= window:
        return None
    return float(s.iloc[-1] / s.iloc[-(window + 1)] - 1.0)


def _breakout(close: pd.Series, window: int) -> Optional[bool]:
    """
    Breakout flag: last close greater than max close over prior window.
    """
    if close is None or close.dropna().empty:
        return None
    s = close.dropna()
    if len(s) <= window:
        return None
    last = float(s.iloc[-1])
    prior_max = float(s.iloc[-(window + 1):-1].max())
    return bool(last > prior_max)


def _adv_dollar(merged: pd.DataFrame, window: int) -> Optional[float]:
    """
    Average daily dollar volume over trailing window using close * volume.
    """
    if merged is None or merged.empty:
        return None
    merged = merged.dropna(subset=["close", "volume"]).copy()
    if merged.empty:
        return None

    merged["dollar_vol"] = merged["close"].astype(float) * merged["volume"].astype(float)
    s = merged["dollar_vol"]
    if len(s) < max(5, window // 4):
        return None
    sub = s.iloc[-window:] if len(s) >= window else s
    return float(sub.mean())
