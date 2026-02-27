"""
scurve.backtest.metrics

Performance and robustness metrics for Agora Lycos backtests.

Designed to work with:
- walkforward.run_walkforward() output DataFrame (quarterly rebalances)
- Optional series/dataframes of cross-sectional predictions for IC / decile spreads

Key goals:
- Deterministic, explicit formulas
- Minimal dependencies (pandas, numpy)
- Clear handling of annualization for quarterly series
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Helpers
# -------------------------

def _to_series(x, name: str = "x") -> pd.Series:
    if isinstance(x, pd.Series):
        return x.dropna()
    if isinstance(x, (list, tuple, np.ndarray)):
        return pd.Series(x, name=name).dropna()
    raise TypeError(f"Expected pd.Series or array-like for {name}, got {type(x)}")


def _ann_factor(freq: str) -> float:
    """
    Annualization factor for period returns.
    """
    f = freq.lower()
    if f in ("daily", "d"):
        return 252.0
    if f in ("weekly", "w"):
        return 52.0
    if f in ("monthly", "m"):
        return 12.0
    if f in ("quarterly", "q"):
        return 4.0
    if f in ("yearly", "y", "annual"):
        return 1.0
    raise ValueError(f"Unknown freq: {freq}")


def _safe_div(a: float, b: float) -> float:
    return float(a / b) if b not in (0.0, -0.0) else float("nan")


# -------------------------
# Core performance metrics
# -------------------------

def equity_curve_from_returns(r: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """
    Convert simple returns into an equity curve.
    """
    r = _to_series(r, "returns")
    return initial_value * (1.0 + r).cumprod()


def cumulative_return(r: pd.Series) -> float:
    """
    Cumulative simple return over the period.
    """
    r = _to_series(r, "returns")
    if r.empty:
        return float("nan")
    return float((1.0 + r).prod() - 1.0)


def cagr(r: pd.Series, freq: str = "quarterly") -> float:
    """
    Compound annual growth rate from period returns.
    """
    r = _to_series(r, "returns")
    if r.empty:
        return float("nan")

    ann = _ann_factor(freq)
    n_periods = float(len(r))
    years = n_periods / ann
    if years <= 0.0:
        return float("nan")

    total = (1.0 + r).prod()
    return float(total ** (1.0 / years) - 1.0)


def annualized_vol(r: pd.Series, freq: str = "quarterly", ddof: int = 1) -> float:
    """
    Annualized volatility of period returns.
    """
    r = _to_series(r, "returns")
    if len(r) < 2:
        return float("nan")
    ann = _ann_factor(freq)
    return float(r.std(ddof=ddof) * np.sqrt(ann))


def sharpe_ratio(
    r: pd.Series,
    freq: str = "quarterly",
    rf_annual: float = 0.0,
    ddof: int = 1,
) -> float:
    """
    Annualized Sharpe ratio.

    rf_annual is annual risk-free rate in decimal (e.g. 0.03 = 3%).
    For quarterly series, period rf is approximated by rf_annual / 4.
    """
    r = _to_series(r, "returns")
    if len(r) < 2:
        return float("nan")

    ann = _ann_factor(freq)
    rf_period = rf_annual / ann
    excess = r - rf_period

    mu = float(excess.mean())
    sig = float(excess.std(ddof=ddof))
    if sig == 0.0:
        return float("nan")

    return float((mu / sig) * np.sqrt(ann))


def max_drawdown(r: pd.Series, initial_value: float = 1.0) -> Tuple[float, pd.Timestamp | None, pd.Timestamp | None]:
    """
    Max drawdown computed from an equity curve.
    Returns: (max_dd, peak_date, trough_date)
    max_dd is negative (e.g., -0.25 means -25% drawdown).
    """
    r = _to_series(r, "returns")
    if r.empty:
        return float("nan"), None, None

    eq = equity_curve_from_returns(r, initial_value=initial_value)
    peaks = eq.cummax()
    dd = (eq / peaks) - 1.0

    trough = dd.idxmin()
    max_dd = float(dd.loc[trough])

    # peak is the last time before trough where the peak was attained
    peak_candidates = peaks.loc[:trough]
    peak_val = float(peak_candidates.max())
    peak_date = peak_candidates[peak_candidates == peak_val].index[-1] if len(peak_candidates) else None

    return max_dd, peak_date, trough


def hit_rate(r: pd.Series) -> float:
    """
    Fraction of periods with positive returns.
    """
    r = _to_series(r, "returns")
    if r.empty:
        return float("nan")
    return float((r > 0.0).mean())


def downside_deviation(r: pd.Series, freq: str = "quarterly", mar_annual: float = 0.0) -> float:
    """
    Annualized downside deviation relative to MAR (minimum acceptable return).
    """
    r = _to_series(r, "returns")
    if len(r) < 2:
        return float("nan")

    ann = _ann_factor(freq)
    mar_period = mar_annual / ann
    downside = np.minimum(r - mar_period, 0.0)
    return float(np.sqrt((downside**2).mean()) * np.sqrt(ann))


def sortino_ratio(r: pd.Series, freq: str = "quarterly", mar_annual: float = 0.0) -> float:
    r = _to_series(r, "returns")
    if len(r) < 2:
        return float("nan")
    ann = _ann_factor(freq)
    mar_period = mar_annual / ann
    excess_mean_annual = float((r - mar_period).mean()) * ann
    dd_annual = downside_deviation(r, freq=freq, mar_annual=mar_annual)
    if dd_annual == 0.0 or np.isnan(dd_annual):
        return float("nan")
    return float(excess_mean_annual / dd_annual)


# -------------------------
# Backtest table summarizer
# -------------------------

@dataclass(frozen=True)
class PerfSummary:
    """
    Standard performance summary for a period-return series.
    """
    cumulative_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    max_drawdown: float
    hit_rate: float


def summarize_returns(
    r: pd.Series,
    *,
    freq: str = "quarterly",
    rf_annual: float = 0.0,
) -> PerfSummary:
    """
    Bundle the common metrics used in model governance.
    """
    r = _to_series(r, "returns")
    mdd, _, _ = max_drawdown(r)
    return PerfSummary(
        cumulative_return=cumulative_return(r),
        cagr=cagr(r, freq=freq),
        ann_vol=annualized_vol(r, freq=freq),
        sharpe=sharpe_ratio(r, freq=freq, rf_annual=rf_annual),
        max_drawdown=mdd,
        hit_rate=hit_rate(r),
    )


def summarize_walkforward(
    wf: pd.DataFrame,
    *,
    freq: str = "quarterly",
    rf_annual: float = 0.0,
    return_col: str = "net_return",
) -> Dict[str, float]:
    """
    Summarize a walkforward DataFrame (from run_walkforward).

    Expects columns:
      - net_return (default) or gross_return
      - turnover (optional)
      - cost_drag (optional)

    Returns a dict of key metrics.
    """
    if wf is None or wf.empty:
        return {}

    r = wf[return_col].dropna()
    out = summarize_returns(r, freq=freq, rf_annual=rf_annual)

    res: Dict[str, float] = {
        "cum_return": out.cumulative_return,
        "cagr": out.cagr,
        "ann_vol": out.ann_vol,
        "sharpe": out.sharpe,
        "max_drawdown": out.max_drawdown,
        "hit_rate": out.hit_rate,
        "n_periods": float(len(r)),
    }

    if "turnover" in wf.columns:
        t = wf["turnover"].dropna()
        res["avg_turnover"] = float(t.mean()) if not t.empty else float("nan")
        res["median_turnover"] = float(t.median()) if not t.empty else float("nan")

    if "cost_drag" in wf.columns:
        cd = wf["cost_drag"].dropna()
        res["avg_cost_drag"] = float(cd.mean()) if not cd.empty else float("nan")
        res["total_cost_drag"] = float(cd.sum()) if not cd.empty else float("nan")

    return res


# -------------------------
# Cross-sectional validation: IC + deciles
# -------------------------

def information_coefficient(
    scores: pd.Series,
    fwd_returns: pd.Series,
    method: str = "spearman",
) -> float:
    """
    Information Coefficient between cross-sectional scores and forward returns.

    method: "spearman" (rank IC) or "pearson".
    """
    s = pd.Series(scores).dropna()
    r = pd.Series(fwd_returns).dropna()
    df = pd.concat([s.rename("score"), r.rename("ret")], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    m = method.lower()
    if m == "spearman":
        return float(df["score"].corr(df["ret"], method="spearman"))
    if m == "pearson":
        return float(df["score"].corr(df["ret"], method="pearson"))
    raise ValueError(f"Unknown method: {method}")


def decile_spread(
    scores: pd.Series,
    fwd_returns: pd.Series,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute decile (or quantile) bucket returns and top-bottom spread.

    Returns dict:
      - top_mean
      - bottom_mean
      - spread
      - monotonic_spearman (corr between bucket index and mean return)
    """
    s = pd.Series(scores).dropna()
    r = pd.Series(fwd_returns).dropna()
    df = pd.concat([s.rename("score"), r.rename("ret")], axis=1).dropna()
    if len(df) < max(20, n_bins * 5):
        return {"top_mean": float("nan"), "bottom_mean": float("nan"), "spread": float("nan"), "monotonic_spearman": float("nan")}

    # qcut can fail with many duplicates; rank first for stability
    df["score_rank"] = df["score"].rank(method="first")
    df["bin"] = pd.qcut(df["score_rank"], q=n_bins, labels=False, duplicates="drop")

    bucket_means = df.groupby("bin")["ret"].mean().sort_index()
    if bucket_means.empty:
        return {"top_mean": float("nan"), "bottom_mean": float("nan"), "spread": float("nan"), "monotonic_spearman": float("nan")}

    bottom_mean = float(bucket_means.iloc[0])
    top_mean = float(bucket_means.iloc[-1])
    spread = float(top_mean - bottom_mean)

    # monotonicity proxy: do higher bins tend to have higher returns?
    idx = pd.Series(bucket_means.index.astype(float), index=bucket_means.index)
    monotonic = float(idx.corr(bucket_means, method="spearman")) if len(bucket_means) >= 3 else float("nan")

    return {
        "top_mean": top_mean,
        "bottom_mean": bottom_mean,
        "spread": spread,
        "monotonic_spearman": monotonic,
    }


def rolling_ic(
    panel: pd.DataFrame,
    score_col: str,
    ret_col: str,
    date_col: str = "date",
    method: str = "spearman",
) -> pd.Series:
    """
    Compute IC per date for a panel with columns: [date_col, score_col, ret_col, ...].

    Returns Series indexed by date.
    """
    if panel is None or panel.empty:
        return pd.Series(dtype=float)

    df = panel[[date_col, score_col, ret_col]].dropna()
    out = {}
    for d, g in df.groupby(date_col):
        out[pd.Timestamp(d)] = information_coefficient(g[score_col], g[ret_col], method=method)
    return pd.Series(out).sort_index()


# -------------------------
# Turnover / stability helpers
# -------------------------

def rank_overlap(top_a: pd.Index, top_b: pd.Index) -> float:
    """
    Overlap fraction between two sets of tickers (e.g., top decile vs prior quarter).
    """
    a = set(map(str, list(top_a)))
    b = set(map(str, list(top_b)))
    if not a:
        return float("nan")
    return float(len(a & b) / len(a))


def rank_correlation(
    scores_a: pd.Series,
    scores_b: pd.Series,
    method: str = "spearman",
) -> float:
    """
    Correlation of scores across two dates on intersection of tickers.
    Useful for measuring rank stability / drift.
    """
    a = pd.Series(scores_a).dropna()
    b = pd.Series(scores_b).dropna()
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    if len(df) < 5:
        return float("nan")
    return float(df["a"].corr(df["b"], method=method))