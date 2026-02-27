"""
scurve.backtest.walkforward

Walk-forward backtest harness for Agora Lycos S-Curve model.

Core idea:
- On each rebalance date (quarterly), call a user-provided function to generate target weights.
- Hold those weights until next rebalance date.
- Compute portfolio return from constituent returns over the hold period.
- Apply turnover-based transaction costs using scurve.backtest.cost.

This module is intentionally model-agnostic:
- It doesn't fit curves or compute scores.
- It expects "weights_by_date" or a "weights_fn(asof)->weights" callback.

Assumptions:
- Long-only weights (>=0) that sum to 1.0 (cash optional if you include it explicitly).
- Returns are provided as simple returns (not log returns).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from scurve.backtest.cost import CostModel, net_return_after_rebalance


Weights = Dict[str, float]


@dataclass(frozen=True)
class BacktestConfig:
    """
    Backtest controls. Keep this thin; model config stays elsewhere.
    """
    rebalance: str = "quarterly"         # "quarterly" only for now
    initial_value: float = 1_000_000.0
    # If True, normalize weights to sum to 1.0 (after dropping invalid tickers)
    normalize_weights: bool = True
    # If True, drop tickers not present in returns for the holding period
    drop_missing_tickers: bool = True


def _quarter_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """
    Generate quarter-end dates within the provided date index range.

    We use calendar quarter ends (Mar/Jun/Sep/Dec).
    Returned dates are intersected with the provided index (nearest previous date).
    """
    if len(index) == 0:
        return pd.DatetimeIndex([])

    start = index.min()
    end = index.max()
    q_ends = pd.date_range(start=start, end=end, freq="QE")  # quarter end
    # Align to index by taking the last available date <= q_end
    aligned = []
    for d in q_ends:
        # slice up to d and pick last
        sub = index[index <= d]
        if len(sub) == 0:
            continue
        aligned.append(sub[-1])
    return pd.DatetimeIndex(sorted(set(aligned)))


def _normalize_weights(w: Weights) -> Weights:
    """
    Ensure non-negative and sum to 1.0 (if possible).
    """
    ww = {k: float(v) for k, v in w.items() if v is not None}
    # Drop negatives (long-only); you can relax later if needed
    ww = {k: v for k, v in ww.items() if v > 0.0}
    s = sum(ww.values())
    if s <= 0.0:
        return {}
    return {k: v / s for k, v in ww.items()}


def _portfolio_return(weights: Weights, period_rets: pd.Series) -> float:
    """
    Compute portfolio simple return for the period using end-of-period returns.

    weights: dict ticker->weight at start of holding period
    period_rets: Series indexed by ticker, values are simple returns over hold period

    Returns: simple portfolio return
    """
    if not weights:
        return 0.0
    # Align
    pr = 0.0
    for tkr, w in weights.items():
        if tkr in period_rets.index and pd.notna(period_rets.loc[tkr]):
            pr += float(w) * float(period_rets.loc[tkr])
        else:
            # Missing ticker return => treated as 0 if not dropped earlier
            pr += 0.0
    return float(pr)


def _period_simple_returns(
    daily_returns: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.Series:
    """
    Compute simple returns over [start_date+1 .. end_date] inclusive.

    daily_returns: DataFrame indexed by dates, columns=tickers, values=simple daily returns
    Returns a Series per ticker: (prod(1+r) - 1)
    """
    if start_date >= end_date:
        return pd.Series(dtype=float)

    # hold period starts AFTER rebalance close; assume weights set at start_date close
    # So returns accrue from next trading day.
    sub = daily_returns.loc[(daily_returns.index > start_date) & (daily_returns.index <= end_date)]
    if sub.empty:
        return pd.Series(dtype=float)

    gross = (1.0 + sub).prod(axis=0) - 1.0
    return gross


def _filter_weights_to_available_returns(
    w: Weights,
    period_rets: pd.Series,
) -> Weights:
    """
    Remove tickers with missing returns for the period (if desired).
    """
    if not w:
        return {}
    ok = {}
    for tkr, wt in w.items():
        if tkr in period_rets.index and pd.notna(period_rets.loc[tkr]):
            ok[tkr] = float(wt)
    return ok


def run_walkforward(
    daily_returns: pd.DataFrame,
    weights_fn: Callable[[pd.Timestamp], Weights],
    *,
    bt_cfg: Optional[BacktestConfig] = None,
    cost_model: Optional[CostModel] = None,
    rebalance_dates: Optional[Iterable[pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Run a quarterly walk-forward backtest.

    Parameters
    ----------
    daily_returns:
        DataFrame of simple daily returns.
        Index: DatetimeIndex (trading dates)
        Columns: tickers
        Values: simple daily returns (e.g., 0.01 for +1%)

    weights_fn:
        Callable(asof_date)->weights dict. The asof_date is the rebalance date.
        This is where you plug your S-Curve model output (top-decile weights).

    bt_cfg:
        BacktestConfig; default uses quarterly rebalance and $1,000,000 start.

    cost_model:
        CostModel; default matches charter typical assumptions (20 bps + 5 bps).

    rebalance_dates:
        Optional explicit rebalance dates; if None, use quarter-ends aligned to trading dates.

    Returns
    -------
    DataFrame with index=rebalance_date and columns:
        - hold_end
        - gross_return
        - net_return
        - turnover
        - cost_dollars
        - cost_drag
        - value_start
        - value_end
        - n_names
    """
    if bt_cfg is None:
        bt_cfg = BacktestConfig()
    if cost_model is None:
        cost_model = CostModel()

    if daily_returns.empty:
        return pd.DataFrame()

    daily_returns = daily_returns.sort_index()
    idx = daily_returns.index

    if rebalance_dates is None:
        if bt_cfg.rebalance != "quarterly":
            raise ValueError("Only quarterly rebalance is supported in this harness.")
        rdates = _quarter_end_dates(idx)
    else:
        rdates = pd.DatetimeIndex(sorted(pd.to_datetime(list(rebalance_dates))))
        # Align to nearest previous available trading date
        aligned = []
        for d in rdates:
            sub = idx[idx <= d]
            if len(sub) == 0:
                continue
            aligned.append(sub[-1])
        rdates = pd.DatetimeIndex(sorted(set(aligned)))

    if len(rdates) < 2:
        raise ValueError("Need at least 2 rebalance dates to compute a holding period.")

    rows = []
    portfolio_value = float(bt_cfg.initial_value)
    prev_weights: Weights = {}  # assume all cash initially

    for i in range(len(rdates) - 1):
        reb_date = pd.Timestamp(rdates[i])
        hold_end = pd.Timestamp(rdates[i + 1])

        # Target weights at rebalance date
        next_weights = weights_fn(reb_date) or {}

        # Compute holding period returns per ticker
        period_rets = _period_simple_returns(daily_returns, reb_date, hold_end)

        # Optional: drop weights that don't have returns in the hold period
        if bt_cfg.drop_missing_tickers:
            next_weights = _filter_weights_to_available_returns(next_weights, period_rets)

        # Optional: normalize weights back to 1.0
        if bt_cfg.normalize_weights:
            next_weights = _normalize_weights(next_weights)

        gross = _portfolio_return(next_weights, period_rets)

        # Costs are incurred at rebalance as we move prev_weights -> next_weights
        net, turnover, cost_dollars = net_return_after_rebalance(
            gross_return=gross,
            model=cost_model,
            portfolio_value=portfolio_value,
            prev_weights=prev_weights,
            next_weights=next_weights,
        )
        cost_drag = (cost_dollars / portfolio_value) if portfolio_value > 0 else 0.0

        value_start = portfolio_value
        portfolio_value = portfolio_value * (1.0 + net)
        value_end = portfolio_value

        rows.append(
            {
                "rebalance_date": reb_date,
                "hold_end": hold_end,
                "gross_return": gross,
                "net_return": net,
                "turnover": turnover,
                "cost_dollars": cost_dollars,
                "cost_drag": cost_drag,
                "value_start": value_start,
                "value_end": value_end,
                "n_names": len(next_weights),
            }
        )

        prev_weights = next_weights

    df = pd.DataFrame(rows).set_index("rebalance_date")
    return df


# -------------------------
# Example usage (optional)
# -------------------------

def example_weights_fn_factory(weights_by_date: Dict[pd.Timestamp, Weights]) -> Callable[[pd.Timestamp], Weights]:
    """
    Build a weights_fn from a precomputed dict of weights.
    Useful for testing walkforward independent of the model.

    weights_by_date keys should be rebalance dates (aligned to trading dates).
    """
    def _fn(asof: pd.Timestamp) -> Weights:
        return weights_by_date.get(pd.Timestamp(asof), {})
    return _fn