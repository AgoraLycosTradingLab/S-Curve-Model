"""
scurve.backtest.cost

Transaction cost and slippage utilities for Agora Lycos S-Curve backtests.

Design goals:
- Deterministic, simple, and explicit.
- Works with quarterly rebalance portfolios.
- Cost model is expressed in basis points (bps) of traded notional.
- Supports both per-trade bps (entry/exit) and a combined per-turnover cost.

Charter defaults (can be overridden by config):
- Transaction costs: 20 bps per trade
- Slippage: 5 bps per trade
Applied on both entry and exit, via turnover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


def _bps_to_rate(bps: float) -> float:
    """Convert basis points to a decimal rate (e.g., 10 bps -> 0.001)."""
    return float(bps) / 10_000.0


@dataclass(frozen=True)
class CostModel:
    """
    Cost model expressed in bps per trade.

    Interpretation:
    - "per_trade" costs are incurred when notional is traded.
    - When using turnover-based accounting, total traded notional for a rebalance
      is approximated by portfolio turnover (L1 change in weights).

    Fields:
    - cost_bps_per_trade: commissions/fees/impact proxy per trade (bps)
    - slippage_bps_per_trade: additional slippage per trade (bps)
    """
    cost_bps_per_trade: float = 20.0
    slippage_bps_per_trade: float = 5.0

    @property
    def total_bps_per_trade(self) -> float:
        """Total bps per trade (cost + slippage)."""
        return float(self.cost_bps_per_trade + self.slippage_bps_per_trade)

    @property
    def total_rate_per_trade(self) -> float:
        """Total decimal rate per trade."""
        return _bps_to_rate(self.total_bps_per_trade)

    def cost_from_traded_notional(self, traded_notional: float) -> float:
        """
        Compute absolute cost in dollars from traded notional.

        Example:
          traded_notional = $100,000
          total_bps_per_trade = 25 bps = 0.25%
          cost = $250
        """
        traded_notional = float(traded_notional)
        if traded_notional <= 0.0:
            return 0.0
        return traded_notional * self.total_rate_per_trade


def turnover_l1(prev_w: dict[str, float], next_w: dict[str, float]) -> float:
    """
    Compute portfolio turnover as L1 change in weights:
        turnover = sum_i |w_i_new - w_i_old|

    Notes:
    - This is a standard approximation for traded notional as a fraction
      of portfolio value at rebalance time.
    - Weights should sum to ~1.0 (cash can be an explicit key if desired).
    """
    keys = set(prev_w) | set(next_w)
    t = 0.0
    for k in keys:
        t += abs(float(next_w.get(k, 0.0)) - float(prev_w.get(k, 0.0)))
    return float(t)


def traded_notional_from_turnover(
    portfolio_value: float,
    turnover: float,
) -> float:
    """
    Convert turnover fraction to traded notional.

    traded_notional = portfolio_value * turnover

    Example:
      portfolio_value = 1,000,000
      turnover = 0.20  -> traded_notional = 200,000
    """
    portfolio_value = float(portfolio_value)
    turnover = float(turnover)
    if portfolio_value <= 0.0 or turnover <= 0.0:
        return 0.0
    return portfolio_value * turnover


def cost_from_turnover(
    model: CostModel,
    portfolio_value: float,
    turnover: float,
) -> float:
    """
    Compute absolute dollar costs for a rebalance using turnover accounting.

    This treats turnover as the fraction of portfolio value traded.
    Costs are applied on traded notional at total_bps_per_trade.

    Returns:
      dollar_cost (>= 0)
    """
    tn = traded_notional_from_turnover(portfolio_value, turnover)
    return model.cost_from_traded_notional(tn)


def apply_cost_to_return(
    gross_return: float,
    cost_dollars: float,
    portfolio_value: float,
) -> float:
    """
    Convert dollar costs into a drag on return.

    net_return = gross_return - (cost_dollars / portfolio_value)

    If portfolio_value <= 0, returns gross_return unchanged.
    """
    gross_return = float(gross_return)
    cost_dollars = float(cost_dollars)
    portfolio_value = float(portfolio_value)

    if portfolio_value <= 0.0:
        return gross_return
    return gross_return - (cost_dollars / portfolio_value)


def net_return_after_rebalance(
    gross_return: float,
    model: CostModel,
    portfolio_value: float,
    prev_weights: dict[str, float],
    next_weights: dict[str, float],
) -> tuple[float, float, float]:
    """
    Convenience helper for a rebalance step.

    Steps:
    - compute turnover from weight changes
    - compute dollar cost from turnover
    - compute net return after subtracting cost drag

    Returns:
      (net_return, turnover, cost_dollars)
    """
    t = turnover_l1(prev_weights, next_weights)
    c = cost_from_turnover(model, portfolio_value, t)
    net = apply_cost_to_return(gross_return, c, portfolio_value)
    return float(net), float(t), float(c)


# -------------------------
# Simple self-check helpers
# -------------------------

def _approx_equal(a: float, b: float, tol: float = 1e-12) -> bool:
    return abs(float(a) - float(b)) <= tol


def self_test() -> None:
    """
    Lightweight deterministic checks.
    Run manually if desired: python -c "from scurve.backtest.cost import self_test; self_test()"
    """
    cm = CostModel(cost_bps_per_trade=20.0, slippage_bps_per_trade=5.0)
    assert _approx_equal(cm.total_bps_per_trade, 25.0)

    # 10% turnover on $1,000,000 -> $100,000 traded notional
    pv = 1_000_000.0
    t = 0.10
    tn = traded_notional_from_turnover(pv, t)
    assert _approx_equal(tn, 100_000.0)

    # 25 bps of $100,000 = $250
    c = cost_from_turnover(cm, pv, t)
    assert _approx_equal(c, 250.0)

    # If gross return is 1%, cost drag is 0.025% -> net = 0.975%
    gross = 0.01
    net = apply_cost_to_return(gross, c, pv)
    assert _approx_equal(net, 0.01 - 250.0 / 1_000_000.0)

    # Turnover computation check
    prev = {"A": 0.5, "B": 0.5}
    nxt = {"A": 0.6, "B": 0.4}
    assert _approx_equal(turnover_l1(prev, nxt), 0.2)