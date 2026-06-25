"""
Delta-neutral hedge optimizer via CVXPY.

Given a portfolio of option positions (from mispricing signals), computes the
minimum-cost stock trade that brings net delta within a target band.

Sepp (2017) key result: Sharpe as a function of hedge frequency is hump-shaped.
  - Too frequent: transaction costs dominate, Sharpe collapses
  - Too infrequent: large delta drift → losses from directional moves
  - Optimal: ~4-hour intervals for liquid equities (config.HEDGE_INTERVAL_HOURS)

We solve this per rebalance:
    min   c_t * |h_trade|                    (transaction cost of the trade)
    s.t.  |Δ_portfolio + h_new| ≤ δ_band    (keep delta within band)
          h_new = h_current + h_trade        (position update)
          |h_new| ≤ max_hedge_shares         (position limit)

where:
    h_trade      = shares to trade (positive = buy, negative = sell)
    Δ_portfolio  = signed option delta of entire portfolio (in share-equivalents)
    δ_band       = target delta band (default: 0 → exact hedge)
    c_t          = transaction cost per share (bid-ask spread)

Black-Scholes delta is used for hedging (not Heston delta). This is standard
practice — the hedge ratio from BS is robust and requires no ODE solve per step.
Heston delta and BS delta are close for near-ATM options.
"""

import numpy as np
import cvxpy as cp
from dataclasses import dataclass
from scipy.stats import norm
from typing import Optional

import config


@dataclass
class OptionPosition:
    """A single option position in the portfolio."""
    ticker: str
    strike: float
    maturity: float         # years remaining
    option_type: str        # 'C' or 'P'
    qty: float              # signed: positive = long, negative = short
    spot: float
    r: float
    q: float
    implied_vol: float      # market IV used for delta computation


@dataclass
class HedgeResult:
    """Output of the hedge optimizer for one rebalance."""
    portfolio_delta: float      # signed net delta before hedge (share-equivalents)
    current_hedge: float        # current stock hedge position (shares)
    hedge_trade: float          # shares to trade (+ = buy, - = sell)
    new_hedge: float            # resulting stock position after trade
    net_delta_after: float      # residual delta after hedge

    transaction_cost: float     # estimated cost of the trade in USD
    spot: float

    @property
    def hedge_effective(self) -> bool:
        """True if residual delta is within a 1% band."""
        return abs(self.net_delta_after) < 0.01 * abs(self.portfolio_delta + 1e-10)


def bs_delta(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    option_type: str,
) -> float:
    """Black-Scholes delta — ∂C/∂S or ∂P/∂S.

    Sign convention per CLAUDE.md (Carr-Lee):
      Call delta ∈ (0, 1), Put delta ∈ (-1, 0).
    """
    if T <= 1e-6 or sigma <= 1e-8:
        # At expiry: delta is 0 or 1 depending on moneyness
        if option_type == 'C':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    if option_type == 'C':
        return np.exp(-q * T) * norm.cdf(d1)
    else:
        return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def portfolio_delta(positions: list[OptionPosition]) -> float:
    """Net signed delta of a portfolio, in SHARES of the underlying.

    Each option contract controls config.CONTRACT_MULTIPLIER (=100) shares, so the
    share-delta of a position is qty × per_share_delta × multiplier. Returning shares
    (not per-share-delta) is what lets the stock hedge — which trades in shares —
    actually neutralize the book. Omitting the multiplier under-hedged by ~100×.
    """
    total = 0.0
    for pos in positions:
        delta = bs_delta(
            pos.spot, pos.strike, pos.maturity,
            pos.r, pos.q, pos.implied_vol, pos.option_type,
        )
        total += pos.qty * delta * config.CONTRACT_MULTIPLIER

    return total


def optimize_hedge(
    positions: list[OptionPosition],
    current_hedge: float = 0.0,
    delta_band: float = 0.0,
    max_hedge_shares: float = 1e6,
    cost_per_share: Optional[float] = None,
) -> HedgeResult:
    """Find the minimum-cost stock trade to delta-neutralize the portfolio.

    Args:
        positions: current option portfolio.
        current_hedge: existing stock hedge position in shares (signed).
        delta_band: acceptable residual delta range (0 = exact hedge).
            Setting delta_band > 0 reduces unnecessary trading near zero.
        max_hedge_shares: hard position limit on the hedge stock.
        cost_per_share: transaction cost per share traded. Defaults to
            config.EQUITY_BID_ASK_BPS / 10_000 × spot (bid-ask half-spread).

    Returns:
        HedgeResult with hedge_trade, new_hedge, residual delta, cost.
    """
    if not positions:
        return HedgeResult(
            portfolio_delta=0.0,
            current_hedge=current_hedge,
            hedge_trade=0.0,
            new_hedge=current_hedge,
            net_delta_after=current_hedge,
            transaction_cost=0.0,
            spot=0.0,
        )

    spot = positions[0].spot

    if cost_per_share is None:
        cost_per_share = spot * config.EQUITY_BID_ASK_BPS / 10_000.0

    net_delta = portfolio_delta(positions)
    # Net delta of full book including existing hedge
    book_delta = net_delta + current_hedge

    # If already within band, no trade needed
    if abs(book_delta) <= delta_band:
        return HedgeResult(
            portfolio_delta=net_delta,
            current_hedge=current_hedge,
            hedge_trade=0.0,
            new_hedge=current_hedge,
            net_delta_after=book_delta,
            transaction_cost=0.0,
            spot=spot,
        )

    # CVXPY optimization
    # Variables
    h_trade = cp.Variable()          # shares to trade (signed)
    h_new = current_hedge + h_trade  # resulting position

    # Objective: minimize transaction cost
    objective = cp.Minimize(cost_per_share * cp.abs(h_trade))

    # Constraints
    residual_delta = net_delta + h_new   # delta after hedge

    constraints = [
        residual_delta >= -delta_band,
        residual_delta <= delta_band,
        h_new >= -max_hedge_shares,
        h_new <= max_hedge_shares,
    ]

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.HIGHS, verbose=False)
    except Exception:
        prob.solve(solver=cp.ECOS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"] or h_trade.value is None:
        # Fallback: exact hedge ignoring position limits
        exact_trade = -(net_delta + current_hedge)
        trade = float(np.clip(exact_trade, -max_hedge_shares - current_hedge,
                               max_hedge_shares - current_hedge))
        new_h = current_hedge + trade
        return HedgeResult(
            portfolio_delta=net_delta,
            current_hedge=current_hedge,
            hedge_trade=trade,
            new_hedge=new_h,
            net_delta_after=net_delta + new_h,
            transaction_cost=abs(trade) * cost_per_share,
            spot=spot,
        )

    trade = float(h_trade.value)
    new_h = current_hedge + trade

    return HedgeResult(
        portfolio_delta=net_delta,
        current_hedge=current_hedge,
        hedge_trade=trade,
        new_hedge=new_h,
        net_delta_after=float(net_delta + new_h),
        transaction_cost=abs(trade) * cost_per_share,
        spot=spot,
    )


def hedge_report(result: HedgeResult) -> str:
    """Human-readable summary of one hedge rebalance."""
    lines = [
        f"  Portfolio delta:  {result.portfolio_delta:+.4f} shares",
        f"  Current hedge:    {result.current_hedge:+.4f} shares",
        f"  Hedge trade:      {result.hedge_trade:+.4f} shares "
        f"({'buy' if result.hedge_trade > 0 else 'sell' if result.hedge_trade < 0 else 'none'})",
        f"  New hedge:        {result.new_hedge:+.4f} shares",
        f"  Residual delta:   {result.net_delta_after:+.4f} shares",
        f"  Transaction cost: ${result.transaction_cost:.4f}",
        f"  Effective:        {'YES' if result.hedge_effective else 'NO (check position limits)'}",
    ]
    return "\n".join(lines)
