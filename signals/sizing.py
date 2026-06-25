"""
Position sizing: Kelly criterion on vol gap with portfolio contract limits.

Kelly fraction for a vol-arb trade, in contract units:
    f* = (gap / iv_vol) * (1 / halflife_days) / iv_vol^2
       = gap / (iv_vol^2 * halflife_days)

where:
  gap         = |model_iv - market_iv| (absolute vol points)
  iv_vol      = daily standard deviation of IV (assumed constant, IV_DAILY_VOL)
  halflife    = expected days for gap to close (mean-reversion rate)

Simplifies to: f* = gap / (IV_DAILY_VOL^2 * halflife)

This is in "vol-point units." We convert to contracts via:
  qty = round(f* * KELLY_FRACTION) clipped to [1, MAX_QTY_PER_POSITION]

Portfolio caps: maximum total open contracts across all positions.

Units: everything is in fraction-of-IV space (no dollar normalization needed).
The dollar value of a position = qty * contract_multiplier * vega * IV_move.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm

import config
from signals.mispricing import MispricingSignal


# ── Config ────────────────────────────────────────────────────────────────────

KELLY_FRACTION = 0.25           # use 25% Kelly to be conservative
MAX_PORTFOLIO_CONTRACTS = 50    # total open contracts across all positions
MAX_QTY_PER_POSITION = 10       # hard cap per individual signal
IV_DAILY_VOL = 0.02             # assumed daily vol of implied vol (2 vol pts/day)
MEAN_REVERSION_HALFLIFE_DAYS = 5  # expected trading days for gap to close


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class SizingInput:
    """Market and position data needed to compute Kelly size."""
    signal: MispricingSignal
    spot: float
    r: float
    q: float

    # Current portfolio state (sum across all open positions)
    portfolio_contracts: int = 0


@dataclass
class SizingResult:
    """Recommended position size with Kelly derivation."""
    signal: MispricingSignal
    qty: int                    # recommended number of contracts (0 = do not enter)

    # Greeks (informational — for risk reporting only, not used in sizing)
    bs_vega: float              # per-contract vega ($/sigma per share)
    bs_gamma: float             # per-contract gamma (d^2C/dS^2 per share)

    # Kelly derivation intermediates
    kelly_raw: float            # raw Kelly qty before KELLY_FRACTION
    kelly_fractional: float     # after KELLY_FRACTION multiplier
    contract_cap_qty: int       # max qty from portfolio contract limit

    reason: str                 # "entered" or why rejected


# ── BS greeks ─────────────────────────────────────────────────────────────────

def _bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """BS vega: dC/dsigma ($ per unit sigma change per share). Returns 0 for bad inputs."""
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


def _bs_gamma(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """BS gamma: d^2C/dS^2 (per share). Returns 0 for bad inputs."""
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)))


# ── Kelly sizing ──────────────────────────────────────────────────────────────

def kelly_size(inp: SizingInput) -> SizingResult:
    """Compute Kelly-optimal position size for a vol-arb signal.

    Kelly criterion (in contract-count units):
      f* = gap / (IV_DAILY_VOL^2 * halflife_days)

    Intuition: larger gap -> bigger bet; larger IV vol or longer halflife -> smaller bet.
    The KELLY_FRACTION multiplier cuts f* by 75% for safety margin.
    """
    sig = inp.signal
    S = inp.spot
    K = sig.strike
    T = sig.maturity
    iv = sig.market_iv

    vega = _bs_vega(S, K, T, inp.r, inp.q, iv)
    gamma = _bs_gamma(S, K, T, inp.r, inp.q, iv)

    gap = abs(sig.vol_gap)

    # Kelly qty in contract units
    raw_kelly = gap / (IV_DAILY_VOL**2 * MEAN_REVERSION_HALFLIFE_DAYS)
    fractional = raw_kelly * KELLY_FRACTION

    # Portfolio contract cap
    remaining_contracts = max(0, MAX_PORTFOLIO_CONTRACTS - inp.portfolio_contracts)
    contract_cap_qty = min(remaining_contracts, MAX_QTY_PER_POSITION)

    qty = int(min(max(1, round(fractional)), contract_cap_qty))

    if contract_cap_qty == 0:
        qty = 0
        reason = "portfolio contract limit reached"
    else:
        reason = "entered"

    return SizingResult(
        signal=sig,
        qty=qty,
        bs_vega=vega,
        bs_gamma=gamma,
        kelly_raw=float(raw_kelly),
        kelly_fractional=float(fractional),
        contract_cap_qty=contract_cap_qty,
        reason=reason,
    )


def size_portfolio(
    signals: list[MispricingSignal],
    spot: float,
    r: float,
    q: float,
    max_new_positions: int = 5,
) -> list[SizingResult]:
    """Size a full set of signals, enforcing portfolio contract limit.

    Processes signals in order (assumed sorted by |gap| descending by detect_mispricings).
    Accumulates contract count greedily — largest mispricings get priority.
    """
    results = []
    portfolio_contracts = 0
    n_entered = 0

    for sig in signals:
        if n_entered >= max_new_positions:
            results.append(SizingResult(
                signal=sig, qty=0, bs_vega=0.0, bs_gamma=0.0,
                kelly_raw=0.0, kelly_fractional=0.0, contract_cap_qty=0,
                reason="max_new_positions reached",
            ))
            continue

        inp = SizingInput(
            signal=sig, spot=spot, r=r, q=q,
            portfolio_contracts=portfolio_contracts,
        )
        result = kelly_size(inp)
        results.append(result)

        if result.qty > 0:
            portfolio_contracts += result.qty
            n_entered += 1

    return results


def sizing_report(results: list[SizingResult]) -> str:
    """Human-readable table of sizing decisions."""
    entered = [r for r in results if r.qty > 0]
    skipped = [r for r in results if r.qty == 0]

    lines = [
        f"{'='*70}",
        f"POSITION SIZING ({len(entered)} entered, {len(skipped)} skipped)",
        f"{'='*70}",
        f"  {'K':>8} {'T':>6} {'dir':>5} {'gap':>8} {'kelly':>7} {'qty':>5}  reason",
        f"  {'-'*8} {'-'*6} {'-'*5} {'-'*8} {'-'*7} {'-'*5}  {'-'*20}",
    ]
    for r in results:
        s = r.signal
        lines.append(
            f"  {s.strike:8.1f} {s.maturity:6.3f} {s.direction:>5} "
            f"{s.vol_gap:+8.4f} {r.kelly_fractional:7.2f} "
            f"{r.qty:5d}  {r.reason}"
        )

    if entered:
        total_contracts = sum(r.qty for r in entered)
        total_vega = sum(r.qty * r.bs_vega for r in entered)
        lines += [
            f"  {'-'*60}",
            f"  Total contracts: {total_contracts}  (cap: {MAX_PORTFOLIO_CONTRACTS})",
            f"  Total vega ($/sigma/share): {total_vega:.1f}",
        ]
    return "\n".join(lines)
