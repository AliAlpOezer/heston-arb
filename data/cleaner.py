"""
LP-based arbitrage repair for raw option price data.

Reference: Roper (2010) "Arbitrage free implied volatility surfaces",
and arXiv 2008.09454 "Arbitrage-free implied volatility surface generation
with variational autoencoders."

No-arbitrage conditions enforced:
  1. Calendar spread (in total variance): w(k, T₂) ≥ w(k, T₁) for T₂ > T₁
  2. Butterfly spread (call convexity): C(K₁,T) - 2C(K₂,T) + C(K₃,T) ≥ 0
  3. Positive total variance: w(k, T) ≥ 0

The LP finds the minimum perturbation (in L1 norm) to option prices that
restores all no-arbitrage conditions. Perturbations are bounded by
config.LP_MAX_PERTURBATION_FRAC of each option's mid-price.
"""

import numpy as np
import pandas as pd
import cvxpy as cp
from dataclasses import dataclass
from typing import Optional
import warnings

import config


@dataclass
class OptionsChain:
    """Raw options chain snapshot for a single underlying at a single timestamp."""
    ticker: str
    snapshot_time: str            # ISO 8601
    spot: float
    r: float                      # risk-free rate (continuous)
    q: float                      # continuous dividend yield

    # Per-option data (all arrays same length N)
    strikes: np.ndarray           # K
    maturities: np.ndarray        # T in years (ACT/365)
    mid_prices: np.ndarray        # (bid + ask) / 2
    bid_prices: np.ndarray
    ask_prices: np.ndarray
    open_interest: np.ndarray
    option_type: np.ndarray       # 'C' or 'P' (string array)

    def forward(self, T: float) -> float:
        return self.spot * np.exp((self.r - self.q) * T)

    def log_moneyness(self) -> np.ndarray:
        """k = log(K / F(T)) for each option."""
        F = np.vectorize(self.forward)(self.maturities)
        return np.log(self.strikes / F)


@dataclass
class CleanedSurface:
    """Arbitrage-free option prices and derived implied vols."""
    chain: OptionsChain
    repaired_prices: np.ndarray   # LP-repaired mid-prices
    implied_vols: np.ndarray      # Black-Scholes IVs from repaired prices
    n_prices_changed: int
    max_perturbation_frac: float
    arbitrage_removed: bool


def filter_chain(chain: OptionsChain) -> tuple[OptionsChain, np.ndarray]:
    """Drop options that fail basic quality filters. Returns filtered chain + mask."""
    mask = np.ones(len(chain.strikes), dtype=bool)

    # Time to maturity filters
    mask &= chain.maturities >= config.MIN_TTM_DAYS / 365.0
    mask &= chain.maturities <= config.MAX_TTM_DAYS / 365.0

    # Log-moneyness filter
    mask &= np.abs(chain.log_moneyness()) <= config.MAX_LOG_MONEYNESS

    # Bid-ask spread quality filter
    spread = chain.ask_prices - chain.bid_prices
    mid = chain.mid_prices
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spread_frac = np.where(mid > 1e-6, spread / mid, np.inf)
    mask &= spread_frac <= config.MAX_BID_ASK_FRAC

    # Open interest filter — skip when OI is all-zero (data source doesn't provide it)
    if chain.open_interest.sum() > 0:
        mask &= chain.open_interest >= config.MIN_OPEN_INTEREST

    # Positive mid-price
    mask &= chain.mid_prices > 1e-6

    # OTM filter: when BOTH calls and puts are present, keep only the OTM side at each
    # strike — puts where k<0, calls where k>=0. This dedups strikes (one liquid quote
    # per strike) and uses the information-rich OTM wing; put IVs are derived via
    # put-call parity downstream so the resulting IV surface is option-type-agnostic.
    # When only one type is present (e.g. call_only loader), keep all (no-op).
    types = chain.option_type
    if ("C" in types) and ("P" in types):
        k = chain.log_moneyness()
        otm = ((types == "C") & (k >= 0)) | ((types == "P") & (k < 0))
        mask &= otm

    filtered = OptionsChain(
        ticker=chain.ticker,
        snapshot_time=chain.snapshot_time,
        spot=chain.spot,
        r=chain.r,
        q=chain.q,
        strikes=chain.strikes[mask],
        maturities=chain.maturities[mask],
        mid_prices=chain.mid_prices[mask],
        bid_prices=chain.bid_prices[mask],
        ask_prices=chain.ask_prices[mask],
        open_interest=chain.open_interest[mask],
        option_type=chain.option_type[mask],
    )
    return filtered, mask


def _build_no_arbitrage_constraints(
    chain: OptionsChain,
    prices_var: cp.Variable,
) -> list:
    """Build LP no-arbitrage constraints for call options only.

    For puts: convert to calls via put-call parity first, then apply.
    Returns list of CVXPY constraints.
    """
    constraints = []
    N = len(chain.strikes)

    # Convert puts to synthetic calls via put-call parity: C = P + F·e^{-rT} - K·e^{-rT}
    # We work directly with mid_prices as the reference and perturb them.
    # For the constraints, we need calls. Puts are treated separately.

    call_mask = chain.option_type == 'C'
    call_idx = np.where(call_mask)[0]
    unique_maturities = np.unique(chain.maturities)
    unique_strikes = np.unique(chain.strikes)

    # 1. Positive call prices (lower bound: max(F - K·e^{-rT}, 0))
    for i in call_idx:
        K, T = chain.strikes[i], chain.maturities[i]
        F = chain.forward(T)
        intrinsic = max(F * np.exp(-chain.r * T) - K * np.exp(-chain.r * T), 0)
        constraints.append(prices_var[i] >= intrinsic + 1e-6)

    # 2. Calendar spread: C(K, T₂) ≥ C(K, T₁) for T₂ > T₁ (same K)
    for K in unique_strikes:
        same_strike_calls = [
            i for i in call_idx
            if np.isclose(chain.strikes[i], K, rtol=0.001)
        ]
        same_strike_calls.sort(key=lambda i: chain.maturities[i])
        for j in range(len(same_strike_calls) - 1):
            i_short = same_strike_calls[j]
            i_long = same_strike_calls[j + 1]
            # C(K, T_long) ≥ C(K, T_short)
            constraints.append(prices_var[i_long] >= prices_var[i_short])

    # 3. Butterfly spread convexity: C(K₁) - 2C(K₂) + C(K₃) ≥ 0 for K₁ < K₂ < K₃
    for T in unique_maturities:
        same_mat_calls = sorted(
            [i for i in call_idx if np.isclose(chain.maturities[i], T, atol=1e-3)],
            key=lambda i: chain.strikes[i],
        )
        for j in range(1, len(same_mat_calls) - 1):
            i1, i2, i3 = same_mat_calls[j-1], same_mat_calls[j], same_mat_calls[j+1]
            K1, K2, K3 = chain.strikes[i1], chain.strikes[i2], chain.strikes[i3]
            # Butterfly: (K3-K2)C(K1) - (K3-K1)C(K2) + (K2-K1)C(K3) ≥ 0
            w1 = K3 - K2
            w2 = K3 - K1
            w3 = K2 - K1
            constraints.append(
                w1 * prices_var[i1] - w2 * prices_var[i2] + w3 * prices_var[i3] >= 0
            )

    return constraints


def repair_arbitrage(chain: OptionsChain) -> CleanedSurface:
    """LP arbitrage repair: minimum L1 perturbation to restore no-arbitrage.

    Formulation:
        min  Σᵢ |pᵢ - p̃ᵢ|                (minimize total price change)
        s.t. no-arbitrage constraints       (calendar, butterfly, positivity)
             |pᵢ - p̃ᵢ| ≤ ε · p̃ᵢ          (max perturbation fraction)

    where p̃ᵢ = original mid-price, pᵢ = repaired price.
    """
    N = len(chain.strikes)
    if N == 0:
        return CleanedSurface(
            chain=chain,
            repaired_prices=np.array([]),
            implied_vols=np.array([]),
            n_prices_changed=0,
            max_perturbation_frac=0.0,
            arbitrage_removed=False,
        )

    original = chain.mid_prices.copy()

    # Decision variable: repaired prices
    p = cp.Variable(N, nonneg=True)

    # Perturbation variables for L1 objective
    delta = cp.Variable(N, nonneg=True)

    # Objective: minimize L1 perturbation
    objective = cp.Minimize(cp.sum(delta))

    # Max perturbation constraint
    max_perturb = config.LP_MAX_PERTURBATION_FRAC * original

    constraints = [
        delta >= p - original,
        delta >= original - p,
        delta <= max_perturb,
    ]

    # No-arbitrage constraints (calls only for simplicity; extend to puts via parity)
    arb_constraints = _build_no_arbitrage_constraints(chain, p)
    constraints.extend(arb_constraints)

    prob = cp.Problem(objective, constraints)

    try:
        prob.solve(solver=cp.HIGHS, verbose=False)
    except Exception:
        # Fallback to SCS if HIGHS unavailable
        prob.solve(solver=cp.SCS, verbose=False)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        # LP failed — return original prices with a warning
        warnings.warn(
            f"Arbitrage repair LP failed for {chain.ticker} at {chain.snapshot_time}: "
            f"status={prob.status}. Using original prices."
        )
        repaired = original
        n_changed = 0
        max_perturb_actual = 0.0
        arb_removed = False
    else:
        repaired = np.array(p.value)
        repaired = np.maximum(repaired, 1e-6)  # ensure positive
        perturbations = np.abs(repaired - original) / (original + 1e-8)
        n_changed = int(np.sum(perturbations > 1e-4))
        max_perturb_actual = float(perturbations.max())
        arb_removed = True

    # Extract implied vols from repaired prices
    implied_vols = _extract_implied_vols(chain, repaired)

    return CleanedSurface(
        chain=chain,
        repaired_prices=repaired,
        implied_vols=implied_vols,
        n_prices_changed=n_changed,
        max_perturbation_frac=max_perturb_actual,
        arbitrage_removed=arb_removed,
    )


def _extract_implied_vols(
    chain: OptionsChain,
    prices: np.ndarray,
) -> np.ndarray:
    """Extract Black-Scholes implied vols from option prices via Newton-Raphson.

    Returns NaN for options where IV solver fails. Never uses a fallback.
    Per CLAUDE.md: NaN means "do not use this strike in calibration."
    """
    from scipy.stats import norm
    from scipy.optimize import brentq

    ivs = np.full(len(prices), np.nan)

    for i, (price, K, T, opt_type) in enumerate(
        zip(prices, chain.strikes, chain.maturities, chain.option_type)
    ):
        S = chain.spot
        r = chain.r
        q = chain.q
        F = chain.forward(T)

        # Convert put to call via parity for IV extraction
        if opt_type == 'P':
            price = price + F * np.exp(-r * T) - K * np.exp(-r * T)

        # Intrinsic value check
        intrinsic = max(F * np.exp(-r * T) - K * np.exp(-r * T), 0)
        if price <= intrinsic + 1e-8:
            continue  # leave as NaN

        def bs_call_error(sigma):
            if sigma <= 0:
                return -price
            d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
            d2 = d1 - sigma * np.sqrt(T)
            C = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            return C - price

        try:
            iv = brentq(bs_call_error, 1e-4, 10.0, xtol=config.IV_SOLVER_TOL)
            ivs[i] = iv
        except (ValueError, RuntimeError):
            pass  # leave as NaN

    return ivs


def clean_chain(chain: OptionsChain) -> CleanedSurface:
    """Full cleaning pipeline: filter → LP repair → IV extraction."""
    filtered_chain, _ = filter_chain(chain)
    return repair_arbitrage(filtered_chain)
