"""
Mathematical invariant checks for every priced surface.

These are structural properties of any no-arbitrage option surface. They must
hold regardless of the calibrated model. If they fail on a given day's data,
that day's signals are quarantined.

Invariants checked:
  1. Put-call parity: C - P = S·e^{-qT} - K·e^{-rT}  (each strike-maturity pair)
  2. Calendar monotonicity: total variance w(k,T) non-decreasing in T
  3. Butterfly: C(K1) - 2C(K2) + C(K3) ≥ 0  (convexity in strike)
  4. Feller condition: 2κθ ≥ ξ² on calibrated params

Run: python -m evals.invariants
"""

import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass

from calibration.heston import HestonParams, price_call, feller_satisfied
from data.cleaner import OptionsChain
import config


# ── Invariant result types ────────────────────────────────────────────────────

@dataclass
class InvariantResult:
    name: str
    passed: bool
    n_violations: int
    max_violation: float
    details: str


# ── 1. Put-call parity ────────────────────────────────────────────────────────

def check_put_call_parity(
    S: float,
    r: float,
    q: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    tol: float = 1e-2,
) -> InvariantResult:
    """Check C - P = F·e^{-rT} - K·e^{-rT} for each paired (K, T).

    tol: allowable violation in USD (mid-price noise + discretisation).
         Default 1 cent — larger than typical 1e-4 because real data has
         bid-ask spread.
    """
    n = len(strikes)
    violations = []
    for i in range(n):
        K = strikes[i]
        T = maturities[i]
        C = call_prices[i]
        P = put_prices[i]
        F = S * np.exp((r - q) * T)
        parity = C - P - (F * np.exp(-r * T) - K * np.exp(-r * T))
        if abs(parity) > tol:
            violations.append(abs(parity))

    max_v = max(violations) if violations else 0.0
    return InvariantResult(
        name="put_call_parity",
        passed=len(violations) == 0,
        n_violations=len(violations),
        max_violation=max_v,
        details=f"{len(violations)}/{n} pairs violate parity by >{tol:.4f}; "
                f"max={max_v:.6f}",
    )


# ── 2. Calendar spread monotonicity ──────────────────────────────────────────

def check_calendar_monotonicity(
    strikes: np.ndarray,
    maturities: np.ndarray,
    total_variances: np.ndarray,
    tol: float = config.BUTTERFLY_TOLERANCE,
) -> InvariantResult:
    """Check w(k, T) non-decreasing in T for each log-moneyness bucket.

    Works on a flat list of (K, T, w) — groups by strike and checks ordering.
    """
    unique_strikes = np.unique(np.round(strikes, 2))
    violations = []
    n_pairs = 0

    for K in unique_strikes:
        mask = np.isclose(strikes, K, rtol=0.002)
        if mask.sum() < 2:
            continue
        T_this = maturities[mask]
        w_this = total_variances[mask]
        order = np.argsort(T_this)
        T_sorted = T_this[order]
        w_sorted = w_this[order]

        for j in range(len(T_sorted) - 1):
            n_pairs += 1
            diff = w_sorted[j + 1] - w_sorted[j]
            if diff < -tol:
                violations.append(abs(diff))

    max_v = max(violations) if violations else 0.0
    return InvariantResult(
        name="calendar_monotonicity",
        passed=len(violations) == 0,
        n_violations=len(violations),
        max_violation=max_v,
        details=f"{len(violations)}/{n_pairs} (K,T) pairs violate calendar; "
                f"max={max_v:.6f}",
    )


# ── 3. Butterfly (call convexity in strike) ───────────────────────────────────

def check_butterfly_convexity(
    strikes: np.ndarray,
    maturities: np.ndarray,
    call_prices: np.ndarray,
    tol: float = config.BUTTERFLY_TOLERANCE,
) -> InvariantResult:
    """Check C(K1) - 2C(K2) + C(K3) ≥ 0 for adjacent strike triples.

    Only checks triples within the same maturity slice.
    """
    unique_mats = np.unique(np.round(maturities, 4))
    violations = []
    n_triples = 0

    for T in unique_mats:
        mask = np.isclose(maturities, T, atol=1e-3)
        Ks = strikes[mask]
        Cs = call_prices[mask]
        order = np.argsort(Ks)
        Ks = Ks[order]
        Cs = Cs[order]

        for j in range(1, len(Ks) - 1):
            K1, K2, K3 = Ks[j-1], Ks[j], Ks[j+1]
            C1, C2, C3 = Cs[j-1], Cs[j], Cs[j+1]
            # Normalized butterfly value
            w1 = K3 - K2
            w2 = K3 - K1
            w3 = K2 - K1
            butterfly = w1 * C1 - w2 * C2 + w3 * C3
            n_triples += 1
            if butterfly < -tol:
                violations.append(abs(butterfly))

    max_v = max(violations) if violations else 0.0
    return InvariantResult(
        name="butterfly_convexity",
        passed=len(violations) == 0,
        n_violations=len(violations),
        max_violation=max_v,
        details=f"{len(violations)}/{n_triples} triples violate butterfly; "
                f"max={max_v:.6f}",
    )


# ── 4. Feller condition ───────────────────────────────────────────────────────

def check_feller(params: HestonParams) -> InvariantResult:
    """2κθ ≥ ξ² must hold for calibrated parameters."""
    ok = feller_satisfied(params)
    lhs = 2 * params.kappa * params.theta
    rhs = params.xi ** 2
    violation = max(0.0, rhs - lhs)
    return InvariantResult(
        name="feller_condition",
        passed=ok,
        n_violations=0 if ok else 1,
        max_violation=violation,
        details=f"2kth={lhs:.6f}  xi2={rhs:.6f}  "
                f"margin={lhs - rhs:+.6f}  {'OK' if ok else 'VIOLATED'}",
    )


# ── 5. Heston self-consistency (prices back to IVs) ──────────────────────────

def check_pricing_self_consistency(
    S: float,
    r: float,
    q: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    params: HestonParams,
    iv_tol: float = 1e-3,
) -> InvariantResult:
    """Price calls with Heston, extract IVs, verify they round-trip cleanly.

    For each (K, T): price_call → IV → price_call must match to iv_tol.
    This catches numerical issues in the pricer itself.
    """
    from calibration.heston import heston_implied_vols
    from scipy.stats import norm

    def bs_call(S, K, T, r, q, sigma):
        if sigma < 1e-8 or T < 1e-6:
            return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

    ivs = np.array(heston_implied_vols(
        S, jnp.array(strikes), jnp.array(maturities), r, q, params
    ))

    violations = []
    n = 0
    for i in range(len(strikes)):
        if np.isnan(ivs[i]):
            continue
        n += 1
        heston_price = float(price_call(S, float(strikes[i]), float(maturities[i]),
                                         r, q, params))
        bs_price = bs_call(S, float(strikes[i]), float(maturities[i]), r, q, float(ivs[i]))
        diff = abs(heston_price - bs_price)
        if diff > iv_tol:
            violations.append(diff)

    max_v = max(violations) if violations else 0.0
    return InvariantResult(
        name="pricing_self_consistency",
        passed=len(violations) == 0,
        n_violations=len(violations),
        max_violation=max_v,
        details=f"{len(violations)}/{n} options fail round-trip; max diff={max_v:.6f}",
    )


# ── Runner ────────────────────────────────────────────────────────────────────

def run_invariant_suite(
    S: float,
    r: float,
    q: float,
    chain: OptionsChain,
    call_prices: np.ndarray,
    put_prices: np.ndarray,
    total_variances: np.ndarray,
    params: HestonParams,
) -> list[InvariantResult]:
    """Run all invariant checks and return results."""
    results = []

    if len(call_prices) == len(put_prices):
        results.append(check_put_call_parity(
            S, r, q, chain.strikes, chain.maturities, call_prices, put_prices
        ))

    results.append(check_calendar_monotonicity(
        chain.strikes, chain.maturities, total_variances
    ))

    results.append(check_butterfly_convexity(
        chain.strikes, chain.maturities, call_prices
    ))

    results.append(check_feller(params))

    results.append(check_pricing_self_consistency(
        S, r, q, chain.strikes[:20], chain.maturities[:20], params
    ))

    return results


# ── Standalone test runner ────────────────────────────────────────────────────

if __name__ == "__main__":
    import jax.numpy as jnp
    from calibration.heston import HestonParams, heston_implied_vols, price_call
    from data.cleaner import OptionsChain

    print("=" * 60)
    print("INVARIANT CHECKS — HESTON SYNTHETIC SURFACE")
    print("=" * 60)

    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    S, r, q = 100.0, 0.05, 0.02

    ttms = np.array([3/12, 6/12, 1.0, 1.5])
    strikes_1d = np.array([80.0, 90.0, 95.0, 100.0, 105.0, 110.0, 120.0])

    pairs = [(K, T) for T in ttms for K in strikes_1d]
    strikes = np.array([p[0] for p in pairs])
    maturities = np.array([p[1] for p in pairs])

    # Price calls
    call_prices = np.array([
        float(price_call(S, K, T, r, q, params))
        for K, T in zip(strikes, maturities)
    ])

    # Synthetic put prices via put-call parity (no independent put pricer needed)
    F_arr = S * np.exp((r - q) * maturities)
    put_prices = call_prices - F_arr * np.exp(-r * maturities) + strikes * np.exp(-r * maturities)

    # Total variance from IVs
    ivs = np.array(heston_implied_vols(S, jnp.array(strikes), jnp.array(maturities), r, q, params))
    total_variances = np.where(~np.isnan(ivs), ivs**2 * maturities, np.nan)

    chain = OptionsChain(
        ticker="SYNTH", snapshot_time="2026-06-21T00:00:00Z",
        spot=S, r=r, q=q,
        strikes=strikes, maturities=maturities,
        mid_prices=call_prices, bid_prices=call_prices * 0.99,
        ask_prices=call_prices * 1.01, open_interest=np.ones(len(strikes)) * 100,
        option_type=np.array(["C"] * len(strikes)),
    )

    results = run_invariant_suite(S, r, q, chain, call_prices, put_prices,
                                   total_variances, params)

    all_passed = True
    for res in results:
        status = "PASS" if res.passed else "FAIL"
        print(f"\n  [{status}] {res.name}")
        print(f"         {res.details}")
        if not res.passed:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("SUMMARY: All invariants hold on synthetic surface.")
    else:
        failed = [r.name for r in results if not r.passed]
        print(f"SUMMARY: FAILED — {failed}")
