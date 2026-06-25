"""
Synthetic Heston surface generator and parameter recovery tests.

Popper rule: before the calibrator touches real data, it must recover known
parameters from surfaces it generated itself. If it cannot do that, it cannot
do anything useful in production.

Run directly: python -m evals.synthetic
"""

import jax.numpy as jnp
import numpy as np
import pytest

from calibration.heston import (
    HestonParams,
    heston_implied_vols,
    feller_satisfied,
    feller_penalty,
)
from calibration.calibrator import (
    CalibrationInput,
    compute_weights,
    calibrate,
)
import config


# ── Known-parameter test cases ────────────────────────────────────────────────
# Each tuple: (label, HestonParams, description)
# All test params must satisfy Feller (2κθ ≥ ξ²). If they violate it, the
# calibrator's Feller penalty pushes fitted params away from the true values,
# making recovery tests structurally impossible to pass.
#
# Feller check per case:
#   equity_typical:       2*2.0*0.04=0.16 ≥ 0.3²=0.09  ✓
#   high_vol_regime:      2*4.0*0.09=0.72 ≥ 0.8²=0.64  ✓
#   low_vol_regime:       2*1.0*0.01=0.02 ≥ 0.1²=0.01  ✓
#   near_feller_boundary: 2*1.0*0.05=0.10 ≥ 0.31²=0.096 ✓
TEST_PARAMS = [
    (
        "equity_typical",
        HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04),
        "Typical equity: 20% LR vol, negative skew, moderate mean reversion",
    ),
    (
        "high_vol_regime",
        HestonParams(kappa=4.0, theta=0.09, xi=0.8, rho=-0.5, v0=0.16),
        "High vol regime: 30% LR vol, ~40% spot vol, fast mean reversion",
    ),
    (
        "low_vol_regime",
        HestonParams(kappa=1.0, theta=0.01, xi=0.1, rho=-0.8, v0=0.01),
        "Low vol regime: 10% LR vol, slow mean reversion, steep skew",
    ),
    (
        "near_feller_boundary",
        HestonParams(kappa=1.0, theta=0.05, xi=0.31, rho=-0.3, v0=0.05),
        "Near Feller boundary: 2κθ=0.10, ξ²=0.096 — just satisfies condition",
    ),
]


def make_synthetic_surface(
    true_params: HestonParams,
    S: float = 100.0,
    r: float = 0.05,
    q: float = 0.02,
    add_noise_vol: float = 0.005,
    seed: int = 42,
) -> CalibrationInput:
    """Generate a synthetic implied vol surface from known Heston parameters.

    Covers a realistic grid of strikes and maturities. Optionally adds
    small Gaussian noise to simulate market bid-ask uncertainty.

    Returns CalibrationInput ready for calibrate().
    """
    rng = np.random.default_rng(seed)

    # Realistic strike and maturity grid
    ttms_years = np.array([1/12, 2/12, 3/12, 6/12, 1.0, 1.5, 2.0])
    log_moneyness = np.linspace(-0.3, 0.3, 9)  # ±30% log-moneyness

    # Build paired (K, T) arrays (not a meshgrid — flat pairs)
    pairs = [
        (S * np.exp(k), T)
        for T in ttms_years
        for k in log_moneyness
    ]
    strikes = jnp.array([p[0] for p in pairs])
    maturities = jnp.array([p[1] for p in pairs])

    # Compute true model IVs
    true_ivs = heston_implied_vols(S, strikes, maturities, r, q, true_params)

    # Add noise to simulate market bid-ask
    if add_noise_vol > 0.0:
        noise = rng.normal(0, add_noise_vol, size=len(pairs))
        noisy_ivs = jnp.array(np.array(true_ivs) + noise)
        noisy_ivs = jnp.clip(noisy_ivs, 0.01, 5.0)
    else:
        noisy_ivs = true_ivs

    # Drop any NaN IVs (failed pricing for extreme parameters)
    valid_mask = ~jnp.isnan(noisy_ivs) & ~jnp.isnan(true_ivs)
    strikes = strikes[valid_mask]
    maturities = maturities[valid_mask]
    noisy_ivs = noisy_ivs[valid_mask]

    # Uniform weights (synthetic data has no bid-ask spread or OI)
    weights = jnp.ones(len(strikes)) / len(strikes)

    return CalibrationInput(
        strikes=strikes,
        maturities=maturities,
        market_ivs=noisy_ivs,
        weights=weights,
        S=S,
        r=r,
        q=q,
    )


# ── Parameter recovery test ───────────────────────────────────────────────────

def run_recovery_test(
    label: str,
    true_params: HestonParams,
    description: str,
    n_steps: int = 2000,
    add_noise: bool = False,
    verbose: bool = False,
) -> dict:
    """Run calibration on a synthetic surface and report recovery accuracy."""
    print(f"\n{'-'*60}")
    print(f"Test: {label}")
    print(f"  {description}")
    print(f"  True params: κ={true_params.kappa:.3f} θ={true_params.theta:.4f} "
          f"ξ={true_params.xi:.3f} ρ={true_params.rho:.3f} v₀={true_params.v0:.4f}")
    print(f"  Feller satisfied: {feller_satisfied(true_params)} "
          f"(2κθ={2*true_params.kappa*true_params.theta:.4f} ξ²={true_params.xi**2:.4f})")

    cal = make_synthetic_surface(
        true_params,
        add_noise_vol=0.005 if add_noise else 0.0,
    )

    # Perturbed starting point (simulate cold start, not warm start)
    init = HestonParams(
        kappa=true_params.kappa * 1.5,
        theta=true_params.theta * 1.3,
        xi=true_params.xi * 0.8,
        rho=true_params.rho * 0.7,
        v0=true_params.v0 * 1.2,
    )

    fitted, final_loss = calibrate(
        cal,
        initial_params=init,
        prev_params=None,
        n_steps=n_steps,
        verbose=verbose,
    )

    # Compute relative errors
    errs = {
        "kappa": abs(fitted.kappa - true_params.kappa) / true_params.kappa,
        "theta": abs(fitted.theta - true_params.theta) / true_params.theta,
        "xi":    abs(fitted.xi    - true_params.xi)    / true_params.xi,
        "rho":   abs(fitted.rho   - true_params.rho)   / max(abs(true_params.rho), 0.01),
        "v0":    abs(fitted.v0    - true_params.v0)    / true_params.v0,
    }

    passed = {
        "kappa": errs["kappa"] < 0.10,
        "theta": errs["theta"] < 0.05,
        "xi":    errs["xi"]    < 0.10,
        "rho":   errs["rho"]   < 0.10,
        "v0":    errs["v0"]    < 0.05,
    }
    feller_ok = feller_satisfied(fitted)
    all_passed = all(passed.values()) and feller_ok

    print(f"  Fitted:      κ={fitted.kappa:.3f} θ={fitted.theta:.4f} "
          f"ξ={fitted.xi:.3f} ρ={fitted.rho:.3f} v₀={fitted.v0:.4f}")
    print(f"  Rel errors:  κ={errs['kappa']:.1%} θ={errs['theta']:.1%} "
          f"ξ={errs['xi']:.1%} ρ={errs['rho']:.1%} v₀={errs['v0']:.1%}")
    print(f"  Feller OK:   {feller_ok}")
    print(f"  Final loss:  {final_loss:.6f}")
    print(f"  RESULT: {'PASS' if all_passed else 'FAIL'}")

    if not all_passed:
        failed = [k for k, v in passed.items() if not v]
        if not feller_ok:
            failed.append("feller")
        print(f"  Failed:      {failed}")

    return {
        "label": label,
        "true": true_params,
        "fitted": fitted,
        "errors": errs,
        "feller_ok": feller_ok,
        "final_loss": final_loss,
        "passed": all_passed,
    }


# ── pytest wrappers ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("label,params,desc", TEST_PARAMS)
def test_parameter_recovery_noiseless(label, params, desc):
    """Calibration must recover known parameters from its own noiseless surface."""
    result = run_recovery_test(label, params, desc, add_noise=False)
    assert result["passed"], (
        f"Parameter recovery failed for '{label}': "
        f"errors={result['errors']}, feller={result['feller_ok']}"
    )


@pytest.mark.parametrize("label,params,desc", TEST_PARAMS)
def test_parameter_recovery_noisy(label, params, desc):
    """Calibration must recover known parameters from surface with 0.5% IV noise."""
    result = run_recovery_test(label, params, desc, add_noise=True)
    # Noisy test has relaxed tolerances — Feller must hold, and major params must be close
    assert result["feller_ok"], f"Feller condition violated after calibration for '{label}'"
    assert result["errors"]["theta"] < 0.10, f"Theta error too large: {result['errors']['theta']:.1%}"
    assert result["errors"]["v0"] < 0.10, f"v0 error too large: {result['errors']['v0']:.1%}"


def test_feller_condition_always_satisfied():
    """Calibrator must never return parameters violating the Feller condition."""
    for label, params, desc in TEST_PARAMS:
        result = run_recovery_test(label, params, desc, add_noise=True)
        assert feller_satisfied(result["fitted"]), f"Feller violated for '{label}'"


# ── Standalone runner ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("HESTON CALIBRATION — SYNTHETIC PARAMETER RECOVERY TESTS")
    print("(Popper falsification: calibrator must recover its own data)")
    print("=" * 60)

    results = []
    for label, params, desc in TEST_PARAMS:
        r = run_recovery_test(label, params, desc, add_noise=False, verbose=False)
        results.append(r)

    print(f"\n{'='*60}")
    passed = sum(1 for r in results if r["passed"])
    print(f"SUMMARY: {passed}/{len(results)} tests passed")

    if passed < len(results):
        print("\n[!] CALIBRATION IS NOT READY FOR PRODUCTION DATA")
        print("   Fix parameter recovery before touching any real options chain.")
    else:
        print("\n[OK] Calibrator passes Popper falsification. Proceed to data cleaning.")
