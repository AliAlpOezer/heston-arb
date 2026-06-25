"""
Integration test: VolSurface + mispricing signal detection.

Generates a synthetic Heston surface, builds a VolSurface from it,
then injects a deliberate model with different parameters and checks
that the resulting gap is detected correctly.

Run: python -m evals.test_surface_mispricing
"""

import numpy as np
import jax.numpy as jnp

from calibration.heston import HestonParams, heston_implied_vols
from calibration.calibrator import CalibrationInput, calibrate
from data.cleaner import OptionsChain, CleanedSurface
from data.surface import VolSurface, build_surface
from signals.mispricing import detect_mispricings, signal_report
import config


def _make_synthetic_chain(params: HestonParams, S=100.0, r=0.05, q=0.02) -> CleanedSurface:
    """Build a fake CleanedSurface from known Heston parameters."""
    ttms = np.array([1/12, 3/12, 6/12, 1.0, 1.5])
    log_moneyness_grid = np.linspace(-0.25, 0.25, 7)

    pairs = [(S * np.exp(k), T) for T in ttms for k in log_moneyness_grid]
    strikes = np.array([p[0] for p in pairs])
    maturities = np.array([p[1] for p in pairs])

    ivs_jax = heston_implied_vols(S, jnp.array(strikes), jnp.array(maturities), r, q, params)
    ivs = np.array(ivs_jax)

    valid = ~np.isnan(ivs) & (ivs > 0)
    strikes = strikes[valid]
    maturities = maturities[valid]
    ivs = ivs[valid]

    chain = OptionsChain(
        ticker="SYNTH",
        snapshot_time="2026-06-21T00:00:00Z",
        spot=S,
        r=r,
        q=q,
        strikes=strikes,
        maturities=maturities,
        mid_prices=np.zeros(len(strikes)),      # not used downstream
        bid_prices=np.zeros(len(strikes)),
        ask_prices=np.zeros(len(strikes)),
        open_interest=np.ones(len(strikes)) * 100,
        option_type=np.array(["C"] * len(strikes)),
    )

    return CleanedSurface(
        chain=chain,
        repaired_prices=chain.mid_prices,
        implied_vols=ivs,
        n_prices_changed=0,
        max_perturbation_frac=0.0,
        arbitrage_removed=True,
    )


def test_surface_builds_and_interpolates():
    """VolSurface must build and return finite IVs near the grid points."""
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    cleaned = _make_synthetic_chain(params)
    surf = build_surface(cleaned)

    assert surf._spline is not None, "Spline must build successfully"
    assert surf.n_options > 10, f"Expected >10 options, got {surf.n_options}"
    assert not np.isnan(surf.atm_iv), f"ATM IV is NaN"
    assert 0.05 < surf.atm_iv < 1.0, f"ATM IV {surf.atm_iv:.4f} out of range"

    # Interpolate at a few interior points
    S = cleaned.chain.spot
    r, q = cleaned.chain.r, cleaned.chain.q
    for T in [0.25, 0.5, 1.0]:
        F = S * np.exp((r - q) * T)
        iv = surf.iv(F, T)  # ATM
        assert not np.isnan(iv), f"ATM IV NaN at T={T}"
        assert 0.05 < iv < 1.0, f"ATM IV {iv:.4f} out of range at T={T}"

    print(f"  [OK] VolSurface: {surf.n_options} options, ATM IV={surf.atm_iv:.4f}")
    print(f"       IV range: [{surf.summary()['iv_min']:.4f}, {surf.summary()['iv_max']:.4f}]")
    print(f"       Calendar violations: {surf.summary()['calendar_violations']}")


def test_zero_gap_with_correct_model():
    """When we use the SAME parameters to generate and price, gap must be near zero."""
    params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    cleaned = _make_synthetic_chain(params)
    surf = build_surface(cleaned)

    sm = detect_mispricings(surf, params, calibration_rmse=0.001)

    assert sm.feller_ok, "Feller must hold for equity_typical params"
    assert len(sm.signals) == 0, (
        f"Expected 0 signals with correct model, got {len(sm.signals)}. "
        f"Max gap: {sm.max_abs_gap:.4f}"
    )
    print(f"  [OK] Zero gap: {len(sm.signals)} signals with correct params")


def test_nonzero_gap_with_wrong_model():
    """When we use DIFFERENT parameters, gap must exceed the Buffett gate."""
    market_params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    model_params  = HestonParams(kappa=3.0, theta=0.06, xi=0.4, rho=-0.4, v0=0.06)

    cleaned = _make_synthetic_chain(market_params)
    surf = build_surface(cleaned)

    sm = detect_mispricings(surf, model_params, calibration_rmse=0.05)

    assert sm.feller_ok, "Feller must hold for model params"
    assert len(sm.signals) > 0, (
        f"Expected signals with mismatched model, got 0. "
        f"MAX gap check: possibly gap < MIN_VOL_GAP={config.MIN_VOL_GAP}"
    )
    assert sm.max_abs_gap >= config.MIN_VOL_GAP, (
        f"Max gap {sm.max_abs_gap:.4f} below threshold {config.MIN_VOL_GAP}"
    )
    print(f"  [OK] Nonzero gap: {len(sm.signals)} signals, max gap={sm.max_abs_gap:.4f}")
    print(f"       Buy: {sm.n_buy}, Sell: {sm.n_sell}")


def test_feller_violation_suppresses_signals():
    """A Feller-violating model must return zero signals, not garbage."""
    market_params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
    # xi=0.5 → ξ²=0.25 > 2κθ=0.16: violates Feller
    bad_params = HestonParams(kappa=2.0, theta=0.04, xi=0.5, rho=-0.7, v0=0.04)

    cleaned = _make_synthetic_chain(market_params)
    surf = build_surface(cleaned)

    sm = detect_mispricings(surf, bad_params, calibration_rmse=0.20)

    assert not sm.feller_ok, "Should detect Feller violation"
    assert len(sm.signals) == 0, f"Signals suppressed on Feller violation: got {len(sm.signals)}"
    print(f"  [OK] Feller gate: 0 signals returned for violating params")


if __name__ == "__main__":
    print("=" * 60)
    print("SURFACE + MISPRICING INTEGRATION TESTS")
    print("=" * 60)

    tests = [
        ("VolSurface builds and interpolates", test_surface_builds_and_interpolates),
        ("Zero gap with correct model",         test_zero_gap_with_correct_model),
        ("Nonzero gap with wrong model",        test_nonzero_gap_with_wrong_model),
        ("Feller violation suppresses signals", test_feller_violation_suppresses_signals),
    ]

    passed = 0
    for name, fn in tests:
        print(f"\n{'-'*60}")
        print(f"Test: {name}")
        try:
            fn()
            passed += 1
            print(f"  RESULT: PASS")
        except AssertionError as e:
            print(f"  RESULT: FAIL — {e}")
        except Exception as e:
            print(f"  RESULT: ERROR — {type(e).__name__}: {e}")

    print(f"\n{'='*60}")
    print(f"SUMMARY: {passed}/{len(tests)} tests passed")
    if passed == len(tests):
        print("\n[OK] Surface and mispricing layers pass. Proceed to hedging.")
    else:
        print("\n[!] Fix failing tests before proceeding.")
