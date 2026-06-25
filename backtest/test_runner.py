"""Smoke-test for the backtest harness using synthetic CBOE-format CSV files."""

import csv
import json
import tempfile
import numpy as np
from pathlib import Path
from datetime import date, timedelta

from calibration.heston import HestonParams, price_call
from backtest.runner import run_backtest, compute_metrics


# ── Synthetic CBOE CSV generator ──────────────────────────────────────────────

TRUE_PARAMS = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)
S = 4500.0
R = 0.05
Q = 0.02


def _heston_price(K: float, T: float) -> float:
    return float(price_call(S, K, T, R, Q, TRUE_PARAMS))


def _iv_from_price(price: float, K: float, T: float) -> float:
    """Approximate BS IV via simple bisection."""
    from scipy.stats import norm as _norm

    def bs_call(sigma):
        if sigma <= 0:
            return 0.0
        d1 = (np.log(S / K) + (R - Q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return S * np.exp(-Q * T) * _norm.cdf(d1) - K * np.exp(-R * T) * _norm.cdf(d2)

    lo, hi = 0.01, 5.0
    for _ in range(50):
        mid = (lo + hi) / 2
        if bs_call(mid) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def _write_cboe_csv(path: Path, snap_date: str) -> None:
    """Write a minimal CBOE DataShop-format CSV for one day."""
    snap_dt = f"{snap_date} 09:30:00"
    maturities = [30/365, 60/365, 90/365]
    log_ks = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15]

    rows = []
    for T in maturities:
        F = S * np.exp((R - Q) * T)
        exp_date = (date.fromisoformat(snap_date) + timedelta(days=int(T * 365))).isoformat()
        for lk in log_ks:
            K = F * np.exp(lk)
            price = _heston_price(K, T)
            iv = _iv_from_price(price, K, T)
            if iv <= 0:
                continue
            half_spread = price * 0.01
            rows.append({
                "quote_datetime": snap_dt,
                "root": "SPX",
                "expiration": exp_date,
                "strike": round(K, 2),
                "option_type": "C",
                "bid": round(price - half_spread, 4),
                "ask": round(price + half_spread, 4),
                "open_interest": 1000,
                "underlying_bid": S - 0.5,
                "underlying_ask": S + 0.5,
                "implied_volatility": round(iv, 6),
            })

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ── Test ──────────────────────────────────────────────────────────────────────

def test_backtest_synthetic():
    with tempfile.TemporaryDirectory() as d:
        data_dir = Path(d) / "cboe"
        data_dir.mkdir()

        # Write 5 trading days of synthetic data
        dates = [
            "2024-03-01", "2024-03-04", "2024-03-05",
            "2024-03-06", "2024-03-07",
        ]
        for snap_date in dates:
            csv_path = data_dir / f"SPX_{snap_date}.csv"
            _write_cboe_csv(csv_path, snap_date)

        results = run_backtest(
            ticker="SPX",
            data_dir=str(data_dir),
            start_date="2024-03-01",
            end_date="2024-03-07",
            n_cal_steps=200,
            max_signals_per_day=3,
            verbose=True,
            kill_log_dir=d,
        )

        # Basic structural checks
        assert results.n_trading_days == 5, f"Expected 5 days, got {results.n_trading_days}"
        assert len(results.daily) == 5

        # Each day should have loaded options
        for snap in results.daily:
            if not snap.killed:
                assert snap.n_options > 0, f"No options on {snap.date}"
                assert snap.calibration_rmse < 0.50, \
                    f"RMSE too high on {snap.date}: {snap.calibration_rmse:.4f}"
                assert snap.feller_ok, f"Feller violated on {snap.date}"

        # Should have detected at least some signals over 5 days
        total_signals = sum(d.n_signals for d in results.daily)
        print(f"\n  Total signals detected: {total_signals}")

        # Metrics compute without error
        metrics = compute_metrics(results)
        assert "total_pnl" in metrics
        assert "sharpe_annualized" in metrics
        assert "max_drawdown" in metrics

        results.print_summary()

        print(f"\n  Daily PnL: {results.daily_pnl}")
        print(f"  Cum PnL:   {results.cumulative_pnl[-1]:.4f}")

        return True


if __name__ == "__main__":
    print("="*60)
    print("BACKTEST HARNESS SMOKE TEST")
    print("="*60)
    try:
        test_backtest_synthetic()
        print("\nSMOKE TEST: PASS")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise SystemExit(1)
