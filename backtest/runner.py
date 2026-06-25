"""
Heston-arb backtest harness.

Feeds historical CBOE DataShop data through the full pipeline day-by-day:
  loader → cleaner → surface → calibration → kill check → signals → P&L

P&L model (vega-weighted IV reversion):
  - Enter when |model_iv - market_iv| > MIN_VOL_GAP
  - Mark-to-market daily: pnl_i = vega_i × Δ(market_iv) × direction
  - Exit when gap < EXIT_GAP_THRESHOLD, option expires, or max_hold_days reached
  - Delta hedge cost: EQUITY_BID_ASK_BPS per day per unit vega

Usage:
    from backtest.runner import run_backtest
    results = run_backtest(
        ticker="SPX",
        data_dir="/data/cboe/spx/",
        start_date="2022-01-03",
        end_date="2022-12-30",
    )
    results.print_summary()

Or via CLI:
    python -m backtest.runner --ticker SPX --data-dir /data/cboe/spx \
        --start 2022-01-03 --end 2022-12-30
"""

import argparse
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.stats import norm

import config
from calibration.heston import HestonParams
from calibration.calibrator import CalibrationInput, calibrate
from data.loader import CBOELoader
from data.cleaner import clean_chain, OptionsChain
from data.surface import build_surface, VolSurface
from evals.kill import KillCondition
from signals.mispricing import detect_mispricings, MispricingSignal, SurfaceMispricing
from signals.sizing import size_portfolio

import jax.numpy as jnp

# ── Backtest config ───────────────────────────────────────────────────────────

EXIT_GAP_THRESHOLD = 0.005     # exit position when gap narrows below 0.5 vol pts
MAX_HOLD_DAYS = 10             # forced exit after N trading days
MIN_TTM_EXIT_DAYS = 5          # exit if option has < 5 days to expiry
HEDGE_COST_PER_VEGA_DAY = (
    config.EQUITY_BID_ASK_BPS * 1e-4 * 2   # bid-ask round-trip per delta rebalance
)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Position:
    """A single open vol-arb position."""
    ticker: str
    entry_date: str
    strike: float
    maturity: float              # years at entry
    direction: str               # "buy" or "sell"
    entry_market_iv: float
    entry_model_iv: float
    entry_vol_gap: float
    qty: int = 1                 # number of contracts (Kelly-sized)
    age_days: int = 0
    cumulative_pnl: float = 0.0  # total P&L across all contracts
    exited: bool = False
    exit_date: Optional[str] = None
    exit_reason: Optional[str] = None


@dataclass
class DailySnapshot:
    """One day's pipeline output."""
    date: str
    n_options: int
    calibration_rmse: float
    feller_ok: bool
    n_signals: int
    n_entries: int
    n_exits: int
    open_positions: int
    daily_pnl: float            # sum of all position mark-to-market moves
    hedge_cost: float
    net_pnl: float              # daily_pnl - hedge_cost
    killed: bool = False
    params: Optional[HestonParams] = None


@dataclass
class BacktestResults:
    """Full backtest output."""
    ticker: str
    start_date: str
    end_date: str
    daily: list[DailySnapshot]
    positions: list[Position]   # all positions (open + closed)

    @property
    def n_trading_days(self) -> int:
        return len(self.daily)

    @property
    def daily_pnl(self) -> np.ndarray:
        return np.array([d.net_pnl for d in self.daily])

    @property
    def cumulative_pnl(self) -> np.ndarray:
        return np.cumsum(self.daily_pnl)

    @property
    def closed_positions(self) -> list[Position]:
        return [p for p in self.positions if p.exited]

    @property
    def sharpe(self) -> float:
        pnl = self.daily_pnl
        if len(pnl) < 2 or pnl.std() == 0:
            return np.nan
        return float(pnl.mean() / pnl.std() * np.sqrt(252))

    @property
    def max_drawdown(self) -> float:
        cum = self.cumulative_pnl
        if len(cum) == 0:
            return 0.0
        peak = np.maximum.accumulate(cum)
        drawdown = peak - cum
        return float(drawdown.max())

    @property
    def win_rate(self) -> float:
        closed = self.closed_positions
        if not closed:
            return np.nan
        winners = sum(1 for p in closed if p.cumulative_pnl > 0)
        return winners / len(closed)

    @property
    def avg_hold_days(self) -> float:
        closed = self.closed_positions
        if not closed:
            return np.nan
        return float(np.mean([p.age_days for p in closed]))

    def print_summary(self) -> None:
        print(f"\n{'='*60}")
        print(f"BACKTEST SUMMARY: {self.ticker}")
        print(f"  Period:          {self.start_date} to {self.end_date}")
        print(f"  Trading days:    {self.n_trading_days}")
        print(f"{'='*60}")
        pnl = self.daily_pnl
        print(f"  Total PnL:       {self.cumulative_pnl[-1] if len(pnl) else 0:.4f} vol-pts·vega")
        print(f"  Sharpe (ann.):   {self.sharpe:.2f}")
        print(f"  Max drawdown:    {self.max_drawdown:.4f}")
        print(f"  Win rate:        {self.win_rate:.1%}" if not np.isnan(self.win_rate) else "  Win rate:        N/A")
        print(f"  Avg hold days:   {self.avg_hold_days:.1f}" if not np.isnan(self.avg_hold_days) else "  Avg hold days:   N/A")
        print(f"  Total positions: {len(self.positions)}")
        total_qty = sum(p.qty for p in self.positions)
        print(f"  Total contracts: {total_qty}")
        print(f"  Closed:          {len(self.closed_positions)}")
        killed = sum(1 for d in self.daily if d.killed)
        print(f"  Kill-halted days:{killed}")


# ── BS vega helper ────────────────────────────────────────────────────────────

def _bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes vega (per unit notional). Returns 0 if sigma <= 0 or T <= 0."""
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return float(S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T))


# ── Core daily step ───────────────────────────────────────────────────────────

def _build_cal_input(surface: VolSurface, chain: OptionsChain) -> CalibrationInput:
    S = chain.spot
    r = chain.r
    q = chain.q
    F = S * np.exp((r - q) * surface.maturities)
    strikes_cal = jnp.array(F * np.exp(surface.log_moneyness))
    return CalibrationInput(
        strikes=strikes_cal,
        maturities=jnp.array(surface.maturities),
        market_ivs=jnp.array(surface.market_ivs),
        weights=jnp.ones(surface.n_options) / surface.n_options,
        S=S, r=r, q=q,
    )


def _mark_positions(
    open_positions: list[Position],
    surface: VolSurface,
    params: HestonParams,
    date: str,
) -> tuple[list[Position], float]:
    """Update open positions: mark-to-market, age, and flag for exit.

    Returns updated position list and total daily_pnl (before hedge cost).
    """
    from signals.mispricing import compute_model_ivs

    chain = surface.chain
    model_ivs = compute_model_ivs(surface, params)
    F = chain.spot * np.exp((chain.r - chain.q) * surface.maturities)
    strikes_arr = F * np.exp(surface.log_moneyness)

    daily_pnl = 0.0

    for pos in open_positions:
        if pos.exited:
            continue

        pos.age_days += 1

        # Check forced exit conditions first
        remaining_ttm_days = pos.maturity * 365
        if remaining_ttm_days < MIN_TTM_EXIT_DAYS:
            pos.exited = True
            pos.exit_date = date
            pos.exit_reason = "expiry"
            continue

        if pos.age_days >= MAX_HOLD_DAYS:
            pos.exited = True
            pos.exit_date = date
            pos.exit_reason = "max_hold"
            continue

        # Find closest market IV for this position on today's surface
        dists = np.abs(strikes_arr - pos.strike) + 10 * np.abs(surface.maturities - pos.maturity)
        nearest_idx = int(np.argmin(dists))
        current_market_iv = float(surface.market_ivs[nearest_idx])
        current_model_iv = float(model_ivs[nearest_idx]) if not np.isnan(model_ivs[nearest_idx]) else np.nan

        if np.isnan(current_market_iv) or np.isnan(current_model_iv):
            continue

        current_gap = abs(current_model_iv - current_market_iv)

        # P&L: for "sell" position (sold expensive vol), we gain when market_iv falls
        # direction = "sell" → entry was market_iv > model_iv
        # pnl_today = vega × (prior_market_iv - current_market_iv) for sell
        # We track using entry_market_iv as reference; mark-to-market each day
        vega = _bs_vega(chain.spot, pos.strike, pos.maturity, chain.r, chain.q,
                        pos.entry_market_iv)

        if pos.direction == "sell":
            # Sold overpriced vol: gain when market_iv falls toward model_iv
            move_pnl = pos.qty * vega * (pos.entry_market_iv - current_market_iv)
        else:
            # Bought underpriced vol: gain when market_iv rises toward model_iv
            move_pnl = pos.qty * vega * (current_market_iv - pos.entry_market_iv)

        pos.cumulative_pnl += move_pnl
        daily_pnl += move_pnl

        # Exit check: gap has closed
        if current_gap < EXIT_GAP_THRESHOLD:
            pos.exited = True
            pos.exit_date = date
            pos.exit_reason = "gap_closed"

    return open_positions, daily_pnl


def _enter_positions(
    signals: list[MispricingSignal],
    open_positions: list[Position],
    date: str,
    spot: float,
    r: float,
    q: float,
    max_signals_per_day: int,
) -> tuple[list[Position], int]:
    """Size and enter new positions from today's signals.

    Uses Kelly criterion via size_portfolio(). Deduplicates against already-open keys.
    """
    open_keys = {
        (round(p.strike, 1), round(p.maturity, 3))
        for p in open_positions if not p.exited
    }

    # Filter to genuinely new signals before sizing
    new_signals = [
        sig for sig in signals
        if (round(sig.strike, 1), round(sig.maturity, 3)) not in open_keys
    ]

    sizing = size_portfolio(new_signals, spot=spot, r=r, q=q,
                            max_new_positions=max_signals_per_day)

    n_entered = 0
    for result in sizing:
        if result.qty == 0:
            continue
        sig = result.signal
        key = (round(sig.strike, 1), round(sig.maturity, 3))
        pos = Position(
            ticker=sig.ticker,
            entry_date=date,
            strike=sig.strike,
            maturity=sig.maturity,
            direction=sig.direction,
            entry_market_iv=sig.market_iv,
            entry_model_iv=sig.model_iv,
            entry_vol_gap=sig.vol_gap,
            qty=result.qty,
        )
        open_positions.append(pos)
        open_keys.add(key)
        n_entered += 1

    return open_positions, n_entered


# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest(
    ticker: str,
    data_dir: str,
    start_date: str,
    end_date: str,
    initial_params: Optional[HestonParams] = None,
    n_cal_steps: int = 300,
    max_signals_per_day: int = 5,
    verbose: bool = True,
    kill_log_dir: Optional[str] = None,
) -> BacktestResults:
    """Run the full backtest pipeline.

    Args:
        ticker: e.g. "SPX"
        data_dir: Directory of CBOE DataShop CSV files.
        start_date / end_date: Inclusive date range ("YYYY-MM-DD").
        initial_params: Starting Heston params for day 1. Defaults to typical SPX values.
        n_cal_steps: Calibration gradient steps per day (fewer = faster, less accurate).
        max_signals_per_day: Cap on new positions entered per day (position sizing placeholder).
        verbose: Print progress each day.
        kill_log_dir: Override directory for kill condition log files.
    """
    if initial_params is None:
        initial_params = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)

    loader = CBOELoader(data_dir=data_dir)

    kill_log_path = (Path(kill_log_dir) / "kill_log.json") if kill_log_dir else Path(data_dir).parent / "kill_log.json"
    kill_flag_path = kill_log_path.parent / "kill_flag.json"
    kc = KillCondition(log_path=kill_log_path, flag_path=kill_flag_path)

    # Enumerate trading dates from the CBOE data directory
    trading_dates = _find_trading_dates(data_dir, ticker, start_date, end_date)
    if not trading_dates:
        raise ValueError(
            f"No CBOE data files found for {ticker} in {data_dir} "
            f"between {start_date} and {end_date}. "
            f"Expected filenames like: {ticker}_YYYY-MM-DD.csv or {ticker}_YYYYMMDD.csv"
        )

    if verbose:
        print(f"[backtest] {ticker}: {len(trading_dates)} trading days "
              f"({start_date} to {end_date})")

    daily_snapshots: list[DailySnapshot] = []
    all_positions: list[Position] = []
    open_positions: list[Position] = []
    prev_params: Optional[HestonParams] = None
    current_params = initial_params

    for date in trading_dates:
        snap = _run_one_day(
            date=date,
            ticker=ticker,
            loader=loader,
            kc=kc,
            open_positions=open_positions,
            current_params=current_params,
            prev_params=prev_params,
            n_cal_steps=n_cal_steps,
            max_signals_per_day=max_signals_per_day,
            verbose=verbose,
        )
        daily_snapshots.append(snap)
        all_positions.extend([p for p in open_positions if p.entry_date == date])

        # Warm-start next day with today's calibrated params
        if snap.params is not None:
            prev_params = current_params
            current_params = snap.params

    return BacktestResults(
        ticker=ticker,
        start_date=start_date,
        end_date=end_date,
        daily=daily_snapshots,
        positions=all_positions,
    )


def _run_one_day(
    date: str,
    ticker: str,
    loader: CBOELoader,
    kc: KillCondition,
    open_positions: list[Position],
    current_params: HestonParams,
    prev_params: Optional[HestonParams],
    n_cal_steps: int,
    max_signals_per_day: int,
    verbose: bool,
) -> DailySnapshot:
    """Process one trading day end-to-end."""

    # ── Check kill flag before doing any work ─────────────────────────────
    if kc.is_halted():
        if verbose:
            print(f"[{date}] KILLED — strategy halted, skipping")
        return DailySnapshot(
            date=date, n_options=0, calibration_rmse=np.nan, feller_ok=False,
            n_signals=0, n_entries=0, n_exits=0, open_positions=len(open_positions),
            daily_pnl=0.0, hedge_cost=0.0, net_pnl=0.0, killed=True,
        )

    # ── Load + clean ──────────────────────────────────────────────────────
    try:
        chain = loader.fetch(ticker, date)
    except Exception as e:
        if verbose:
            print(f"[{date}] Load failed: {e}")
        return DailySnapshot(
            date=date, n_options=0, calibration_rmse=np.nan, feller_ok=False,
            n_signals=0, n_entries=0, n_exits=0, open_positions=len(open_positions),
            daily_pnl=0.0, hedge_cost=0.0, net_pnl=0.0,
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = clean_chain(chain)

    surface = build_surface(cleaned)

    if surface.n_options < 10:
        if verbose:
            print(f"[{date}] Only {surface.n_options} options after cleaning — skipping")
        return DailySnapshot(
            date=date, n_options=surface.n_options, calibration_rmse=np.nan,
            feller_ok=False, n_signals=0, n_entries=0, n_exits=0,
            open_positions=len(open_positions), daily_pnl=0.0, hedge_cost=0.0, net_pnl=0.0,
        )

    # ── Calibrate ─────────────────────────────────────────────────────────
    cal = _build_cal_input(surface, chain)
    try:
        fitted, rmse = calibrate(cal, initial_params=current_params,
                                 prev_params=prev_params, n_steps=n_cal_steps)
    except Exception as e:
        if verbose:
            print(f"[{date}] Calibration failed: {e}")
        rmse = np.inf
        fitted = current_params

    # ── Kill condition update ─────────────────────────────────────────────
    kill_state = kc.record(ticker, date, float(rmse))
    if kill_state.halted:
        if verbose:
            print(f"[{date}] KILL TRIGGERED: {kill_state.reason}")
        return DailySnapshot(
            date=date, n_options=surface.n_options, calibration_rmse=float(rmse),
            feller_ok=False, n_signals=0, n_entries=0, n_exits=0,
            open_positions=len(open_positions), daily_pnl=0.0, hedge_cost=0.0,
            net_pnl=0.0, killed=True, params=fitted,
        )

    # ── Mark open positions ───────────────────────────────────────────────
    n_open_before = sum(1 for p in open_positions if not p.exited)
    open_positions, daily_pnl = _mark_positions(open_positions, surface, fitted, date)
    n_exited = sum(1 for p in open_positions if p.exited and p.exit_date == date)
    n_open_after_exits = sum(1 for p in open_positions if not p.exited)

    # ── Signals → new entries (Kelly-sized) ──────────────────────────────
    sm = detect_mispricings(surface, fitted, calibration_rmse=float(rmse))
    open_positions, n_entered = _enter_positions(
        sm.signals, open_positions, date,
        spot=chain.spot, r=chain.r, q=chain.q,
        max_signals_per_day=max_signals_per_day,
    )

    n_open_total = sum(1 for p in open_positions if not p.exited)

    # ── Hedge cost (scales with total open contracts) ─────────────────────
    total_contracts = sum(p.qty for p in open_positions if not p.exited)
    hedge_cost = total_contracts * HEDGE_COST_PER_VEGA_DAY

    net_pnl = daily_pnl - hedge_cost

    if verbose:
        print(
            f"[{date}] n={surface.n_options:3d}  rmse={rmse:.4f}  "
            f"signals={len(sm.signals):3d}  entries={n_entered:2d}  "
            f"exits={n_exited:2d}  open={n_open_total:2d}  "
            f"pnl={net_pnl:+.4f}"
        )

    return DailySnapshot(
        date=date,
        n_options=surface.n_options,
        calibration_rmse=float(rmse),
        feller_ok=sm.feller_ok,
        n_signals=len(sm.signals),
        n_entries=n_entered,
        n_exits=n_exited,
        open_positions=n_open_total,
        daily_pnl=daily_pnl,
        hedge_cost=hedge_cost,
        net_pnl=net_pnl,
        params=fitted,
    )


# ── Date discovery ────────────────────────────────────────────────────────────

def _find_trading_dates(
    data_dir: str,
    ticker: str,
    start_date: str,
    end_date: str,
) -> list[str]:
    """Find sorted list of dates with available CBOE data files."""
    import re
    data_path = Path(data_dir)

    # Match patterns: TICKER_YYYY-MM-DD.csv, TICKER_YYYYMMDD.csv,
    # YYYY-MM-DD.csv, YYYYMMDD.csv
    date_re = re.compile(
        r'(\d{4}-\d{2}-\d{2}|\d{8})',
    )

    found_dates = []
    for f in data_path.glob("*.csv"):
        match = date_re.search(f.name)
        if not match:
            continue
        raw = match.group(1)
        if len(raw) == 8:  # YYYYMMDD
            date_str = f"{raw[:4]}-{raw[4:6]}-{raw[6:]}"
        else:
            date_str = raw
        if start_date <= date_str <= end_date:
            found_dates.append(date_str)

    return sorted(set(found_dates))


# ── Metrics helpers ───────────────────────────────────────────────────────────

def compute_metrics(results: BacktestResults) -> dict:
    """Compute standard performance metrics from backtest results."""
    pnl = results.daily_pnl
    cum = results.cumulative_pnl

    if len(pnl) == 0:
        return {"error": "no trading days"}

    peak = np.maximum.accumulate(cum)
    drawdowns = peak - cum

    return {
        "total_pnl": float(cum[-1]),
        "sharpe_annualized": results.sharpe,
        "max_drawdown": results.max_drawdown,
        "calmar": float(cum[-1] / results.max_drawdown) if results.max_drawdown > 0 else np.nan,
        "win_rate": results.win_rate,
        "avg_hold_days": results.avg_hold_days,
        "n_positions": len(results.positions),
        "n_closed": len(results.closed_positions),
        "daily_pnl_mean": float(pnl.mean()),
        "daily_pnl_std": float(pnl.std()),
        "n_trading_days": results.n_trading_days,
        "kill_days": sum(1 for d in results.daily if d.killed),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Heston-arb backtest harness")
    parser.add_argument("--ticker", default="SPX")
    parser.add_argument("--data-dir", required=True, help="Directory of CBOE CSV files")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--cal-steps", type=int, default=300, help="Calibration gradient steps")
    parser.add_argument("--max-signals", type=int, default=5, help="Max new positions per day")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    results = run_backtest(
        ticker=args.ticker,
        data_dir=args.data_dir,
        start_date=args.start,
        end_date=args.end,
        n_cal_steps=args.cal_steps,
        max_signals_per_day=args.max_signals,
        verbose=not args.quiet,
    )

    results.print_summary()

    metrics = compute_metrics(results)
    print("\nDetailed metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
