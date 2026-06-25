"""
Live trading loop.

One tick = one full pipeline pass:
  1. Kill-flag guard
  2. Fetch Polygon options chain
  3. Clean (LP arbitrage repair)
  4. Build vol surface
  5. Calibrate Heston (warm-start from prev tick)
  6. Record RMSE to Popper kill log
  7. Laplace uncertainty posterior
  8. Detect mispricings + posterior Buffett gate
  9. Kelly-size new entries, submit option orders
 10. Mark-to-market existing positions, check exits, submit exit orders
 11. Delta rebalance: compute net portfolio delta, submit hedge

Run modes:
  dry_run=True  (default): all broker calls go to PaperBroker — no real orders.
  dry_run=False: routes to the supplied broker (IBKR or other).

Usage:
    from trading.loop import run_live
    from trading.broker import PaperBroker

    run_live(
        ticker="SPY",
        polygon_api_key="your_key",
        broker=PaperBroker(),
        interval_minutes=240,   # 4 hours
        dry_run=True,
    )

Or via CLI:
    python -m trading.loop --ticker SPY --interval 240 --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


def _load_dotenv(path: str = ".env") -> None:
    """Load key=value pairs from .env into os.environ (no external dependency)."""
    env_path = Path(path)
    if not env_path.exists():
        return
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val

import numpy as np
from scipy.stats import norm

import config
from calibration.heston import HestonParams, feller_satisfied
from calibration.calibrator import CalibrationInput, calibrate, calibration_rmse
from calibration.uncertainty import run_laplace
from data.cleaner import clean_chain, OptionsChain
from data.surface import build_surface
from data.polygon import PolygonLoader
from data.alpaca_loader import AlpacaLoader
from data.rates import get_rates
from evals.kill import KillCondition
from risk.hedging import portfolio_delta, optimize_hedge, OptionPosition
from calibration.uncertainty import filter_signals_by_posterior
from signals.mispricing import detect_mispricings, MispricingSignal, compute_model_ivs
from signals.sizing import size_portfolio, SizingResult
from trading.broker import Broker, PaperBroker, OptionContract, Order
from trading.state import LiveState, LivePosition

import jax.numpy as jnp

# ── Constants ─────────────────────────────────────────────────────────────────

_STATE_PATH = Path("data/live_state.json")
_KILL_LOG_PATH = Path("data/kill_log.json")
_KILL_FLAG_PATH = Path("data/kill_flag.json")
_TICK_LOG_PATH = Path("data/tick_log.jsonl")   # one JSON object per tick (append-only)
_N_CAL_STEPS = 300
_LAPLACE_SAMPLES = 200      # fewer samples per tick to save time
_LAPLACE_REG = 0.05
_MIN_OPTIONS_AFTER_CLEAN = 10
_DELTA_BAND = 0.02          # tolerate ±2% delta before hedging
_MAX_HEDGE_SHARES = 10_000


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _maturity_to_expiry(snapshot_date: str, maturity_years: float) -> str:
    """Convert TTM in years to an approximate expiry date string."""
    snap = datetime.fromisoformat(snapshot_date)
    exp = snap + timedelta(days=round(maturity_years * 365))
    return exp.strftime("%Y-%m-%d")


def _bs_delta(S: float, K: float, T: float, r: float, q: float, sigma: float,
               opt_type: str = "C") -> float:
    if sigma <= 0 or T <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    if opt_type == "C":
        return math.exp(-q * T) * float(norm.cdf(d1))
    return math.exp(-q * T) * float(norm.cdf(d1) - 1)


def _nbbo_mid(chain: OptionsChain, strike: float, maturity: float) -> Optional[float]:
    """Return NBBO mid price for the contract closest to (strike, maturity).

    Returns None if the chain is empty or no contract is close enough.
    Uses relative strike distance + 2x maturity distance to rank contracts.
    """
    if len(chain.strikes) == 0:
        return None
    rel_strike = np.abs(chain.strikes - strike) / max(strike, 1e-6)
    rel_mat = 2.0 * np.abs(chain.maturities - maturity)
    idx = int(np.argmin(rel_strike + rel_mat))
    if rel_strike[idx] > 0.02:
        return None
    bid = float(chain.bid_prices[idx])
    ask = float(chain.ask_prices[idx])
    if bid > 0 and ask > bid:
        return (bid + ask) / 2.0
    return float(chain.mid_prices[idx]) if chain.mid_prices[idx] > 0 else None


def _nbbo_bid_ask(chain: OptionsChain, strike: float, maturity: float):
    """Return (bid, ask) for the contract closest to (strike, maturity), or None.

    Used to price capped marketable-limit exits instead of uncapped market orders.
    """
    if len(chain.strikes) == 0:
        return None
    rel_strike = np.abs(chain.strikes - strike) / max(strike, 1e-6)
    rel_mat = 2.0 * np.abs(chain.maturities - maturity)
    idx = int(np.argmin(rel_strike + rel_mat))
    if rel_strike[idx] > 0.02:
        return None
    bid = float(chain.bid_prices[idx])
    ask = float(chain.ask_prices[idx])
    if bid > 0 and ask > bid:
        return (bid, ask)
    return None


def _market_is_open(broker) -> bool:
    """True if the market is open. Prefers the broker's clock; falls back to UTC RTH.

    Fallback (no broker clock): Mon-Fri 13:30-20:00 UTC (US regular trading hours,
    ignoring holidays). Brokers that expose is_market_open() (AlpacaBroker) are authoritative.
    """
    fn = getattr(broker, "is_market_open", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            pass
    now = datetime.now(timezone.utc)
    if now.weekday() >= 5:
        return False
    minutes = now.hour * 60 + now.minute
    return 13 * 60 + 30 <= minutes <= 20 * 60


def _bs_vega(S: float, K: float, T: float, r: float, q: float, sigma: float) -> float:
    """Black-Scholes vega (per share, per unit vol). 0 for degenerate inputs."""
    if sigma <= 0 or T <= 0 or S <= 0 or K <= 0:
        return 0.0
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    return S * math.exp(-q * T) * float(norm.pdf(d1)) * math.sqrt(T)


def _filter_by_transaction_cost(signals, chain, spot, r, q):
    """Cost-aware Buffett gate: keep signals whose edge clears the round-trip spread.

    The round-trip option spread (ask - bid, per share) is converted to vol points via
    BS vega: cost_vol = (ask - bid) / vega. A signal survives only when
    |vol_gap| >= MIN_VOL_GAP + cost_vol — i.e. the mispricing is real net of what it
    costs to enter and exit. Signals we cannot price (no NBBO in the snapshot) are dropped.
    """
    kept = []
    for sig in signals:
        ba = _nbbo_bid_ask(chain, sig.strike, sig.maturity)
        if ba is None:
            continue
        bid, ask = ba
        vega = _bs_vega(spot, sig.strike, sig.maturity, r, q, sig.market_iv)
        if vega <= 1e-8:
            continue
        cost_vol = (ask - bid) / vega
        if abs(sig.vol_gap) >= config.MIN_VOL_GAP + cost_vol:
            kept.append(sig)
    return kept


_CAL_N_MATS = 8     # target number of maturities for calibration grid
_CAL_N_STRIKES = 12  # target number of strikes per maturity (spread across moneyness)


def _select_calibration_grid(surface, n_mats: int = _CAL_N_MATS,
                              n_strikes: int = _CAL_N_STRIKES) -> np.ndarray:
    """Return indices into surface arrays for a sparse (n_mats × n_strikes) grid.

    Selects maturities log-spaced between surface min/max, then for each picks
    n_strikes contracts SPREAD across the log-moneyness range (wings + ATM). Skew (ρ)
    and vol-of-vol (ξ) live in the wings, so an ATM-only grid leaves them unidentified.
    This keeps calibration tractable regardless of how many raw contracts exist.
    """
    mats = surface.maturities
    lm = surface.log_moneyness

    unique_mats = np.unique(np.round(mats, 4))
    if len(unique_mats) == 0:
        return np.arange(len(mats))

    # Log-spaced target maturities
    t_min, t_max = unique_mats[0], unique_mats[-1]
    if t_max > t_min:
        target_mats = np.exp(np.linspace(np.log(t_min), np.log(t_max), n_mats))
    else:
        target_mats = unique_mats[:1]

    # For each target maturity, find nearest actual maturity in the surface
    selected_mats = []
    for tm in target_mats:
        nearest = unique_mats[np.argmin(np.abs(unique_mats - tm))]
        if nearest not in selected_mats:
            selected_mats.append(nearest)

    # For each selected maturity, pick n_strikes contracts spread evenly across the
    # slice's log-moneyness range, so the fit sees the wings (skew/volvol), not just ATM.
    indices = []
    for T in selected_mats:
        mat_mask = np.abs(mats - T) < 1e-4
        mat_idx = np.where(mat_mask)[0]
        if len(mat_idx) == 0:
            continue
        slice_lm = lm[mat_idx]
        # Restrict to the band a single-factor Heston can fit — keeps RMSE sane.
        in_band = np.abs(slice_lm) <= config.MAX_CAL_LOG_MONEYNESS
        if in_band.any():
            mat_idx = mat_idx[in_band]
            slice_lm = slice_lm[in_band]
        k_lo, k_hi = float(slice_lm.min()), float(slice_lm.max())
        n_pick = min(n_strikes, len(mat_idx))
        if k_hi > k_lo and n_pick > 1:
            targets = np.linspace(k_lo, k_hi, n_pick)
            chosen = [int(mat_idx[np.argmin(np.abs(slice_lm - tk))]) for tk in targets]
        else:
            chosen = mat_idx.tolist()
        indices.extend(chosen)

    return np.array(sorted(set(indices)))


def _build_cal_input(surface, chain,
                     max_options: int = _CAL_N_MATS * _CAL_N_STRIKES) -> CalibrationInput:
    S = chain.spot
    F = S * np.exp((chain.r - chain.q) * surface.maturities)
    strikes_full = F * np.exp(surface.log_moneyness)

    if surface.n_options > max_options:
        idx = _select_calibration_grid(surface)
        strikes_cal = strikes_full[idx]
        mats_cal    = surface.maturities[idx]
        ivs_cal     = surface.market_ivs[idx]
    else:
        idx = np.arange(surface.n_options)
        strikes_cal = strikes_full
        mats_cal    = surface.maturities
        ivs_cal     = surface.market_ivs

    n = len(ivs_cal)
    return CalibrationInput(
        strikes=jnp.array(strikes_cal),
        maturities=jnp.array(mats_cal),
        market_ivs=jnp.array(ivs_cal),
        weights=jnp.ones(n) / n,
        S=S, r=chain.r, q=chain.q,
    )


# ── Tick ──────────────────────────────────────────────────────────────────────

def _append_tick_log(path: Optional[Path], record: dict) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception as e:
        warnings.warn(f"[loop] tick log write failed: {e}")


def _append_chain_log(chain_log_dir: Optional[Path], ticker: str, cleaned,
                      now: str, today: str) -> None:
    """Append the cleaned options chain for this tick to a daily Parquet file.

    Writes one file per (ticker, date): <chain_log_dir>/<ticker>/<today>.parquet,
    accumulated via read-concat-rewrite (each tick adds its row-batch). This is the
    training dataset: full post-clean microstructure (quotes, OI, repaired price, IV)
    plus the snapshot's spot/r/q. Failures are warned, never fatal to the loop.
    """
    if chain_log_dir is None:
        return
    try:
        import pandas as pd  # local import: only needed when capture is enabled
        ch = cleaned.chain
        n = len(ch.strikes)
        if n == 0:
            return
        mats = np.asarray(ch.maturities, dtype=float)
        df = pd.DataFrame({
            "time":           now,
            "date":           today,
            "ticker":         ticker,
            "strike":         np.asarray(ch.strikes, dtype=float),
            "maturity":       mats,
            "expiry":         [_maturity_to_expiry(today, float(T)) for T in mats],
            "opt_type":       np.asarray(ch.option_type).astype(str),
            "bid":            np.asarray(ch.bid_prices, dtype=float),
            "ask":            np.asarray(ch.ask_prices, dtype=float),
            "mid":            np.asarray(ch.mid_prices, dtype=float),
            "repaired_price": np.asarray(cleaned.repaired_prices, dtype=float),
            "oi":             np.asarray(ch.open_interest, dtype=float),
            "iv":             np.asarray(cleaned.implied_vols, dtype=float),
            "spot":           float(ch.spot),
            "r":              float(ch.r),
            "q":              float(ch.q),
        })
        out_dir = Path(chain_log_dir) / ticker
        out_dir.mkdir(parents=True, exist_ok=True)
        fp = out_dir / f"{today}.parquet"
        if fp.exists():
            df = pd.concat([pd.read_parquet(fp), df], ignore_index=True)
        df.to_parquet(fp, index=False)
    except Exception as e:
        warnings.warn(f"[loop] chain log write failed: {e}")


def _gap_uncertainty_diag(signals, posterior,
                          level: float = config.UNCERTAINTY_CREDIBLE_LEVEL) -> Optional[dict]:
    """Per-signal gap vs posterior uncertainty half-width.

    This answers the diagnostic question 'why did the uncertainty gate kill these
    signals?': is the edge genuinely weak (|gap| small) or is the calibration
    uncertainty wide (halfwidth large)? gap/halfwidth < 1 => the gap is within
    calibration noise (correct no-trade); >> 1 yet 0 traded => a gate problem.

    Returns None when there are no raw signals. Mirrors the nearest-grid-point
    mapping used by filter_signals_by_posterior so the readout matches the gate.
    """
    if not signals:
        return None
    abs_gaps = np.array([abs(float(s.vol_gap)) for s in signals])
    diag = {
        "n": len(signals),
        "max_abs_gap": float(np.max(abs_gaps)),
        "median_abs_gap": float(np.median(abs_gaps)),
        "median_halfwidth": None,
        "median_gap_over_hw": None,
        "n_cleared_uncertainty": None,
    }
    if posterior is None:
        return diag
    posterior._ensure_iv_samples()
    cal = posterior.cal
    strikes_arr = np.array(cal.strikes)
    mats_arr = np.array(cal.maturities)
    hws, ratios, n_cleared = [], [], 0
    for s in signals:
        F = cal.S * np.exp((cal.r - cal.q) * s.maturity)
        K = F * np.exp(s.log_moneyness)
        idx = int(np.argmin((strikes_arr - K) ** 2 + (mats_arr - s.maturity) ** 2 * 100))
        hw = posterior.iv_uncertainty_halfwidth(idx, level)
        if np.isnan(hw):
            continue
        hws.append(hw)
        if hw > 0:
            ratios.append(abs(float(s.vol_gap)) / hw)
        if abs(float(s.vol_gap)) > hw:
            n_cleared += 1
    diag["median_halfwidth"] = float(np.median(hws)) if hws else None
    diag["median_gap_over_hw"] = float(np.median(ratios)) if ratios else None
    diag["n_cleared_uncertainty"] = n_cleared
    return diag


def _run_tick(
    ticker: str,
    loader,
    broker: Broker,
    kc: KillCondition,
    state: LiveState,
    dry_run: bool,
    verbose: bool,
    tick_log_path: Optional[Path] = None,
    chain_log_dir: Optional[Path] = None,
) -> LiveState:
    """Execute one full pipeline pass. Returns updated state."""

    now = _now_iso()
    today = _today()

    if verbose:
        print(f"\n{'='*60}")
        print(f"TICK {state.n_ticks + 1}  {now}")
        print(f"{'='*60}")

    # ── 1. Kill-flag guard ────────────────────────────────────────────
    if kc.is_halted():
        print(f"[loop] HALTED — kill flag set. Run `python -m evals.kill reset` to resume.")
        state.last_tick_time = now
        state.n_ticks += 1
        return state

    # ── 1b. Market-hours guard ────────────────────────────────────────
    # When closed, do NO calibration, kill-recording, P&L marking, or orders — the
    # feed serves frozen last-NBBO off-hours and would poison the kill log and queue
    # orders against the next open. Just advance the tick counter and return.
    if not _market_is_open(broker):
        if verbose:
            print(f"[loop] Market closed — skipping tick (no calibration/orders).")
        state.last_tick_time = now
        state.n_ticks += 1
        return state

    # ── 2. Fetch chain (NBBO bid/ask when using AlpacaLoader) ─────────
    try:
        chain = loader.fetch(ticker, today)
    except Exception as e:
        print(f"[loop] Chain fetch failed: {e}")
        state.last_tick_time = now
        state.n_ticks += 1
        return state

    spot = chain.spot
    if spot <= 0:
        print(f"[loop] Could not determine spot price — skipping tick")
        state.last_tick_time = now
        state.n_ticks += 1
        return state

    if verbose:
        print(f"  Spot: {spot:.2f}   r={chain.r:.4f}  q={chain.q:.4f}  "
              f"n_raw={len(chain.strikes)}")

    # ── 3. Clean + surface ────────────────────────────────────────────
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = clean_chain(chain)
    surface = build_surface(cleaned)

    # Capture the cleaned chain for the training dataset BEFORE any skip below, so the
    # dataset records every tradeable snapshot even when this tick won't calibrate.
    _append_chain_log(chain_log_dir, ticker, cleaned, now, today)

    if surface.n_options < _MIN_OPTIONS_AFTER_CLEAN:
        print(f"[loop] Only {surface.n_options} clean options — skipping tick")
        state.last_tick_time = now
        state.n_ticks += 1
        return state

    if verbose:
        print(f"  Surface: {surface.n_options} options "
              f"({surface.n_maturities} maturities, {surface.n_strikes_per_mat} strikes)")

    # ── 4 & 5. Calibrate ─────────────────────────────────────────────
    init_params = state.get_prev_params() or HestonParams(
        kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04
    )
    cal = _build_cal_input(surface, chain)

    try:
        fitted, _loss = calibrate(cal, initial_params=init_params,
                                  prev_params=state.get_prev_params(),
                                  n_steps=_N_CAL_STEPS)
        # Pure log-IV RMSE (fit quality) — NOT the optimizer objective `_loss`, which
        # also includes Tikhonov + Feller penalties. The Popper kill thresholds on fit.
        rmse = calibration_rmse(fitted, cal)
    except Exception as e:
        print(f"[loop] Calibration failed: {e}")
        state.last_tick_time = now
        state.n_ticks += 1
        return state

    if verbose:
        print(f"  Calibrated: kappa={fitted.kappa:.3f} theta={fitted.theta:.4f} "
              f"xi={fitted.xi:.3f} rho={fitted.rho:.3f} v0={fitted.v0:.4f} "
              f"RMSE={rmse:.4f}")

    # ── 6. Kill log ───────────────────────────────────────────────────
    kill_state = kc.record(ticker, today, float(rmse))
    if kill_state.halted:
        print(f"[loop] KILL TRIGGERED: {kill_state.reason}")
        state.last_tick_time = now
        state.n_ticks += 1
        return state

    state.set_prev_params(fitted)

    # ── 6b. Intraday circuit-breaker (hybrid: complements the daily Popper kill) ──
    # Pause NEW entries after POPPER_KILL_TICKS consecutive failing ticks; the pause
    # lapses at the next calendar day. Existing positions are still marked/exited/hedged.
    if rmse > config.CALIBRATION_FAIL_RMSE:
        state.consec_fail_ticks += 1
    else:
        state.consec_fail_ticks = 0
    if (state.consec_fail_ticks >= config.POPPER_KILL_TICKS
            and state.intraday_halt_date != today):
        state.intraday_halt_date = today
        print(f"[loop] INTRADAY BREAKER: {state.consec_fail_ticks} consecutive "
              f"RMSE>{config.CALIBRATION_FAIL_RMSE} ticks — pausing new entries for {today}")
    intraday_paused = (state.intraday_halt_date == today)

    # ── 7. Uncertainty posterior ──────────────────────────────────────
    try:
        posterior = run_laplace(cal, fitted, n_samples=_LAPLACE_SAMPLES,
                                regularisation=_LAPLACE_REG, verbose=False)
    except Exception as e:
        print(f"[loop] Laplace failed: {e} — proceeding without uncertainty gate")
        posterior = None

    # ── 8. Signals ────────────────────────────────────────────────────
    # Only signal within the calibrated maturity range — the model is global in
    # moneyness but uncalibrated tenors are extrapolation.
    cal_t_min = float(np.min(cal.maturities))
    cal_t_max = float(np.max(cal.maturities))
    sm = detect_mispricings(surface, fitted, calibration_rmse=float(rmse),
                            cal_t_min=cal_t_min, cal_t_max=cal_t_max)
    if posterior:
        unc_signals = filter_signals_by_posterior(sm.signals, posterior)
    else:
        unc_signals = sm.signals

    # Cost-aware Buffett gate: require the edge to clear round-trip transaction cost.
    raw_signals = _filter_by_transaction_cost(unc_signals, chain, spot, chain.r, chain.q)

    # Diagnostic: WHY did signals survive/die? Distinguishes weak edge from wide uncertainty.
    gap_diag = _gap_uncertainty_diag(sm.signals, posterior)
    rmse_suppressed = float(rmse) > config.SIGNAL_MAX_RMSE

    if verbose:
        print(f"  Signals: {len(sm.signals)} raw -> {len(unc_signals)} after uncertainty "
              f"-> {len(raw_signals)} after cost gate")
        if gap_diag:
            hw = gap_diag["median_halfwidth"]
            ratio = gap_diag["median_gap_over_hw"]
            hw_s = "n/a" if hw is None else f"{hw:.3f}"
            ratio_s = "n/a" if ratio is None else f"{ratio:.2f}"
            print(f"    gap |max|={gap_diag['max_abs_gap']:.3f} |median|={gap_diag['median_abs_gap']:.3f}"
                  f"  uncertainty halfwidth(median)={hw_s}"
                  f"  gap/halfwidth(median)={ratio_s}"
                  f"  cleared uncertainty={gap_diag['n_cleared_uncertainty']}")
        elif rmse_suppressed:
            print(f"    (0 raw signals: RMSE {rmse:.4f} > SIGNAL_MAX_RMSE "
                  f"{config.SIGNAL_MAX_RMSE} — fit-quality gate suppressed the whole surface)")

    # ── 9. New entries ────────────────────────────────────────────────
    new_signals = [
        sig for sig in raw_signals
        if not state.has_position(
            sig.ticker, sig.strike,
            _maturity_to_expiry(today, sig.maturity),
            sig.direction,
        )
    ]

    if intraday_paused:
        if verbose and new_signals:
            print(f"  Entries paused — intraday breaker active for {today}")
        new_signals = []

    sizing = size_portfolio(new_signals, spot=spot, r=chain.r, q=chain.q)
    n_entered = 0

    for result in sizing:
        if result.qty == 0:
            continue
        sig = result.signal
        expiry = _maturity_to_expiry(today, sig.maturity)

        # NBBO mid price from the chain for limit order pricing. If the contract is
        # not in the snapshot, skip the entry rather than submitting a blind price.
        nbbo_mid = _nbbo_mid(chain, sig.strike, sig.maturity)
        if nbbo_mid is None:
            if verbose:
                print(f"  SKIP entry {ticker} {sig.strike:.0f} — no NBBO in snapshot")
            continue
        limit_price = round(nbbo_mid, 2)

        contract = OptionContract(
            underlying=ticker,
            expiry=expiry,
            strike=sig.strike,
            right="C",
        )

        action = "BUY" if sig.direction == "buy" else "SELL"
        if verbose:
            print(f"  ENTER {action} {result.qty}x {ticker} {sig.strike:.0f} "
                  f"exp={expiry} gap={sig.vol_gap:+.3f} lmt={limit_price:.2f}")

        order_id = None
        if not dry_run:
            try:
                order = broker.submit_option_order(
                    contract=contract,
                    action=action,
                    qty=result.qty,
                    order_type="LMT",
                    limit_price=limit_price,
                    close=False,
                )
                order_id = order.order_id
            except Exception as e:
                print(f"  [!] Order submission failed: {e}")
                continue
        else:
            order_id = f"DRY-{n_entered+1:04d}"

        pos = LivePosition(
            ticker=sig.ticker,
            entry_date=today,
            entry_time=now,
            strike=sig.strike,
            maturity=sig.maturity,
            expiry=expiry,
            direction=sig.direction,
            qty=result.qty,
            entry_market_iv=sig.market_iv,
            entry_model_iv=sig.model_iv,
            entry_vol_gap=sig.vol_gap,
            entry_spot=spot,
            option_order_id=order_id,
        )
        state.open_positions.append(pos)
        n_entered += 1

    # ── 10. Mark existing positions + exits ───────────────────────────
    F = spot * np.exp((chain.r - chain.q) * surface.maturities)
    strikes_arr = F * np.exp(surface.log_moneyness)
    model_ivs = compute_model_ivs(surface, fitted)
    tick_pnl = 0.0

    for pos in state.active_positions:
        # Wall-clock holding age in days — interval-agnostic. (Was `age_days += 1`
        # per tick, which at the 5-min production cadence force-closed every
        # position after 10 ticks = ~50 min instead of 10 days.)
        try:
            entry_dt = datetime.fromisoformat(pos.entry_time.replace("Z", "+00:00"))
            age_days = (datetime.now(timezone.utc) - entry_dt).total_seconds() / 86400.0
        except (ValueError, AttributeError, TypeError):
            age_days = float(pos.age_days)  # fallback for legacy/malformed records
        pos.age_days = int(age_days)

        # Find closest surface point
        dists = (np.abs(strikes_arr - pos.strike)
                 + 10 * np.abs(surface.maturities - pos.maturity))
        idx = int(np.argmin(dists))
        cur_market_iv = float(surface.market_ivs[idx])
        cur_model_iv = float(model_ivs[idx]) if not np.isnan(model_ivs[idx]) else np.nan

        if not np.isnan(cur_market_iv):
            vega = (pos.entry_spot * math.exp(-chain.q * pos.maturity)
                    * float(norm.pdf(0.0)) * math.sqrt(pos.maturity))
            if pos.direction == "sell":
                move = pos.qty * vega * (pos.entry_market_iv - cur_market_iv)
            else:
                move = pos.qty * vega * (cur_market_iv - pos.entry_market_iv)
            pos.cumulative_pnl += move
            tick_pnl += move

        # Exit conditions
        exit_reason = None
        # Days to expiry from the fixed expiry date and the current date — NOT
        # maturity*365 - age_days (which mixed calendar days with a tick counter).
        try:
            ttm_days = (date.fromisoformat(pos.expiry) - date.fromisoformat(today)).days
        except (ValueError, TypeError):
            ttm_days = pos.maturity * 365 - age_days  # fallback
        if ttm_days < config.MIN_TTM_DAYS:
            exit_reason = "expiry"
        elif age_days >= config.MAX_HOLD_DAYS:
            exit_reason = "max_hold"
        elif not np.isnan(cur_market_iv) and not np.isnan(cur_model_iv):
            if abs(cur_model_iv - cur_market_iv) < 0.005:
                exit_reason = "gap_closed"

        if exit_reason:
            # In live mode, only finalize the exit if we can price and submit a capped
            # marketable-limit order. If no NBBO is available, leave the position open
            # and retry next tick rather than crossing blind with a market order.
            if not dry_run:
                ba = _nbbo_bid_ask(chain, pos.strike, pos.maturity)
                if ba is None:
                    if verbose:
                        print(f"  EXIT deferred {pos.ticker} {pos.strike:.0f} "
                              f"({exit_reason}) — no NBBO, retry next tick")
                    continue
                bid, ask = ba
                exit_action = "SELL" if pos.direction == "buy" else "BUY"
                band = config.EXIT_SLIPPAGE_BPS / 10_000.0
                exit_limit = round(
                    bid * (1.0 - band) if exit_action == "SELL" else ask * (1.0 + band), 2
                )
                try:
                    exit_contract = OptionContract(
                        underlying=pos.ticker, expiry=pos.expiry,
                        strike=pos.strike, right="C",
                    )
                    exit_order = broker.submit_option_order(
                        contract=exit_contract, action=exit_action,
                        qty=pos.qty, order_type="LMT",
                        limit_price=exit_limit, close=True,
                    )
                    pos.exit_order_id = exit_order.order_id
                except Exception as e:
                    print(f"  [!] Exit order failed: {e}")
                    continue

            if verbose:
                print(f"  EXIT {pos.direction.upper()} {pos.qty}x {pos.ticker} "
                      f"{pos.strike:.0f} exp={pos.expiry} "
                      f"reason={exit_reason} pnl={pos.cumulative_pnl:+.4f}")
            pos.exited = True
            pos.exit_date = today
            pos.exit_reason = exit_reason

    state.session_pnl += tick_pnl

    # ── 11. Delta rebalance ───────────────────────────────────────────
    positions_for_hedge = [
        OptionPosition(
            ticker=p.ticker,
            strike=p.strike,
            maturity=p.maturity,
            option_type="C",
            qty=(p.qty if p.direction == "buy" else -p.qty),
            spot=spot,
            r=chain.r,
            q=chain.q,
            implied_vol=p.entry_market_iv,
        )
        for p in state.active_positions
    ]

    port_delta = portfolio_delta(positions_for_hedge) if positions_for_hedge else 0.0
    hedge_result = optimize_hedge(
        positions=positions_for_hedge,
        current_hedge=state.current_hedge,
        delta_band=_DELTA_BAND,
        max_hedge_shares=_MAX_HEDGE_SHARES,
        cost_per_share=config.EQUITY_BID_ASK_BPS / 10_000 * spot,
    )

    if abs(hedge_result.hedge_trade) >= 1:
        qty_trade = int(round(hedge_result.hedge_trade))
        if verbose:
            print(f"  HEDGE {'+' if qty_trade > 0 else ''}{qty_trade} shares  "
                  f"delta={port_delta:+.3f}  net_after={hedge_result.net_delta_after:+.3f}")
        if not dry_run:
            try:
                broker.submit_hedge_order(underlying=ticker, qty=qty_trade)
            except Exception as e:
                print(f"  [!] Hedge order failed: {e}")
        state.current_hedge = int(round(hedge_result.new_hedge))
    else:
        if verbose:
            print(f"  HEDGE none needed (delta={port_delta:+.3f} within band "
                  f"+/-{_DELTA_BAND})")

    n_exited = sum(1 for p in state.open_positions if p.exit_date == today)
    active = state.active_positions

    # ── Summary ───────────────────────────────────────────────────────
    if verbose:
        print(f"  Summary: {n_entered} entered  "
              f"{n_exited} exited  "
              f"{len(active)} open  "
              f"tick_pnl={tick_pnl:+.4f}  session_pnl={state.session_pnl:+.4f}")

    # ── Tick log (JSONL) ──────────────────────────────────────────────
    _append_tick_log(tick_log_path, {
        "tick": state.n_ticks + 1,
        "time": now,
        "date": today,
        "spot": round(spot, 4),
        "r": round(float(chain.r), 6),
        "q": round(float(chain.q), 6),
        "n_raw": len(chain.strikes),
        "n_clean": surface.n_options,
        "n_cal": int(len(cal.strikes)),
        "kappa": round(float(fitted.kappa), 4),
        "theta": round(float(fitted.theta), 6),
        "xi": round(float(fitted.xi), 4),
        "rho": round(float(fitted.rho), 4),
        "v0": round(float(fitted.v0), 6),
        "rmse": round(float(rmse), 6),
        "feller_ok": bool(2 * fitted.kappa * fitted.theta >= fitted.xi ** 2),
        "n_signals_raw": len(sm.signals),
        "n_signals_after_uncertainty": len(unc_signals),
        "n_signals_filtered": len(raw_signals),
        "rmse_gate_suppressed": bool(rmse_suppressed),
        "signal_max_rmse": config.SIGNAL_MAX_RMSE,
        "gap_diag": gap_diag,
        "n_entered": n_entered,
        "n_exited": n_exited,
        "open_positions": len(active),
        "tick_pnl": round(tick_pnl, 6),
        "session_pnl": round(state.session_pnl, 6),
    })

    state.last_tick_time = now
    state.n_ticks += 1
    return state


# ── Main entry point ──────────────────────────────────────────────────────────

def run_live(
    ticker: str,
    broker: Optional[Broker] = None,
    loader=None,
    polygon_api_key: Optional[str] = None,
    interval_minutes: int = config.HEDGE_INTERVAL_HOURS * 60,
    n_ticks: Optional[int] = None,
    dry_run: bool = True,
    reset_state: bool = False,
    verbose: bool = True,
    state_path: Path = _STATE_PATH,
    kill_log_path: Path = _KILL_LOG_PATH,
    kill_flag_path: Path = _KILL_FLAG_PATH,
    tick_log_path: Optional[Path] = None,
    chain_log_dir: Optional[Path] = None,
) -> None:
    """Run the live trading loop indefinitely (or for n_ticks if specified).

    Data source: PolygonLoader (Massive) by default; pass loader=AlpacaLoader(...)
    to use Alpaca's indicative feed instead (requires only Alpaca keys, no Polygon plan).

    Args:
        ticker: Underlying symbol ("SPX", "SPY", etc.)
        broker: Broker instance. Defaults to PaperBroker (no real orders).
        loader: Options chain loader. Defaults to PolygonLoader if POLYGON_API_KEY is set,
            otherwise raises. Pass AlpacaLoader() to use Alpaca data.
        polygon_api_key: Massive API key. Falls back to POLYGON_API_KEY env var.
            Ignored when loader is provided explicitly.
        interval_minutes: Minutes between ticks. Default: HEDGE_INTERVAL_HOURS * 60.
        n_ticks: If set, stop after this many ticks.
        dry_run: If True, no real orders submitted (orders logged locally as DRY-XXXX).
            Pass dry_run=False to submit orders to the broker (paper or live).
        reset_state: If True, archive old state and start fresh.
        verbose: Print tick-level detail.
        state_path: Override path for live_state.json.
    """
    if broker is None:
        broker = PaperBroker(log_path=Path("data/paper_log.json"))
        dry_run = True

    if loader is None:
        poly_key = polygon_api_key or os.environ.get("POLYGON_API_KEY")
        if not poly_key:
            raise ValueError(
                "No data loader provided. Either pass loader=AlpacaLoader() "
                "or set POLYGON_API_KEY in .env for Massive."
            )
        loader = PolygonLoader(api_key=poly_key, call_only=False)
        data_label = "Massive"
    else:
        data_label = type(loader).__name__

    if verbose:
        print(f"[loop] Data: {data_label}  Execution: {type(broker).__name__}")

    kc = KillCondition(log_path=kill_log_path, flag_path=kill_flag_path)

    if reset_state:
        state = LiveState.reset(state_path)
        print(f"[loop] State reset. Fresh start.")
    else:
        state = LiveState.load(state_path)
        if state.n_ticks > 0:
            print(f"[loop] Resuming from tick {state.n_ticks}  "
                  f"last_tick={state.last_tick_time}  "
                  f"open={len(state.active_positions)}")

    mode = "DRY RUN" if dry_run else "LIVE"
    print(f"[loop] Starting {mode} loop — {ticker}  "
          f"interval={interval_minutes}min  "
          f"broker={type(broker).__name__}")

    tick = 0
    while True:
        state = _run_tick(ticker, loader, broker, kc, state, dry_run, verbose,
                          tick_log_path=tick_log_path, chain_log_dir=chain_log_dir)
        state.save(state_path)
        tick += 1

        if n_ticks is not None and tick >= n_ticks:
            print(f"\n[loop] Completed {n_ticks} ticks. Exiting.")
            break

        if kc.is_halted():
            print(f"\n[loop] Kill condition triggered. Halting loop.")
            break

        interval_sec = interval_minutes * 60
        next_tick = datetime.now(timezone.utc) + timedelta(seconds=interval_sec)
        print(f"\n[loop] Sleeping {interval_minutes}min. Next tick: "
              f"{next_tick.strftime('%H:%M:%S')} UTC")
        time.sleep(interval_sec)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    # Load .env before argparse so keys are in os.environ
    _load_dotenv()

    parser = argparse.ArgumentParser(
        description="Heston-arb live trading loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  Paper trading (Alpaca data + execution):\n"
            "    python -m trading.loop --ticker SPY --alpaca --live\n"
            "  Polygon data + Alpaca paper execution:\n"
            "    python -m trading.loop --ticker SPY --alpaca --live --polygon-key YOUR_KEY\n"
            "  Local simulation only (no orders):\n"
            "    python -m trading.loop --ticker SPY\n"
        ),
    )
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument(
        "--polygon-key",
        default=None,
        help="Massive (Polygon.io) API key. If omitted and --alpaca is set, "
             "Alpaca data feed is used instead.",
    )
    parser.add_argument("--interval", type=int, default=240, help="Minutes between ticks")
    parser.add_argument("--ticks", type=int, default=None, help="Stop after N ticks")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Submit real orders to the broker. Without this flag, orders are "
             "simulated locally (DRY RUN). Required even for Alpaca paper trading.",
    )
    parser.add_argument("--reset", action="store_true", help="Reset state and start fresh")
    parser.add_argument("--alpaca", action="store_true",
                        help="Use Alpaca paper account for execution (and data if no Polygon key)")
    parser.add_argument("--alpaca-live", action="store_true",
                        help="Use Alpaca live account for execution (implies --live)")
    parser.add_argument("--ibkr", action="store_true", help="Use IBKR broker (requires TWS/Gateway)")
    parser.add_argument("--ibkr-port", type=int, default=7497, help="IBKR port (7497=paper, 7496=live)")
    parser.add_argument("--tick-log", default=str(_TICK_LOG_PATH),
                        help="Path to append the per-tick JSONL log (incl. gap/uncertainty "
                             "diagnostics). Pass '' to disable. Default: data/tick_log.jsonl")
    parser.add_argument("--state-path", default=str(_STATE_PATH),
                        help="Path to live_state.json (positions/params/counters). "
                             "Default: data/live_state.json")
    parser.add_argument("--kill-log", default=str(_KILL_LOG_PATH),
                        help="Path to the Popper kill log. Default: data/kill_log.json")
    parser.add_argument("--kill-flag", default=str(_KILL_FLAG_PATH),
                        help="Path to the kill flag. Default: data/kill_flag.json")
    parser.add_argument("--chain-log-dir", default=None,
                        help="Directory for per-tick cleaned-chain Parquet capture "
                             "(training dataset): <dir>/<ticker>/<date>.parquet. "
                             "Omit to disable.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    dry_run = not (args.live or args.alpaca_live)

    # ── Broker ────────────────────────────────────────────────────────────────
    if args.alpaca or args.alpaca_live:
        from trading.broker import AlpacaBroker
        paper = not args.alpaca_live
        broker = AlpacaBroker(paper=paper)
        mode = "paper" if paper else "LIVE"
        print(f"[loop] Alpaca broker ({mode})")
    elif args.ibkr:
        from trading.broker import IBKRBroker
        broker = IBKRBroker(port=args.ibkr_port)
        broker.connect()
        print(f"[loop] Connected to IBKR on port {args.ibkr_port}")
    else:
        broker = PaperBroker()
        print("[loop] PaperBroker (no real orders)")

    # ── Data loader ───────────────────────────────────────────────────────────
    # Explicit --polygon-key always uses Polygon.
    # --alpaca / --alpaca-live uses AlpacaLoader (even if POLYGON_API_KEY is in env).
    # Fallback: POLYGON_API_KEY env var only when no broker flag was set.
    explicit_polygon_key = args.polygon_key
    if explicit_polygon_key:
        loader = PolygonLoader(api_key=explicit_polygon_key, call_only=False)
    elif args.alpaca or args.alpaca_live:
        loader = AlpacaLoader(call_only=False)
    elif os.environ.get("POLYGON_API_KEY"):
        loader = PolygonLoader(api_key=os.environ["POLYGON_API_KEY"], call_only=False)
    else:
        parser.error(
            "No data source available. Either set POLYGON_API_KEY in .env "
            "or use --alpaca to use Alpaca's options feed."
        )

    try:
        run_live(
            ticker=args.ticker,
            broker=broker,
            loader=loader,
            interval_minutes=args.interval,
            n_ticks=args.ticks,
            dry_run=dry_run,
            reset_state=args.reset,
            verbose=not args.quiet,
            state_path=Path(args.state_path),
            kill_log_path=Path(args.kill_log),
            kill_flag_path=Path(args.kill_flag),
            tick_log_path=Path(args.tick_log) if args.tick_log else None,
            chain_log_dir=Path(args.chain_log_dir) if args.chain_log_dir else None,
        )
    finally:
        if args.ibkr and hasattr(broker, "disconnect"):
            broker.disconnect()


if __name__ == "__main__":
    main()
