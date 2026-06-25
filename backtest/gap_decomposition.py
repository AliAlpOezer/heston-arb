"""
Gap-decomposition diagnostic — the decisive test of the edge thesis.

THE CLAIM UNDER TEST (§1 of the edge analysis):
  The Heston model recalibrates every session with weak regularization, so it *chases the
  market*. When a mispricing "gap" closes, it closes mostly because the next session's re-fit
  moved the MODEL toward the market — not because the MARKET moved toward yesterday's model.
  But a vol-arb position only earns P&L from MARKET moves. If true, realized edge ~= 0.

DECOMPOSITION (one option contract, session t -> next session t+1; gap_t = model_iv_t - market_iv_t):
    Δmarket = market_iv_{t+1} - market_iv_t
    Δmodel  = model_iv_{t+1}  - model_iv_t
    Identity:  Δgap = Δmodel - Δmarket
    market_contribution = sign(gap_t) * Δmarket        # the ONLY part that earns P&L
    model_contribution  = sign(gap_t) * (-Δmodel)      # earns NOTHING (model chasing market)
    total_closure       = market_contribution + model_contribution  ( = -sign(gap_t)*Δgap )

VERDICT: if mean market_contribution <= 0 or its CI spans 0 while model_contribution
dominates, the gap closes by the model chasing the market -> strategy structurally dead.

Data caveat: prices are Alpaca trade closes (no bid/ask). This measures convergence
DIRECTION, not net-of-cost P&L. It can kill the thesis; it cannot prove a durable edge.

Run under PowerShell with .env loaded:
    python -m backtest.gap_decomposition --start 2024-06-03 --end 2024-06-28
"""

import argparse
import csv
import math
from collections import defaultdict
from datetime import date, timedelta
from typing import Optional

import jax
jax.config.update("jax_enable_x64", True)   # MUST precede heston/calibrator import
import jax.numpy as jnp
import numpy as np

import config
from calibration.heston import HestonParams, heston_implied_vols, feller_satisfied
from calibration.calibrator import CalibrationInput, calibrate
from backtest.alpaca_history import AlpacaHistoryLoader, HistChain

_DEFAULT_INIT = HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04)


# ── Per-session pipeline ────────────────────────────────────────────────────────

def _calibrate_session(
    chain: HistChain,
    init: HestonParams,
    prev: Optional[HestonParams],
    n_steps: int,
) -> Optional[dict]:
    """Calibrate Heston to one session and return per-contract market/model IVs.

    Returns None if the session can't be fit (too few valid IVs). Restricts the
    calibration set to |log-moneyness| <= MAX_CAL_LOG_MONEYNESS so a single-factor
    Heston can actually fit it (audit finding #10 / config note)."""
    mkt_iv = chain.market_ivs()
    S, r, q = chain.spot, chain.r, chain.q
    K = chain.strikes
    T = chain.maturities
    F = S * np.exp((r - q) * T)
    k = np.log(K / F)

    valid = ~np.isnan(mkt_iv) & (np.abs(k) <= config.MAX_CAL_LOG_MONEYNESS)
    if valid.sum() < 8:
        return None

    cal = CalibrationInput(
        strikes=jnp.array(K[valid]),
        maturities=jnp.array(T[valid]),
        market_ivs=jnp.array(mkt_iv[valid]),
        weights=jnp.ones(int(valid.sum())) / int(valid.sum()),  # uniform — as live loop runs
        S=float(S), r=float(r), q=float(q),
    )
    fitted, _ = calibrate(cal, initial_params=init, prev_params=prev, n_steps=n_steps)

    # Model IVs for ALL contracts (so we can match any tracked contract next day).
    model_iv = np.array(heston_implied_vols(S, jnp.array(K), jnp.array(T), r, q, fitted))

    # Pure log-IV RMSE on the calibration set (unambiguous; not the total loss).
    m_cal = model_iv[valid]
    ok = ~np.isnan(m_cal)
    rmse = float(np.sqrt(np.mean(
        (np.log(m_cal[ok] + 1e-8) - np.log(mkt_iv[valid][ok] + 1e-8)) ** 2
    ))) if ok.sum() else np.inf

    sess_date = chain.snapshot_time[:10]
    d = date.fromisoformat(sess_date)
    contracts = {}
    for i in range(len(K)):
        if np.isnan(mkt_iv[i]) or np.isnan(model_iv[i]):
            continue
        ttm_days = int(round(T[i] * 365))
        expiry = (d + timedelta(days=ttm_days)).isoformat()
        key = (expiry, round(float(K[i]), 1), str(chain.option_type[i]))
        contracts[key] = {
            "market_iv": float(mkt_iv[i]),
            "model_iv": float(model_iv[i]),
            "gap": float(model_iv[i] - mkt_iv[i]),
            "T": float(T[i]),
            "k": float(k[i]),
        }
    return {"date": sess_date, "params": fitted, "rmse": rmse,
            "feller": feller_satisfied(fitted), "contracts": contracts}


# ── Decomposition over consecutive sessions ──────────────────────────────────────

def build_pairs(sessions: list[dict]) -> list[dict]:
    """Match contracts across consecutive sessions and decompose each gap change."""
    rows = []
    for a, b in zip(sessions, sessions[1:]):
        ca, cb = a["contracts"], b["contracts"]
        for key in ca.keys() & cb.keys():
            ra, rb = ca[key], cb[key]
            gap_t = ra["gap"]
            sgn = 1.0 if gap_t > 0 else -1.0
            d_market = rb["market_iv"] - ra["market_iv"]
            d_model = rb["model_iv"] - ra["model_iv"]
            market_contrib = sgn * d_market
            model_contrib = sgn * (-d_model)
            rows.append({
                "date_t": a["date"], "date_t1": b["date"],
                "expiry": key[0], "strike": key[1], "type": key[2],
                "T": ra["T"], "k": ra["k"], "abs_k": abs(ra["k"]),
                "gap_t": gap_t, "abs_gap_t": abs(gap_t),
                "rmse_t": a["rmse"], "feller_t": a["feller"],
                "d_market": d_market, "d_model": d_model,
                "market_contrib": market_contrib,
                "model_contrib": model_contrib,
                "closure": market_contrib + model_contrib,
                # Tradeable signal: passes the live pipeline's full gate (incl. RMSE).
                "is_signal": (abs(gap_t) > config.MIN_VOL_GAP
                              and abs(ra["k"]) <= config.MAX_SIGNAL_LOG_MONEYNESS
                              and a["rmse"] <= config.SIGNAL_MAX_RMSE
                              and a["feller"]),
                # Gap point: same but WITHOUT the RMSE gate — so the decomposition still has
                # a sample to answer "does the market move toward the model?" when fits are
                # too poor to trade (the math is valid regardless of fit quality).
                "is_gap_point": (abs(gap_t) > config.MIN_VOL_GAP
                                 and abs(ra["k"]) <= config.MAX_SIGNAL_LOG_MONEYNESS
                                 and a["feller"]),
            })
    return rows


# ── Stats ────────────────────────────────────────────────────────────────────────

def _boot_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 0) -> tuple[float, float]:
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _stat(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": np.nan, "t": np.nan, "lo": np.nan, "hi": np.nan, "hit": np.nan}
    mean = float(x.mean())
    sd = float(x.std(ddof=1)) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if sd > 0 else np.nan
    lo, hi = _boot_ci(x)
    return {"n": n, "mean": mean, "t": t, "lo": lo, "hi": hi,
            "hit": float((x > 0).mean())}


def summarize(rows: list[dict]) -> dict:
    sig = [r for r in rows if r["is_signal"]]
    gap = [r for r in rows if r["is_gap_point"]]
    allp = rows
    out = {"n_pairs": len(rows), "n_signals": len(sig), "n_gap_points": len(gap)}
    for name, group in (("signal", sig), ("gap_point", gap), ("placebo_all", allp)):
        mc = np.array([r["market_contrib"] for r in group])
        mo = np.array([r["model_contrib"] for r in group])
        out[name] = {
            "market": _stat(mc),
            "model": _stat(mo),
            "closure_mean": float((mc + mo).mean()) if len(mc) else np.nan,
            "market_share": (float(mc.mean() / (mc.mean() + mo.mean()))
                             if len(mc) and (mc.mean() + mo.mean()) != 0 else np.nan),
        }
    # Stratify signals by tenor and moneyness.
    def strat(group, keyfn, buckets):
        d = {}
        for label, pred in buckets:
            g = [r for r in group if pred(keyfn(r))]
            d[label] = _stat(np.array([r["market_contrib"] for r in g]))
        return d
    out["by_tenor"] = strat(gap, lambda r: r["T"] * 365,
                            [("<30d", lambda v: v < 30),
                             ("30-60d", lambda v: 30 <= v < 60),
                             (">=60d", lambda v: v >= 60)])
    out["by_moneyness"] = strat(gap, lambda r: r["abs_k"],
                                [("ATM<0.05", lambda v: v < 0.05),
                                 ("mid0.05-0.12", lambda v: 0.05 <= v < 0.12),
                                 ("wing>=0.12", lambda v: v >= 0.12)])
    return out


# ── Report ───────────────────────────────────────────────────────────────────────

def _pp(s: dict) -> str:  # vol-points: report in basis points of vol for readability
    return (f"n={s['n']:5d}  mean={s['mean']*100:+.3f}vp  t={s['t']:+.2f}  "
            f"95%CI=[{s['lo']*100:+.3f},{s['hi']*100:+.3f}]vp  hit={s['hit']*100:.1f}%"
            if s["n"] else "n=0")


def report(summ: dict, period: str) -> str:
    L = ["=" * 78,
         f"GAP-DECOMPOSITION DIAGNOSTIC — SPY {period}",
         f"  session pairs: {summ['n_pairs']}",
         f"  tradeable signals (pass full gate incl. RMSE<={config.SIGNAL_MAX_RMSE}): {summ['n_signals']}",
         f"  gap points (|gap|>MIN_VOL_GAP, RMSE gate dropped): {summ['n_gap_points']}",
         "  (units: vol points/day. market_contrib = the only part that earns P&L)",
         "=" * 78]
    for name in ("signal", "gap_point", "placebo_all"):
        b = summ[name]
        L += [f"\n[{name}]",
              f"  market_contrib : {_pp(b['market'])}",
              f"  model_contrib  : {_pp(b['model'])}",
              f"  closure/day    : {b['closure_mean']*100:+.3f}vp   "
              f"market_share_of_closure: {b['market_share']*100:.1f}%"]
    L.append("\n[signal market_contrib by tenor]")
    for k, s in summ["by_tenor"].items():
        L.append(f"  {k:>10}: {_pp(s)}")
    L.append("[signal market_contrib by |log-moneyness|]")
    for k, s in summ["by_moneyness"].items():
        L.append(f"  {k:>13}: {_pp(s)}")

    # Base the verdict on tradeable signals if any exist; else fall back to gap points.
    basis = "signal" if summ["n_signals"] > 0 else "gap_point"
    m = summ[basis]["market"]
    note = "" if basis == "signal" else (
        " [NOTE: 0 signals passed the RMSE gate on this data — verdict uses gap points; "
        "if no day is even calibratable to <=0.10 RMSE, the strategy produces no trades at all.]")
    verdict = (
        "KILL-CONSISTENT: market convergence is <=0 or its CI spans 0 — gaps close by the "
        "model chasing the market, which earns nothing." if (m["n"] == 0 or m["lo"] <= 0 <= m["hi"] or m["mean"] <= 0)
        else "SURVIVES this test: market converges toward the model with a CI above 0. "
             "Necessary but NOT sufficient — costs/regime still untested.")
    L += ["\n" + "-" * 78, f"VERDICT (basis={basis}): " + verdict + note, "-" * 78]
    return "\n".join(L)


# ── Driver ────────────────────────────────────────────────────────────────────────

def run(ticker: str, start: str, end: str, n_steps: int, csv_path: Optional[str],
        verbose: bool = True) -> dict:
    loader = AlpacaHistoryLoader()
    days = loader.trading_days(ticker, start, end)
    if verbose:
        print(f"[diag] {ticker}: {len(days)} sessions {start}..{end}")

    sessions, init, prev = [], _DEFAULT_INIT, None
    for day in days:
        try:
            chain = loader.fetch(ticker, day)
        except ValueError as e:
            if verbose:
                print(f"  {day}: skip ({e})")
            continue
        s = _calibrate_session(chain, init, prev, n_steps)
        if s is None:
            if verbose:
                print(f"  {day}: skip (too few valid IVs)")
            continue
        sessions.append(s)
        prev, init = s["params"], s["params"]    # warm-start + Tikhonov anchor
        if verbose:
            print(f"  {day}: spot={chain.spot:.2f} contracts={len(s['contracts'])} "
                  f"rmse={s['rmse']:.4f} feller={s['feller']}")

    rows = build_pairs(sessions)
    if csv_path and rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        if verbose:
            print(f"[diag] wrote {len(rows)} pair-rows -> {csv_path}")

    summ = summarize(rows)
    print("\n" + report(summ, f"{start}..{end}"))
    return summ


def main() -> None:
    p = argparse.ArgumentParser(description="Gap-decomposition edge diagnostic")
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--start", default="2024-06-01")
    p.add_argument("--end", default="2025-12-31")
    p.add_argument("--cal-steps", type=int, default=200)
    p.add_argument("--csv", default=None, help="Path to write per-pair rows")
    args = p.parse_args()
    run(args.ticker, args.start, args.end, args.cal_steps, args.csv)


if __name__ == "__main__":
    main()
