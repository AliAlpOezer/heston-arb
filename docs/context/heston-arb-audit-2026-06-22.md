# Heston-Arb Production Audit — Context Pack
Updated: 2026-06-22

## Goal & Core Decision
Audit the live Colab production system for misalignments, inefficiencies, and improvement
opportunities, ranked by impact on the real objective: **maximize profit at minimum risk**,
consistent with the encoded philosophies (Popper / Soros / Munger / Buffett / Graham).

## Method
10-agent fan-out (one per subsystem), each finding adversarially verified by an independent
skeptic. 76 findings examined; verification phase was cut short by a session limit after 9
were fully verified. The remaining high-value findings were verified by hand (direct code reads).
Confidence tiers below: **[V]** workflow-verified, **[H]** hand-verified this pass, **[F]** finder-flagged, not yet independently verified.

---

## ROOT CAUSE (the one change that cascaded)
Production switched from the **documented design** (Polygon/Massive data, 240-min/daily ticks)
to **Alpaca calls-only at `interval_minutes=5`** (`colab.ipynb` Cell 6) — but the code's
time bookkeeping, kill counting, Tikhonov anchor, hedge frequency, and Kelly horizon were all
written for daily/4-hourly ticks. Most P0/P1 bugs below are this mismatch surfacing.

The plan lives in `trading-master/context.md` and `docs/context/heston-arb.md`; the `memory/`
dir is empty. Neither was updated when the cadence/data-source changed.

---

## REMEDIATION STATUS (2026-06-22)
**Bucket 1 — DONE & verified:** #1 wall-clock age (`loop.py`), #2 calendar TTM (`loop.py`),
#3 hedge ×100 contract multiplier (`hedging.py`, `config.CONTRACT_MULTIPLIER`). Smoke-tested:
portfolio_delta 528 shares for 10 ATM calls; 5-min-old position age = 0.0035 d (no force-exit).

**Bucket 2 — DONE & verified:** #4 market-hours gate (`loop._market_is_open` + `broker.is_market_open`),
#5 hybrid kill (daily worst-of-day in `kill.py` + intraday breaker `POPPER_KILL_TICKS` in `loop.py`/`state.py`),
#6 pure-RMSE kill metric (`calibrator.calibration_rmse`), #7 capped marketable-LMT exits + drop $1.00 entry
fallback (`loop.py`, `config.EXIT_SLIPPAGE_BPS`). Kill suite 10/10; compile + smoke green.

**Bucket 3 — DONE & verified (the profit-critical signal unblock):** calls+puts calibration
(`call_only=False` + OTM-side dedup in `cleaner`), calibration grid widened across moneyness,
hard-Feller projection (`constraints.project_to_feller`, applied in `calibrate`), signals restricted
to the calibrated maturity range, cost-aware Buffett gate (`loop._filter_by_transaction_cost`).
Verified: synthetic recovery 4/4 PASS (incl. near-Feller-boundary), and the projection turns the
live tick's Feller-violated fit (margin −0.00116) into a valid one (+0.00092) — the L1 fix for 0-trades.

**Deferred:** #13 P&L mark-to-market, #14 naked-short caps, philosophy gaps (Soros/Munger), P3 cleanups.

---

## FINDINGS (priority = profit/risk impact)

### P0 — Critical (strategy is unprofitable / unsafe by construction)

**1. [V] Position clock counts ticks, not days — every position force-exits in ~50 min.**
`trading/loop.py:427` `pos.age_days += 1` runs once per *tick*; `:451` exits on `age_days >= 10`.
At 5-min ticks, "10 days" becomes **10 ticks = 50 minutes**. Every trade is flattened ~50 min
after entry, far inside its ~5-day mean-reversion thesis (`sizing.py:40`), paying round-trip
spread + MKT-exit slippage with near-zero chance of capturing the modeled convergence. *This
single bug likely makes the strategy a guaranteed cost-bleed.*
**Fix:** derive age from wall-clock: `age_days = (now - entry_dt).total_seconds()/86400`; exit on a real `MAX_HOLD_DAYS`.

### P1 — High (material profit leak or broken risk control)

**2. [V] Expiry exit math mixes days and ticks.** `loop.py:448` `ttm_days = pos.maturity*365 - pos.age_days`
subtracts a tick count from calendar days, and `maturity` is never decremented. Any position
entered with 5–15 days to expiry trips a spurious `"expiry"` exit within minutes.
**Fix:** `ttm_days = (date(expiry) - date(today)).days`; drop the `age_days` term.

**3. [H] Delta hedge is ~100× under-sized — the book is effectively unhedged.** `risk/hedging.py:111`
`portfolio_delta` returns `Σ qty·delta` with **no 100-share contract multiplier**; the stock hedge
(`loop.py:516`, `broker.submit_hedge_order(qty=shares)`) is in single shares. So 10 call contracts
(~500 share-deltas of exposure) get hedged with ~5 shares. The entire P&L/vega/sizing stack is
per-share while real fills are per-100-share-contract. Directly violates the "minimum risk" goal.
**Fix:** multiply option deltas (and vega, and P&L) by `CONTRACT_MULTIPLIER = 100`.

**4. [V] No market-hours gate.** `loop.py:641-659` ticks unconditionally; Alpaca's indicative feed
serves frozen last-NBBO when closed. Overnight/weekend ticks calibrate on stale quotes, write that
RMSE into the Popper kill log (poisoning the one falsifiable risk control), age positions, and queue
orders against the next open. **Fix:** first guard in `_run_tick` → `if not broker.is_market_open(): bump tick & return` (Alpaca `get_clock().is_open`).

**5. [V] Popper kill is day-keyed + last-write-wins → blind intraday.** `loop.py:321` records keyed by
`today`; `kill.py:127` overwrites same-day records, so only each day's *last* tick survives and the
halt needs **5 consecutive calendar days** (`POPPER_KILL_DAYS=5`). A day miscalibrated 287/288 ticks
but with one good final tick never trips. The core risk halt cannot fire intraday.
**Fix:** key by tick timestamp and count failing *ticks*, or keep daily but aggregate `max(rmse)` per day.

**6. [H] Kill threshold is fed the wrong number.** `calibrator.py:81,148` returns `best_loss = RMSE +
Tikhonov + Feller_penalty`; `loop.py:306,321` records that as `rmse` and compares to
`CALIBRATION_FAIL_RMSE=0.10`. The Popper kill can fire on regularization, and the reported/logged
"RMSE" is not fit quality. **Fix:** return pure log-IV RMSE separately from total loss; kill and log on the pure RMSE.

**7. [V] Exits are uncapped MKT orders; entries have a $1.00 fallback.** `loop.py:473-477` exits via
`order_type="MKT"` with no slippage cap on wide/thin call strikes; `loop.py:369` `or 1.0` submits a
$1.00 limit when the contract is missing from the snapshot. Amplified by bug #1 (a forced round-trip
every ~50 min). **Fix:** marketable LMT capped to a bps band off fresh NBBO; drop the `or 1.0` (skip entry instead).

**8. [V/H] Stale docs hide the daily-tick assumption.** `loop.py:6,29,574,586` still say "Polygon" /
240-min default. Harmless at runtime (caller passes explicit args) but it's the only in-code marker of
the day-based design that bugs #1/#2 silently violate. **Fix:** update docs *after* fixing the clock; add a warning that age/halflife are day-denominated.

### P2 — Medium (degraded edge / methodology drift)

**9. [H] Buffett margin-of-safety gate omits transaction cost.** `mispricing.py:164` gates only on
`|gap| < MIN_VOL_GAP (0.015)`. The docstring promises "after calibration uncertainty AND transaction
cost," but the spread you pay to enter+exit is never subtracted (uncertainty is gated separately).
Marginal 1.5-vol-pt gaps on wide call spreads are negative-EV after costs. **Fix:** require `|gap| > MIN_VOL_GAP + round_trip_spread_in_vol + k·posterior_sigma`.

**10. [H] Signals trade where the model never fit.** `mispricing.py:149` computes model IVs across the
**full** surface, but calibration only fit a sparse 8×8 near-ATM grid (`loop.py:152`). Far-strike
signals are model *extrapolation* — likely false alpha. **Fix:** restrict signals to (or near) the calibration grid, or widen the calibration grid.

**11. [H] Liquidity weighting is dead; OI is hardcoded 0.** `calibrator.py:36` `compute_weights`
(OI/(spread+ε), per CLAUDE.md) is never called — `loop.py:217` uses uniform weights; and
`alpaca_loader.py:238` sets `open_interest=0`. The documented liquidity-weighted loss is doubly
disabled, so noisy illiquid quotes get equal say in the fit. **Fix:** weight by `1/(spread+ε)` (OI is unavailable on the snapshot) in `_build_cal_input`.

**12. [H] Kelly sizing is daily-denominated and forces ≥1 contract.** `sizing.py:116` uses
`IV_DAILY_VOL` and `MEAN_REVERSION_HALFLIFE_DAYS=5` (both daily) while the loop ticks every 5 min;
`:123` `max(1, round(fractional))` enters ≥1 contract even when fractional Kelly rounds to 0, defeating
"bet small on small edge." The formula is also dimensionally a heuristic, not a true ∈[0,1] fraction.
**Fix:** make the horizon consistent with cadence; allow qty 0.

**13. [V] `session_pnl` is a synthetic vega proxy, not P&L.** `loop.py:437` uses `norm.pdf(0.0)`
(ATM vega for every strike), frozen `entry_spot`, non-decremented maturity, no 100× multiplier, and
books zero transaction cost. It can show profit while the account bleeds. *Reporting only — does not
feed sizing/kill.* **Fix:** mark to current NBBO mid vs entry fill (store fill price); reconcile vs `broker.get_positions()`.

**14. [F] Naked short calls with no margin / max-loss cap.** `sizing.py` treats buy/sell symmetrically;
`loop.py:379` SELL + `close=False` opens short calls (unbounded risk). Tension with "minimum risk."
**Verify & fix:** add a margin/notional cap and a per-position max-loss stop for short legs.

### P3 — Low / cleanup (verified, negligible P&L)

- **[V]** Butterfly repair is price-space convexity, not the Durrleman g(k,T) the docs name — but
  `cleaner.py` and `evals/invariants.py` use the *same* form, so gatekeeping is self-consistent. Doc fix only.
- **[V]** `surface.py:133` reuses `BUTTERFLY_TOLERANCE` for the calendar check; output is unused by the loop. Add `CALENDAR_TOLERANCE`.
- **[F]** Misc to confirm on the next verification pass: Tikhonov in unconstrained space / anchored to 5-min-ago params;
  JIT recompile per tick if grid shape varies; Laplace 200-samples every 5 min (waste); spot fetched via a separate
  quote (timestamp skew) with bid-only downward bias; FRED rate staleness; Alpaca order reconciliation / OCC routing;
  state-resume vs PaperBroker positions; IBKR `cancel_order` stub.

---

## PHILOSOPHY ALIGNMENT SCORECARD
- **Popper (kill condition):** present but degraded at 5-min cadence (findings #5, #6). Day-granular, slow, fed wrong metric.
- **Buffett (margin of safety):** partial. Fractional Kelly **0.25× IS applied** (`sizing.py:36`) ✅; uncertainty gate exists ✅; **transaction-cost margin missing** (#9).
- **Soros (reflexivity/crowding):** **not implemented** anywhere — no crowding scale-down. [F]
- **Munger (multi-model/inversion):** **single Heston model, no ensemble / bull-bear**. [F]
- **Graham (price-vs-value):** N/A for vol-arb (no equity DCF screen). Acceptable scope cut.
- **Mandatory multi-regime backtest (2008/2018/2022):** not present. [F]

## WHAT'S CORRECT — DO NOT "FIX"
- `bs_delta` sign convention (call ∈(0,1), put ∈(-1,0)) — `hedging.py:93`.
- Feller gate returns empty signals when violated — `mispricing.py:139` (conservative, good).
- Calibration loss masks NaN model-IVs before squaring — `calibrator.py:69` (correct JAX 0·NaN handling).
- Fractional Kelly 0.25× is applied — `sizing.py:36`.
- Kill logic is sound *at day granularity* — only the cadence/metric inputs are wrong.

## RECOMMENDED REMEDIATION ORDER
1. **#1 + #2** (wall-clock age/expiry) — unblocks the whole thesis.
2. **#3** (100× hedge multiplier) — restores delta-neutrality.
3. **#4** (market-hours gate) — stops feeding garbage to kill/orders.
4. **#5 + #6** (kill cadence + correct RMSE metric) — restores the falsifier.
5. **#7 + #9 + #14** (execution slippage, cost-aware gate, short-leg caps) — stop the cost bleed.
6. Then #10–#13 (model-fit hygiene), then philosophy gaps (Soros/Munger), then P3.

## KEY FILES
- `trading/loop.py` — 11-step tick; bugs #1,#2,#4,#5,#6,#7,#13 live here.
- `risk/hedging.py:99-113` — missing 100× multiplier (#3).
- `calibration/calibrator.py:81,148` — total-loss-as-RMSE (#6); dead liquidity weights (#11).
- `signals/mispricing.py:149,164` — extrapolated signals (#10), cost-blind gate (#9).
- `signals/sizing.py:116,123` — daily-unit Kelly, forced ≥1 contract (#12).
- `evals/kill.py:127` — day-keyed overwrite (#5).
- `data/alpaca_loader.py:238` — OI hardcoded 0 (#11).
