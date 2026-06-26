# Live Re-enable — Validation TODO
Updated: 2026-06-26

## Status: LIVE again, but NOT end-to-end validated
On **2026-06-26** the `live-tick` workflow was **RE-ENABLED** (`gh workflow enable live-tick`),
now running the **fixed** code on `master` (fast-forward merge of `fix/option-type-reconciliation-mtm`,
commit `c4cee47`). The fixes are **unit-validated only** — the end-to-end Colab validation and a live
broker-positions smoke test have **NOT** been run yet. It was re-enabled deliberately ahead of that;
**this file is the reminder to close that validation debt.**

It is Alpaca **PAPER** (no real money), but the strategy can now place *fillable* paper orders, so it
will actually trade. The workflow runs `master` (cron-job.org dispatches `ref: master`; the in-repo
`schedule:` also runs the default branch), state lives on the `live-data` branch.

## What was fixed (full context: docs/context/live-automation.md)
The live loop had been a market no-op writing fiction: it traded the put/call-merged IV surface as if
every strike were a call, so put-wing signals went out as **deep-ITM CALL** orders priced off the cheap
OTM put (~$0.04) and **never filled**; it then booked them as held (no fill check) and marked the
phantom book with a vega term re-added every tick → fictional **−$5k** P&L while the real account was
flat (~$0, 1 stray SPY share). Fixes (all on master now):
- **Bucket 1** — trade the OTM instrument (`right = "P" if K<F else "C"`), priced off its own NBBO.
- **Bucket 2** — `get_positions()` reconciliation: drops never-filled phantoms, syncs held qty + the
  real fill price, resets the hedge to the broker's real shares (`_reconcile_positions` in trading/loop.py).
- **Bucket 3** — real mark-to-market P&L (a level, not accumulated); `session_pnl` recomputed each tick.
- **Bucket 4** — moneyness gate `MAX_SIGNAL_LOG_MONEYNESS` 0.20→0.10 (near-ATM only); stop-loss 50% /
  take-profit 100% on real MTM P&L; exit priority stop_loss→take_profit→gap_closed→expiry→max_hold.

## TODO before trusting it (do when you have time)
1. [ ] **Run `evals/colab_validate_fixes.ipynb` on Colab** (clones the branch; needs JAX). Confirm all
       four "Bucket N PASS" prints — proves Buckets 1–4 end-to-end on the captured chains. This is the
       Popper gate that was skipped.
2. [ ] **Live `get_positions` smoke test.** On the FIRST fixed live tick, `_reconcile_positions` should
       drop the 30 legacy phantoms and reset the hedge. Check `tick_log.jsonl` (live-data branch) for
       `recon_dropped > 0` and `session_pnl` resetting from ~−5172 to ~0. The only unverified piece is
       the Alpaca SDK field mapping: `avg_entry_price` / `unrealized_pl` / `asset_class` / `side` / `qty`
       (in `AlpacaBroker.get_positions`, trading/broker.py).
3. [ ] **Watch the first session on the Alpaca web UI.** Entries should be **OTM puts/calls** (NOT the
       old 600–616 deep-ITM calls), at real prices, that actually **FILL** (Filled Qty > 0).
4. [ ] **Decide if there's real net-of-cost edge** now that trades are near-ATM. Calibration RMSE (~0.09)
       + near-ATM signal counts are the first evidence; a true backtest needs ≥~20 captured days or
       purchased NBBO history (see memory: historical-edge-test).
5. [ ] **Retune** `STOP_LOSS_FRAC` / `TAKE_PROFIT_FRAC` / `MAX_SIGNAL_LOG_MONEYNESS` once the live P&L
       distribution is visible.

## Kill switch (if it misbehaves)
- **Halt immediately:** `gh workflow disable live-tick` (optionally also pause the cron-job.org job so it
  stops getting failed-POST alerts).
- **Reset corrupted state:** `LiveState.reset()` (archives + clears `live_state.json`) on the live-data branch.

## Where to look
- Per-tick ledger: `tick_log.jsonl` on the `live-data` branch (now carries `recon_dropped`/`recon_untracked`).
- Ground truth: the Alpaca paper web UI (positions, fills, P&L).
- Code: `trading/loop.py` (`_reconcile_positions`, section-10 marking/exits), `trading/broker.py`
  (`get_positions`), `config.py` (gate + stop/TP), `evals/colab_validate_fixes.ipynb` (the validator).
