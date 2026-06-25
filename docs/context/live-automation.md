# Live Automation & Monitoring — Context Pack
Updated: 2026-06-25

A fresh agent reading ONLY this file can resume. It documents how the live loop is
scheduled, what it logs, and how health alerts reach the operator (Telegram).

## Goal & Core Decision
Run the Heston-arb tick loop autonomously every 5 min during US market hours, and
make it observable: classify each tick's outcome and push the ones that matter
(first run of day, buy/sell, errors, health warnings) to Telegram. The per-tick
record is the single source of truth; alerts are derived from it.

## How the automation runs (trigger chain)
```
cron-job.org  ──POST workflow_dispatch──▶  GitHub API  ──▶  GHA workflow "live-tick"
   every 5 min                              (PAT, actions=write)     (.github/workflows/tick.yml)
   */5 13-21 * * 1-5  (UTC)                                          │
                                                                     ▼
                                       one tick: fetch→clean→calibrate→signals→size→orders→hedge
                                                                     │
                                          ┌──────────────────────────┤
                                          ▼                          ▼
                                  notifier step (Telegram)   commit state → live-data branch
```
- **Why not GitHub `schedule:`** — GitHub cron is best-effort; it delayed/dropped every
  scheduled fire (zero fired on 2026-06-25). cron-job.org fires reliably. The in-repo
  `schedule: */5 13-21 * * 1-5` is KEPT as a free backup; the workflow `concurrency`
  group (`group: live-tick`, `cancel-in-progress: false`) guarantees a double-fire never
  runs two ticks at once.
- **Market-hours guard**: `trading.loop._market_is_open` (authoritative via the Alpaca
  clock) no-ops any fire outside RTH, so the wide UTC window 13–21 covers both DST
  regimes (summer RTH 13:30–20:00, winter 14:30–21:00) and the first real tick lands at
  the open. Off-hours fires log `status="skip_closed"` and send nothing.

## External scheduler config (cron-job.org)
- URL: `https://api.github.com/repos/AliAlpOezer/heston-arb/actions/workflows/302012980/dispatches`
  (workflow id `302012980`; `tick.yml` also works in the URL)
- Method POST, body `{"ref":"master"}`
- Headers (KEY must be the bare name — no trailing colon): `Authorization: Bearer <PAT>`,
  `Accept: application/vnd.github+json`, `X-GitHub-Api-Version: 2022-11-28`,
  `User-Agent: heston-cron`, `Content-Type: application/json`
- **Timezone: UTC** (crontab `*/5 13-21 * * 1-5`). Success = HTTP 204.
- Token: fine-grained PAT, **Repository → Actions: Read and write** is REQUIRED (repo
  read alone → 403 "Resource not accessible by personal access token"). Current PAT
  expires **2026-07-25**; when it expires, dispatches 4xx and the loop silently stops —
  cron-job.org failure alert + the workflow's `if: failure()` Telegram alert catch it.
- Full guide: `docs/scheduling.md`.

## WHERE THE LOGS ARE  (all committed to the `live-data` branch each tick)
| Artifact | Path (on `live-data`) | What |
|---|---|---|
| **Per-tick ledger** ⭐ | `tick_log.jsonl` | one JSON line/tick: time, spot, r/q, params κθξρv₀, rmse, feller_ok, signals, entries/exits (per-contract), deployed_capital, capital_capped, pnl, `status` |
| Notifier dedup state | `notifier_state.json` | last_tick, first_run_date, halt_active, warn_throttle |
| Portfolio state | `live_state.json` | open_positions (incl. entry_premium), hedge, prev_params, counters |
| Kill log / flag | `kill_log.json`, `kill_flag.json` | Popper rolling RMSE + halt flag |
| Chain snapshots | `chains/SPY/<date>.parquet` | full cleaned chain per tick (training dataset) |
| Raw stdout transcript | GitHub Actions run logs (~90-day retention) | https://github.com/AliAlpOezer/heston-arb/actions/workflows/tick.yml |

Read the ledger: `git fetch origin live-data && git show origin/live-data:tick_log.jsonl | tail -20`
or `gh api repos/AliAlpOezer/heston-arb/contents/tick_log.jsonl?ref=live-data --jq '.content' | base64 -d`.

## Tick `status` values (set by `trading.loop._emit_outcome`; every tick writes exactly one record)
`ok` (full record; carries laplace_ok, intraday_paused, deployed_capital, capital_capped,
entries[], exits[]) · `skip_closed` · `halted` · `err_fetch` · `err_spot` ·
`err_thin_chain` · `err_calib` · `kill_triggered`. Legacy records with no `status` = `ok`.

## Telegram alerts (eval = `evals/health.py`, transport = `trading/notifier.py`)
- Secrets: `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (GH secrets). Bot @hestonArbTraderBot,
  chat_id 7722967711 (operator "Alp").
- Tiers/kinds: **crit** (attaches raw tick) err_fetch, err_calib, kill_triggered, halted ·
  **warn** err_spot, err_thin_chain, rmse_spike, rmse_gate_suppressed, feller_flip,
  laplace_failed, intraday_breaker, gate_problem, capital_capped · **info** first_run,
  buy, sell, resumed · **silent** skip_closed.
- Dedup/throttle in notifier_state: first-run once/day, halt on transition, persistent
  warns throttled to once per `WARN_THROTTLE_TICKS` (12 ≈ 1h).
- **Cold start**: on first run with no notifier_state, suppress PRIOR days but evaluate the
  most recent day as new (so a mid-session deploy still alerts on today's first run/trades).
- Tests: `evals/test_health.py` (19, pure logic, `pytest evals/test_health.py`).

## Capital limit (config)
`MAX_PORTFOLIO_CAPITAL = 100_000.0` — premium-at-risk cap across the open book. Enforced in
`trading.loop` entry loop: seed from open positions' `entry_premium`, fill largest-gap-first,
STOP entering once the next would breach. Long cost = qty×100×price; short cost =
qty×100×spot×`SHORT_MARGIN_FRAC` (0.20, an estimate — verify vs broker margin before live).
Caused by: sizing was contract-count only (a tick once sized a $2.56M cost basis).

## Current State
- Done: structured per-tick outcomes; health eval + Telegram notifier (PR #1); cold-start
  fix (PR #2); $100k capital cap (PR #3); scheduling doc (PR #4); cron-job.org external
  trigger live and confirmed auto-firing on the 5-min grid (first auto fire 2026-06-25 15:15 UTC).
- Running: SPY, Alpaca PAPER (`--alpaca --live`), dry orders off (real paper orders).

## Open Questions / Known Issues
- **Trade quality (unaddressed)**: entries have appeared at deep-ITM strikes (e.g. 604–610
  calls while spot ~734) — odd for vol-arb; signal/strike selection looks off. Cap protects
  capital but the edge is suspect. Investigate `signals/mispricing.py` + grid selection.
- **Hedge-order failures are not captured as a `status`** — `[!] Hedge order failed` is
  stdout-only (e.g. "insufficient buying power"). Add a `hedge_failed` flag to the OK record
  + a notifier alert (same gap class already closed for other paths).
- **Legacy open positions have entry_premium=0** (booked before the cap) so they seed the
  capital check as $0. A `--reset` of state, or backfill, fixes the understatement.
- **PAT expiry 2026-07-25** — rotate before then (it was also pasted in chat once; regenerate).

## Key Files / Interfaces
- `.github/workflows/tick.yml` — runs one tick; notifier step (`if: always()`); job-failure curl alert (`if: failure()`).
- `trading/loop.py` — `_run_tick` (status on every path via `_emit_outcome`), capital cap, entries/exits logging.
- `trading/state.py` — `LivePosition.entry_premium`.
- `evals/health.py` — `evaluate(records, state) -> EvalResult(events, state)` (pure).
- `trading/notifier.py` — Telegram transport + throttle + CLI (`--test`, `--get-chat-id`, `--dry-run`).
- `config.py` — `MAX_PORTFOLIO_CAPITAL`, `SHORT_MARGIN_FRAC`, RMSE/kill thresholds.
- `docs/scheduling.md` — external-trigger setup.
