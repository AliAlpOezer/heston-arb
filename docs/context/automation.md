# Automation (GitHub Actions live loop) — Context Pack
Updated: 2026-06-25

## Goal & Core Decision
Run the heston-arb live trading loop hands-off every NY trading day, open→close,
on free infrastructure, while capturing a per-tick options dataset for future model
training. Core decision it drives: *can a free, scheduled pipeline trade the paper
account and accumulate training data without anyone clicking "Run"?* — Yes, via
GitHub Actions cron firing one tick per run.

## Requirements
- [must] Daily auto-run across NY regular trading hours (13:30–20:00 UTC), no manual start.
- [must] Self-verifying preflight: deterministic deps + secrets, no cell-order fragility.
- [must] Trade Alpaca paper account (`AlpacaBroker(paper=True)`, `dry_run=False`).
- [must] Persist state (positions, Heston params, kill log, tick log) across runs.
- [must] Capture full cleaned options chain per tick as Parquet for later model training.
- [must] Free tier.
- [should] Loop documented to a file (machine: tick_log.jsonl; human: docs/AUTOMATION.md).
- [should] Survive overlapping/late cron fires without double-trading.
## Non-goals
- GPU acceleration (GHA runners are CPU-only; sparse 96-pt grid fits CPU in <5 min/tick).
- Real-capital trading (paper only for now).
- Colab self-scheduling (impossible on free Colab; rejected — see Decisions).
- Holiday calendar (cron fires Mon–Fri; market-hours guard + Alpaca clock skip closed days).

## Decisions (with rationale)
- Platform = GitHub Actions cron, one tick (`--ticks 1`) per run — because free Colab
  cannot self-schedule, and loop.py already guards market hours + resumes from state,
  so per-tick serverless cron is the natural fit. (2026-06-25)
- Repo made PUBLIC, keep true 5-min cadence — because a private repo's 2000 free
  min/mo can't cover ~96 runs/day; public repos get unlimited free Actions minutes.
  Secrets stay in GH Actions Secrets, never in repo. (2026-06-25)
- State + data live on a dedicated orphan branch `live-data`, NOT master — keeps
  ~80 data commits/day out of code history; prunable independently. (2026-06-25)
- Training artifact = full cleaned chain, one Parquet per symbol per day,
  read-append-rewrite each tick — compact, git-manageable, microstructure-complete. (2026-06-25)
- Overlapping runs serialized via workflow `concurrency` group (no cancel) — prevents
  two ticks acting on stale state / double orders. (2026-06-25)

## Current State
- Done: all 4 buckets built.
  - B1: `_append_chain_log` + `chain_log_dir` threaded through `_run_tick`/`run_live`;
    pyarrow added; Parquet schema/concat/daily-naming verified locally.
  - B2: CLI flags `--state-path/--kill-log/--kill-flag/--chain-log-dir`; py_compile clean.
  - B3: `.github/workflows/tick.yml` (cron 13-21 both DST, concurrency serialize, dual
    checkout, pip cache, CPU JAX, secrets→env, push to live-data); YAML validated.
    Added missing `alpaca-py` to requirements; confirmed diffrax unused.
  - B4: `docs/AUTOMATION.md` (setup, data layout, training schema, ops, prune, limits).
- Remaining (USER actions, outward/irreversible — not done by agent):
  1. Make repo public (after history secret-scrub + key rotation if needed).
  2. Add GH Actions secrets: ALPACA_API_KEY, ALPACA_SECRET_KEY, FRED_API_KEY(opt).
  3. Seed `live-data` orphan branch (commands in docs/AUTOMATION.md).
  4. Commit/push the master-side code changes (loop.py, requirements.txt, workflow, docs).
  5. Manual workflow_dispatch test → confirm green.
- Next: nothing until user does setup; then watch first ticks, tune cal steps if slow.

## Open Questions
- CPU JAX cold-start latency per tick (re-JIT every run) — measure in Bucket 3; if
  >~4 min, trim _N_CAL_STEPS (300) / _LAPLACE_SAMPLES (200).
- Parquet retention/prune cadence on live-data branch (decide in Bucket 4).

## Key Files / Interfaces
- `trading/loop.py` — `run_live(...)` and `_run_tick(...)`; add `chain_log_dir` param +
  CLI flags `--state-path/--kill-log/--kill-flag/--chain-log-dir`.
- `data/cleaner.py` — `clean_chain(chain) -> CleanedSurface`; `OptionsChain` fields:
  strikes, maturities, mid/bid/ask_prices, open_interest, option_type, spot, r, q.
  `CleanedSurface`: chain, repaired_prices, implied_vols. → Parquet source.
- `data/alpaca_loader.py` — `AlpacaLoader().fetch(ticker, date) -> OptionsChain`.
- `trading/broker.py` — `AlpacaBroker(paper=True)`.
- `requirements.txt` — add `pyarrow`; numpyro already present (Colab failure was
  kernel-restart-only, not a missing dep).
- `.github/workflows/tick.yml` — (new) cron `*/5 13-20 * * 1-5`, concurrency guard,
  checkout code + `live-data` branch into `_state/`, pip cache, one tick, commit+push.
- Secrets (GH Actions): ALPACA_API_KEY, ALPACA_SECRET_KEY, FRED_API_KEY (optional).
