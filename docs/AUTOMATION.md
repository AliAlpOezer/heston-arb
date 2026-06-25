# Automated Live Loop (GitHub Actions)

The heston-arb live loop runs **hands-off, one tick per scheduled run**, on free
GitHub Actions. Each run executes a single full pipeline pass against the Alpaca
**paper** account, then commits its state + a per-tick options dataset back to a
dedicated `live-data` branch. There is no server to keep alive and no daily click.

Design rationale and the decisions behind it: `docs/context/automation.md`.

---

## How it works

```
cron */5 13-21 UTC, Mon-Fri
        │
        ▼
.github/workflows/tick.yml
  1. checkout code (master)            ─┐
  2. checkout state (live-data → _state)│ both into one runner
  3. pip install -r requirements.txt   │ (cached)
  4. python -m trading.loop --ticks 1  │ ONE tick, CPU JAX, paper orders
  5. commit + push _state → live-data ─┘
```

- **Market-hours guard is authoritative.** `trading.loop._market_is_open()` uses the
  Alpaca clock, so off-hours and **holiday** fires are fast no-ops — they advance the
  tick counter and exit without calibrating or trading. That's why the cron window is
  wide (`13-21`): it covers **both** DST regimes (summer RTH 13:30–20:00 UTC, winter
  14:30–21:00 UTC) and the guard ignores the slots outside the real session.
- **`--ticks 1`** means the loop runs exactly one pass and exits before its sleep — the
  cron *is* the scheduler. State on `live-data` makes each run resume from the last.
- **`concurrency: live-tick` (no cancel)** serializes runs so two ticks can never act on
  stale state or double up orders.
- **CPU only.** GHA runners have no GPU; `JAX_PLATFORMS=cpu` is set. The calibration grid
  is sparse (≤96 points), so a tick fits well under the 5-min cadence even cold.

---

## One-time setup

1. **Make the repo public** (Settings → General → Danger Zone → Change visibility).
   Public repos get *unlimited* free Actions minutes; private repos cap at 2000/mo,
   which a 5-min cadence blows past.
   - ⚠️ Before flipping: the entire git **history** becomes world-readable. Confirm no
     secret was ever committed (`git log --all -p -- .env` returns nothing) and **rotate
     the Alpaca keys** if anything ever leaked. Strategy code + tick logs go public too.

2. **Add repository secrets** (Settings → Secrets and variables → Actions → New secret):
   - `ALPACA_API_KEY`
   - `ALPACA_SECRET_KEY`
   - `FRED_API_KEY` *(optional — improves the risk-free rate; loop runs without it)*

3. **Seed the `live-data` branch** (orphan; uses a temp worktree so your current branch
   and working changes are untouched):
   ```bash
   git worktree add /tmp/heston-seed --detach
   cd /tmp/heston-seed
   git checkout --orphan live-data
   git rm -rf . 2>/dev/null || true
   printf '# heston-arb live-data\n\nRuntime state + per-tick chain Parquet written by\n.github/workflows/tick.yml. Do not edit by hand.\n' > README.md
   git add README.md
   git commit -m "seed live-data branch"
   git push -u origin live-data
   cd -
   git worktree remove /tmp/heston-seed
   ```
   The loop creates `live_state.json`, `kill_log.json`, `tick_log.jsonl`, and
   `chains/` automatically on the first tick — only the placeholder is needed here.

4. **Enable the workflow.** It's active once on the default branch. Trigger a manual
   test before relying on the cron: Actions → **live-tick** → *Run workflow*
   (`workflow_dispatch`). Outside RTH it logs a no-op tick — that confirms wiring.

---

## What lands on `live-data`

```
live-data branch
├── README.md
├── live_state.json          # open positions, prev Heston params, counters
├── kill_log.json            # rolling per-day calibration RMSE (Popper kill record)
├── kill_flag.json           # present + halted:true  ⇒  trading halted
├── tick_log.jsonl           # one JSON line per tick (machine log of the loop)
└── chains/
    └── SPY/
        └── 2026-06-25.parquet   # one file per symbol per day (training dataset)
```

`tick_log.jsonl` is the **loop's documented record** — every tick appends params, RMSE,
Feller flag, signal counts at each gate, gap/uncertainty diagnostics, entries/exits, and
session P&L. Analyse it with Cell 7 of `colab.ipynb` or `pd.read_json(path, lines=True)`.

---

## Training dataset (`chains/<ticker>/<date>.parquet`)

Full **post-clean** options chain captured every tick (before any skip), so the dataset
records each tradeable snapshot even on ticks that don't calibrate.

| column           | meaning                                   |
|------------------|-------------------------------------------|
| `time`           | tick timestamp (ISO 8601 UTC)             |
| `date`           | trading date (YYYY-MM-DD)                 |
| `ticker`         | underlying                                |
| `strike`         | K                                         |
| `maturity`       | T in years (ACT/365)                      |
| `expiry`         | approx expiry date (YYYY-MM-DD)           |
| `opt_type`       | 'C' / 'P'                                 |
| `bid` `ask` `mid`| NBBO quotes                               |
| `repaired_price` | LP arbitrage-repaired mid                 |
| `oi`             | open interest                             |
| `iv`             | Black-Scholes IV from the repaired price  |
| `spot` `r` `q`   | snapshot spot, risk-free rate, div yield  |

Load one day, or the whole history:
```python
import pandas as pd, glob
df = pd.read_parquet("chains/SPY/2026-06-25.parquet")
alldf = pd.concat(map(pd.read_parquet, glob.glob("chains/SPY/*.parquet")), ignore_index=True)
```

---

## Operations

- **Manual single tick:** Actions → live-tick → Run workflow.
- **Pause/resume (kill switch):**
  - The loop halts automatically after `POPPER_KILL_TICKS` consecutive failing
    calibrations (writes `kill_flag.json` with `halted:true`).
  - **Resume:** delete `kill_flag.json` on the `live-data` branch (a missing flag reads
    as not-halted), or commit it with `{"halted": false}`.
  - **Force-halt:** commit `kill_flag.json` with `{"halted": true}` to `live-data`.
- **Stop entirely:** disable the **live-tick** workflow (Actions → ⋯ → Disable), or
  comment out the `schedule:` block. Note GitHub auto-disables schedules after 60 days
  of **no repo activity** — the daily commits to `live-data` keep it alive.
- **Change cadence:** edit the `cron` in `tick.yml` (e.g. `*/15` for 15-min ticks).

---

## Maintenance / pruning

`live-data` gains ~80 commits/trading-day. The files stay small (one daily Parquet read-
appended in place; `tick_log.jsonl` ~80 lines/day), but history grows. When it gets
unwieldy, squash to a single commit (history only — current data is preserved):
```bash
git worktree add /tmp/heston-squash live-data
cd /tmp/heston-squash
git checkout --orphan squashed && git add -A && git commit -m "squash live-data $(date -u +%F)"
git branch -M live-data && git push -f origin live-data
cd - && git worktree remove /tmp/heston-squash --force
```
To archive raw chains off-branch, download the Parquet and `git rm` old months.

---

## Known limits

- **Cron jitter:** GitHub may delay scheduled runs minutes under load; occasional ticks
  are skipped/merged. The loop is stateless per tick and resumes, so this is tolerable.
- **CPU cold-start:** every run re-JITs JAX. If a tick approaches the cadence, trim
  `_N_CAL_STEPS` (300) / `_LAPLACE_SAMPLES` (200) in `trading/loop.py`.
- **Paper only:** `--alpaca --live` routes to `AlpacaBroker(paper=True)`. Real capital is
  out of scope here.
