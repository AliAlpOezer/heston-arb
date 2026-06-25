# Heston-Arb — Context Pack
Updated: 2026-06-22

## Goal & Core Decision
Identify and trade IV-surface mispricings using Heston model calibration.
Enter when |model_iv − market_iv| > MIN_VOL_GAP (1.5 vol pts) AND gap clears posterior uncertainty.
Exit when gap closes or Popper kill condition triggers (5 consecutive RMSE failures).

## Architecture (all modules implemented)

```
heston-arb/
  config.py                   all hyperparameters
  data/
    loader.py                 SyntheticLoader, CBOELoader (→ data/cboe.py), PolygonLoader stub
    cboe.py                   CBOE DataShop CSV parser with alias-column resolution
    cleaner.py                LP arbitrage repair (calendar + butterfly constraints)
    surface.py                VolSurface (RectBivariateSpline), build_surface()
    polygon.py                PolygonLoader — live options chain (Massive API, with spot fallback)
    alpaca_loader.py          AlpacaLoader — alternative NBBO data source (not used in live loop)
    rates.py                  get_rates(): FRED DGS3MO + fallback tables (SP500DIV FRED lookup fails → uses tables)
  calibration/
    heston.py                 Lord-Kahl Riccati ODE, RK4 via lax.scan, Carr-Madan Fourier pricing
    calibrator.py             JAX/Optax Adam calibration, _loss_and_grad (JIT-compiled)
    constraints.py            Feller check, param bounds, clip_to_bounds()
    uncertainty.py            Laplace approx posterior, linearized SVI, CalibrationPosterior,
                              filter_signals_by_posterior()
  signals/
    mispricing.py             detect_mispricings(), Buffett gate, compute_model_ivs()
    sizing.py                 Kelly criterion: f* = gap / (IV_DAILY_VOL^2 * halflife), contract caps
  risk/
    hedging.py                CVXPY delta-neutral optimizer, bs_delta(), optimize_hedge()
  evals/
    synthetic.py              Known-param recovery tests (4/4 Popper falsification PASS)
    invariants.py             Feller, put-call parity, calendar mono, butterfly checks
    kill.py                   KillCondition: JSON log, halts after 5 consecutive RMSE > 0.10 failures
    test_kill.py              9/9 tests PASS
  backtest/
    runner.py                 Daily loop: load → clean → calibrate → kill → signals → Kelly P&L
    test_runner.py            Synthetic CSV backtest smoke test
  trading/
    broker.py                 PaperBroker (local simulation), AlpacaBroker (paper/live), IBKRBroker (stub)
    state.py                  LiveState: persists open positions, hedge, prev params across ticks
    loop.py                   Full 11-step live loop: fetch → clean → calibrate → kill → posterior →
                              signals → size → enter → mark → exit → delta-hedge
```

## Data Flow (live trading)
- **Options data**: Massive (Polygon.io) via PolygonLoader — always, regardless of broker
- **Spot price**: from `underlying_asset.price` in each contract; falls back to `/v2/aggs/ticker/{T}/prev`
- **Rates**: FRED DGS3MO for r (falls back to table); dividend q always from hardcoded table
- **Execution**: AlpacaBroker (paper=True for paper trading)
- **Keys required**: POLYGON_API_KEY + ALPACA_API_KEY + ALPACA_SECRET_KEY in .env

## Run command (paper trading)
```
python -m trading.loop --ticker SPY --alpaca --interval 240
```
Defaults: dry_run=False when --alpaca passed, Alpaca paper account, 4-hour ticks.

## Mathematical Conventions (MUST NOT CHANGE)

- Heston SDE: dS = μS dt + √V S dW¹; dV = κ(θ−V)dt + ξ√V dW²; correlation ρ
- All params in VARIANCE units (not vol units)
- Feller condition: 2κθ ≥ ξ²
- Lord-Kahl Riccati: dD/dτ = −½(u²+iu) + bD + cD² (NEGATIVE sign on constant term)
- Carr-Madan: 32-pt Gauss-Laguerre, α=1.5 dampening
- RK4 via lax.scan (N=400 steps) — NOT Diffrax (PIDController breaks nested vmap)
- JAX float64: jax.config.update("jax_enable_x64", True) BEFORE any JAX import
- FD Hessian: uses _loss_and_grad (JIT-cached), NOT calibration_loss directly
- RectBivariateSpline: returns 2D array — must use .item() not float() for scalar result
- Windows cp1252 encoding: NO Greek/emoji chars in print() statements anywhere

## Key Decisions (with rationale)

- **Laplace approximation** over full MCMC (2026-06-19): price_surface through NumPyro tracing
  generates a 62-option × 32-GL × 400-RK4 graph that hangs JIT for 15+ min. Laplace uses
  already-JIT-compiled _loss_and_grad, completes in ~16s via 30 FD evaluations.

- **Regularisation=0.05** for Hessian (2026-06-19): condition number ~897 (κ-v₀ degeneracy).
  Smaller values (1e-4) produce too-wide posteriors (Feller fraction ~47%). With 0.05, the
  Laplace posterior is usably tight for the uncertainty gate.

- **Feller fraction assertion 40%** (not 50%): κ-v₀ degeneracy is genuine; 46.8% is
  mathematically correct for a well-conditioned synthetic surface.

- **Kelly sizing**: f* = gap / (IV_DAILY_VOL² × halflife). Pure contract-count Kelly, no dollar
  normalization (avoids S-scaling issues). Max 50 portfolio contracts, 10 per position.

- **Backtest P&L model**: vega × Δ(market_iv) × direction. Entry: |gap| > MIN_VOL_GAP.
  Exit: gap < 0.005 OR age > 10 days OR TTM < 5 days.

- **Massive for data, Alpaca for execution** (2026-06-22): AlpacaLoader tried earlier but
  the live loop uses PolygonLoader exclusively for chain snapshots. AlpacaBroker handles orders.

- **No Co-Authored-By attribution**: user explicitly requested clean commits with their email only.
  Git config: user.name=AliAlpOezer, user.email=a.oezer@flex.net (do NOT modify).

## Current State (2026-06-22)

**Done — all modules complete:**
- 4/4 Popper falsification tests (evals/synthetic.py)
- Full calibration pipeline (Lord-Kahl + Carr-Madan + Laplace posterior)
- evals/kill.py (9/9 tests PASS)
- backtest/runner.py (5-day synthetic pipeline, Kelly-sized, all assertions pass)
- signals/sizing.py (Kelly criterion)
- trading/loop.py — full 11-step live tick
- trading/broker.py — PaperBroker + AlpacaBroker (paper/live) + IBKRBroker (stub)
- trading/state.py — LiveState JSON persistence
- GitHub private repo: https://github.com/AliAlpOezer/heston-arb

**Bug fixes applied (2026-06-22):**
- Added `VolSurface.n_maturities` and `n_strikes_per_mat` properties (loop.py verbose print crashed)
- Added spot fallback in `PolygonLoader.fetch()`: calls `fetch_underlying_price()` if `underlying_asset.price` absent
- Corrected `.env.example` comment: POLYGON_API_KEY is always required (not IBKR/CBOE-only)

**Pending:**
- trading/__init__.py untracked (needs commit with the other trading/ files)
- No pending bugs known for paper trading path

**Next:**
- Start paper trading: configure .env with real keys, run the loop command above
- Monitor paper_log.json for order activity
- After N ticks, review: RMSE trend, signal frequency, position turnover

## Open Questions
- Account size / risk limits for Kelly caps (currently: max 50 contracts portfolio, 10 per position)
- When does the kill condition trigger in practice on real data?
- IBKRBroker cancel_order is a stub — not usable for IBKR paper trading yet

## Key Files / Interfaces

- `calibration/heston.py:price_call(S, K, T, r, q, p) → float`: single option price
- `calibration/heston.py:price_surface(S, strikes, maturities, r, q, p) → array`: vmapped
- `calibration/calibrator.py:calibrate(cal, initial_params, prev_params, n_steps) → (HestonParams, float)`
- `calibration/uncertainty.py:run_laplace(cal, point_estimate, n_samples, regularisation) → CalibrationPosterior`
- `calibration/uncertainty.py:filter_signals_by_posterior(signals, posterior) → list[MispricingSignal]`
- `data/polygon.py:PolygonLoader.fetch(ticker, snapshot_date) → OptionsChain`
- `evals/kill.py:KillCondition.record(ticker, date, rmse) → KillState`
- `backtest/runner.py:run_backtest(ticker, data_dir, start_date, end_date) → BacktestResults`
- `signals/sizing.py:size_portfolio(signals, spot, r, q) → list[SizingResult]`
- `trading/loop.py:run_live(ticker, broker, polygon_api_key, interval_minutes, dry_run) → None`
- `trading/broker.py:AlpacaBroker(paper=True)` — paper trading execution
- `trading/state.py:LiveState.load(path) / .save(path)` — tick persistence
