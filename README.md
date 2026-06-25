# Heston Arb

Implied volatility surface arbitrage via Heston model calibration with JAX autodiff (Diffrax).

**Philosophy:** Popper (falsifiable kill conditions) · Munger (model complexity is a risk, not an asset) · Buffett (margin of safety on the calibration uncertainty)

## How it works

1. **Clean raw options data** — LP arbitrage repair removes calendar/butterfly violations before calibration sees them
2. **Calibrate Heston parameters** — custom fixed-step RK4 via `lax.scan` solves the Lord-Kahl Riccati ODEs; JAX autodiff gives exact gradients for Adam optimisation
3. **Detect mispricings** — gaps between market-implied vol and model-implied vol, filtered by calibration uncertainty and Buffett's margin of safety threshold
4. **Hedge delta-neutral** — CVXPY constructs the minimum-cost delta hedge given transaction cost and optimal hedging frequency (Sepp 2017)
5. **Popper kill** — if calibration RMSE on held-out strikes exceeds threshold for 5 consecutive days, strategy halts

## Run the Popper falsification test first

## Key mathematical reference

- Heston (1993) — original stochastic vol model
- Lord & Kahl (2010) — branch-cut fix for characteristic function
- Carr & Madan (1999) — FFT pricing via characteristic function
- Roper (2010) — no-arbitrage conditions on implied vol surface
- arXiv 2008.09454 — LP arbitrage repair
- Sepp (2017) — optimal delta hedging frequency under transaction costs
- arXiv 1810.09112 — model risk: why more complex ≠ safer

```bash
pip install -r requirements.txt
python -m evals.synthetic
```

This must pass before touching real data. If it fails, the calibrator is broken.

## Project layout

```
CLAUDE.md           Mathematical conventions — read this before writing any code
config.py           All hyperparameters in one place
data/
  cleaner.py        LP arbitrage repair (arXiv 2008.09454)
  loader.py         Raw data fetching (CBOE / Polygon / synthetic)
  surface.py        VolSurface dataclass and interpolation
calibration/
  heston.py         Diffrax Heston model + Carr-Madan pricing
  calibrator.py     Adam gradient descent calibration loop
  constraints.py    Feller check and parameter transforms
signals/
  mispricing.py     Vol gap detection and Buffett gate
risk/
  hedging.py        CVXPY delta-neutral hedge optimiser
evals/
  synthetic.py      Known-parameter recovery tests (Popper)
  invariants.py     Mathematical invariant checks
```
