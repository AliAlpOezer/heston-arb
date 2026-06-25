# Heston Arb — Operating Contract

This file is read at the start of every conversation touching this codebase.
It pins the mathematical conventions that every module shares.
**Do not drift from these. If a formula looks different from what you know, ask — don't silently use an alternative convention.**

---

## Goal
Identify and trade mispricings in the implied volatility surface using Heston model
calibration via Diffrax (JAX autodiff through SDE). Enter positions where the gap
between market-implied vol and model-implied vol exceeds calibration uncertainty plus
transaction cost. Exit when the gap closes or Popper kill condition triggers.

---

## Mathematical Conventions (non-negotiable)

### Heston SDE (Heston 1993, equation 2–3)
```
dS_t = μ S_t dt + √V_t S_t dW¹_t
dV_t = κ(θ − V_t) dt + ξ √V_t dW²_t
d⟨W¹, W²⟩_t = ρ dt
```
- `S`: spot price
- `V`: instantaneous variance (NOT volatility — V = σ²)
- `κ`: mean reversion speed (kappa)
- `θ`: long-term variance (theta) — in variance units, NOT vol units
- `ξ`: volatility of volatility (xi) — also called volvol or sigma_v
- `ρ`: correlation between spot and variance Brownian motions (rho)
- `v₀`: initial variance at t=0

### Feller Condition (MUST be enforced)
```
2κθ ≥ ξ²
```
Violation produces negative variance paths. All calibrated parameters must satisfy this.
Implementation: soft penalty in loss, hard clip in parameter transform, hard check in evals.

### Parameter Bounds (enforced via softplus/tanh transforms)
```
κ  ∈ [0.01, 10.0]    # mean reversion speed
θ  ∈ [0.0001, 4.0]   # long-term variance (var, not vol)
ξ  ∈ [0.01, 3.0]     # vol of vol
ρ  ∈ [-0.999, 0.999] # correlation
v₀ ∈ [0.0001, 4.0]   # initial variance
```
Transform: use `softplus` for κ, θ, ξ, v₀ (positivity). Use `tanh` scaled for ρ.

### Option Pricing Convention
- European options, no discrete dividends in base model
- Risk-free rate `r` and continuous dividend yield `q` are inputs
- Time to maturity `T` in **years** (ACT/365)
- Moneyness: log-forward moneyness `k = log(K / F)` where `F = S·e^{(r−q)T}`
- Implied vol `σ_iv`: Black-Scholes lognormal implied vol (NOT normal/Bachelier)

### Black-Scholes Reference (for IV extraction)
```
C_BS(S, K, T, r, q, σ) = S·e^{−qT}·N(d₁) − K·e^{−rT}·N(d₂)
d₁ = [log(S/K) + (r − q + σ²/2)T] / (σ√T)
d₂ = d₁ − σ√T
```
- For IV: Newton-Raphson on `C_BS(σ) = C_market`, starting from σ=0.3
- If Newton fails to converge in 50 iterations, mark IV as NaN — do not use a fallback

### Greeks Convention (Carr-Lee sign convention)
- `Delta`: ∂C/∂S — positive for calls, negative for puts
- `Gamma`: ∂²C/∂S² — always positive for vanilla options
- `Vega`: ∂C/∂σ — positive for all vanilla options
- `Theta`: −∂C/∂T — negative for long options (time decay costs you)

### Calibration Loss Function
Weighted RMSE in **log-implied-vol** space:
```
L(params) = √[ Σᵢ wᵢ (log σ_model(kᵢ, Tᵢ) − log σ_market(kᵢ, Tᵢ))² / Σᵢ wᵢ ]

weights wᵢ = OI_i / (bid_ask_spread_i + ε)
```
where OI = open interest, ε = 1e-4 (prevents divide-by-zero).
Rationale: log-vol weighting treats vol levels equally; weights by liquidity.

### Regularization (Tikhonov toward prior day)
```
L_total = L_market + λ · ||params − params_prev_day||²
```
λ = 0.1 default. Prevents parameter jumps. See arXiv 1810.09112 for justification.

### No-Arbitrage Conditions (enforced pre-calibration via LP repair)
Calendar spread (in total variance `w = σ²·T`):
```
w(k, T₂) ≥ w(k, T₁)   for T₂ > T₁, same k
```
Butterfly spread (Durrleman 2005):
```
g(k, T) = (1 − k·∂_k I / (2I))² − (∂_k I)²/4 + I·∂_{kk}I ≥ 0
```
where `I = σ_iv(k,T)` and `k = log(K/F)`.
LP repair: perturb minimum number of prices to restore these conditions.

---

## Project Architecture

```
heston-arb/
├── CLAUDE.md              ← you are here
├── config.py              ← all constants and hyperparameters
├── data/
│   ├── loader.py          ← fetch raw options chain (CBOE / Polygon / synthetic)
│   ├── cleaner.py         ← LP arbitrage repair (arXiv 2008.09454)
│   └── surface.py         ← VolSurface dataclass, IV extraction, interpolation
├── calibration/
│   ├── heston.py          ← Diffrax Heston model, characteristic fn, Carr-Madan pricing
│   ├── calibrator.py      ← JAX/Optax gradient descent calibration loop
│   └── constraints.py     ← Feller check, parameter transforms
├── signals/
│   └── mispricing.py      ← gap detection: model vol vs market vol, Buffett gate
├── risk/
│   └── hedging.py         ← CVXPY delta-neutral hedge optimizer
└── evals/
    ├── synthetic.py       ← known-parameter test surfaces (Popper falsification)
    └── invariants.py      ← mathematical invariant checks (Feller, put-call parity, etc.)
```

---

## Validation Protocol (Popper principle — every module)

Before any module is trusted with real data, it must pass:

1. **Synthetic recovery test** (`evals/synthetic.py`):
   Generate surface with `params_true`. Calibrate. Assert `||params_fitted − params_true|| < tolerance`.
   Tolerance: κ±10%, θ±5%, ξ±10%, ρ±0.05, v₀±5%.

2. **Invariant tests** (`evals/invariants.py`):
   - Feller: `2κθ ≥ ξ²` on all fitted parameter sets
   - Put-call parity: `|C − P − (S·e^{−qT} − K·e^{−rT})| < 1e-4`
   - Calendar monotonicity: `σ_iv(K, T₂) ≥ σ_iv(K, T₁) − 0.01` for T₂ > T₁
   - Non-negative butterfly: no `g(k,T) < −0.001` after LP repair

3. **Kill condition** (Popper — runs daily):
   If calibrated model's predicted IVs on held-out strikes exceed RMSE tolerance for
   5 consecutive trading days → flag strategy, halt position sizing, require manual review.

---

## What NOT to trust Claude to generate without a test

- Any formula involving complex numbers (Heston characteristic function branch cuts)
- Discretization schemes for SDEs (Euler-Maruyama vs Milstein — different variance terms)
- Greeks formulas derived by Claude from scratch (always compare against QuantLib or FinancePy)
- Any formula with a sign convention (theta, rho, charm)
- IV solver convergence (always check the IV actually prices back to the input price)

---

## Data Source Notes (from research)

- **CBOE DataShop**: Uses binomial tree with discrete dividends. Pre-computed IVs
  need re-derivation from raw prices for Heston fitting. Reports IV=0 for deep ITM
  or IV>850% — drop these rows. Good for SPX/SPY historical research.
- **Polygon.io**: Real-time options chains. Good API but data quality requires validation.
  Mid-price as risk-neutral price is a material error for illiquid strikes (9-15% vol error).
- **OptionMetrics**: Academic-grade cleaned data, ivy.csv format. Expensive.
- **Synthetic**: Use for all development and backtesting. Generate via `evals/synthetic.py`.

---

## Dependency Stack
```
jax, jaxlib          # autodiff backend
diffrax              # ODE/SDE solvers
optax                # gradient descent optimizers (Adam, L-BFGS)
cvxpy                # delta-hedge optimizer
numpy, scipy         # numerical utilities
pandas               # data manipulation
financepy            # Black-Scholes IV solver reference
quantlib             # Greeks reference implementation
pytest               # test runner for evals/
```
