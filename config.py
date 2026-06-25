"""
All constants and hyperparameters. One place to change numbers — never hardcode elsewhere.
"""

# ── Heston parameter bounds (in variance units, not vol units) ──────────────
PARAM_BOUNDS = {
    "kappa": (0.01, 10.0),    # mean reversion speed
    "theta": (1e-4, 4.0),     # long-term variance (NOT volatility)
    "xi":    (0.01, 3.0),     # vol of vol
    "rho":   (-0.999, 0.999), # spot-variance correlation
    "v0":    (1e-4, 4.0),     # initial variance (NOT volatility)
}

# ── Feller condition: 2κθ ≥ ξ² ──────────────────────────────────────────────
# Soft penalty weight in calibration loss (added when Feller is violated)
FELLER_PENALTY_WEIGHT = 10.0

# ── Calibration loss ─────────────────────────────────────────────────────────
# Regularization toward previous day's parameters (Tikhonov)
# arXiv 1810.09112: frequent recalibration shifts risk, not removes it
TIKHONOV_LAMBDA = 0.1

# Minimum open interest to include a strike in calibration surface
MIN_OPEN_INTEREST = 10

# Maximum bid-ask spread as fraction of mid-price to include a strike
MAX_BID_ASK_FRAC = 0.50

# ── No-arbitrage LP repair ───────────────────────────────────────────────────
# Maximum allowed perturbation to a single option price (as fraction of mid)
LP_MAX_PERTURBATION_FRAC = 0.05

# Tolerance for Durrleman butterfly condition (allow small numerical violations)
BUTTERFLY_TOLERANCE = 1e-3

# ── Mispricing detection (Buffett margin of safety) ──────────────────────────
# Minimum vol gap (in absolute IV points) to consider a mispricing actionable
# After transaction cost and calibration uncertainty
MIN_VOL_GAP = 0.015          # 1.5 vol points

# NumPyro posterior credible interval level for calibration uncertainty
UNCERTAINTY_CREDIBLE_LEVEL = 0.90

# ── Signal quality gates ──────────────────────────────────────────────────────
# Suppress ALL signals when the calibration fit is worse than this — a poor fit means
# model IVs are unreliable everywhere, so every "gap" is a fit artifact, not a mispricing.
# Mirrors CALIBRATION_FAIL_RMSE (the Popper threshold, defined below); kept as a literal
# so it does not depend on definition order within this module.
SIGNAL_MAX_RMSE = 0.10
# Reject implausibly large vol gaps (IV-extraction / model artifacts, not real edges).
MAX_VOL_GAP = 0.10                            # 10 vol points
# Only signal within this |log-moneyness| band — deep ITM/OTM IV is unstable and vega~0.
MAX_SIGNAL_LOG_MONEYNESS = 0.20
# Restrict the calibration grid to this |log-moneyness| band so a single-factor Heston
# can actually fit it; the full ±0.5 surface drives RMSE up and floods false signals.
MAX_CAL_LOG_MONEYNESS = 0.30

# ── Delta hedging (Artur Sepp optimal frequency) ─────────────────────────────
# Target hedging interval in hours. Sepp (2017): Sharpe is humped vs frequency.
# Transaction costs scale with sqrt(frequency). Start conservative.
HEDGE_INTERVAL_HOURS = 4

# Underlying bid-ask spread assumed for equity delta hedging (bps of spot)
EQUITY_BID_ASK_BPS = 2.0

# Option contract multiplier (shares per contract). US equity/ETF options = 100.
# Used to convert per-share Greeks (delta, vega) into per-contract exposure so the
# stock delta-hedge (denominated in shares) actually neutralizes the option book.
CONTRACT_MULTIPLIER = 100

# ── Position management ────────────────────────────────────────────────────────
# Maximum holding period in CALENDAR days before a position is force-closed.
# Sized at ~2× MEAN_REVERSION_HALFLIFE_DAYS so the gap has time to converge.
# NOTE: this is wall-clock days, NOT tick count — must not be driven by tick cadence.
MAX_HOLD_DAYS = 10

# ── Popper kill conditions ────────────────────────────────────────────────────
# Consecutive days of calibration failure before strategy is halted
POPPER_KILL_DAYS = 5

# RMSE threshold (in log-IV space) above which a day counts as a calibration failure
CALIBRATION_FAIL_RMSE = 0.10

# Intraday circuit-breaker (complements the daily POPPER_KILL_DAYS halt): number of
# CONSECUTIVE failing ticks before new entries are paused for the rest of the session.
# At 5-min ticks, 12 ticks ≈ 1 hour of sustained miscalibration.
POPPER_KILL_TICKS = 12

# ── IV solver ────────────────────────────────────────────────────────────────
IV_SOLVER_MAX_ITER = 50
IV_SOLVER_TOL = 1e-8
IV_INITIAL_GUESS = 0.30      # 30% — reasonable starting point for Newton-Raphson

# ── Fourier pricing (Carr-Madan) ─────────────────────────────────────────────
# Number of integration points for Gauss-Laguerre quadrature
N_QUADRATURE_POINTS = 32

# Fixed steps for RK4 Heston ODE solver. lax.scan (used here) composes cleanly
# inside jit+vmap unlike Diffrax's PIDController (lax.while_loop). 400 steps
# gives dt=0.005 for T=2y — accurate for smooth Riccati solutions.
N_ODE_STEPS = 400

# Dampening factor α for Carr-Madan (must satisfy α > 0, α(α+1) < E[S^{α+1}])
CARR_MADAN_ALPHA = 1.5

# ── Data quality thresholds ───────────────────────────────────────────────────
# Drop options with time-to-maturity below this threshold (near-expiry instability)
MIN_TTM_DAYS = 5

# Drop options with time-to-maturity above this threshold
MAX_TTM_DAYS = 730           # 2 years

# Strike range to include: log-moneyness |log(K/F)| ≤ this
MAX_LOG_MONEYNESS = 0.50     # roughly 0.5 to 1.6 in K/F terms

# CBOE DataShop: drop rows where reported IV == 0 (below intrinsic or IV > 850%)
CBOE_DROP_ZERO_IV = True

# ── Execution ──────────────────────────────────────────────────────────────────
# Marketable-limit slippage band for exits (bps of option premium). Caps how far
# through the NBBO an exit limit reaches, instead of an uncapped market order.
EXIT_SLIPPAGE_BPS = 100      # 1.0% of premium
