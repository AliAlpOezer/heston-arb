"""
Heston calibration via JAX/Optax gradient descent.

Loss: weighted RMSE in log-implied-vol space + Tikhonov regularization toward
previous day's parameters + Feller penalty.

Per CLAUDE.md: weights = OI / (bid_ask_spread + ε).
"""

import jax
import jax.numpy as jnp
import optax
from typing import NamedTuple, Optional
import numpy as np

import config
from calibration.heston import (
    HestonParams,
    params_to_unconstrained,
    unconstrained_to_params,
    heston_implied_vols,
    feller_penalty,
)
from calibration.constraints import project_to_feller


class CalibrationInput(NamedTuple):
    strikes:     jnp.ndarray   # shape (N,)
    maturities:  jnp.ndarray   # shape (N,) — in years
    market_ivs:  jnp.ndarray   # shape (N,) — Black-Scholes implied vols
    weights:     jnp.ndarray   # shape (N,) — per-strike calibration weights
    S:           float
    r:           float
    q:           float


def compute_weights(
    open_interest: np.ndarray,
    bid_ask_spread: np.ndarray,
    eps: float = 1e-4,
) -> jnp.ndarray:
    """Calibration weights: OI / (bid_ask_spread + ε). Normalised to sum to 1."""
    raw = open_interest / (bid_ask_spread + eps)
    return jnp.array(raw / (raw.sum() + 1e-12))


def calibration_loss(
    u: jnp.ndarray,
    cal: CalibrationInput,
    prev_params: Optional[HestonParams],
) -> jnp.ndarray:
    """Total calibration loss for unconstrained parameter vector u.

    L_total = RMSE_log_iv(u) + λ · ||params − prev_params||² + Feller_penalty(u)
    """
    p = unconstrained_to_params(u)

    model_ivs = heston_implied_vols(
        cal.S, cal.strikes, cal.maturities, cal.r, cal.q, p
    )

    # Drop NaN model IVs from loss (failed IV solver)
    valid = ~jnp.isnan(model_ivs) & ~jnp.isnan(cal.market_ivs)
    w = jnp.where(valid, cal.weights, 0.0)
    w = w / (w.sum() + 1e-12)

    log_diff = jnp.log(model_ivs + 1e-8) - jnp.log(cal.market_ivs + 1e-8)
    # Mask NaN entries BEFORE squaring. In JAX 0 * NaN = NaN (not 0),
    # so without this mask, any NaN model IV poisons the whole sum.
    safe_diff = jnp.where(valid, log_diff, 0.0)
    rmse = jnp.sqrt(jnp.sum(w * safe_diff**2))

    # Tikhonov regularization toward previous day's parameters
    tikhonov = 0.0
    if prev_params is not None:
        u_prev = params_to_unconstrained(prev_params)
        tikhonov = config.TIKHONOV_LAMBDA * jnp.sum((u - u_prev)**2)

    # Feller penalty (soft constraint)
    feller = config.FELLER_PENALTY_WEIGHT * feller_penalty(p)

    return rmse + tikhonov + feller


def calibration_rmse(params: HestonParams, cal: CalibrationInput) -> float:
    """Pure weighted log-IV RMSE for fitted params — the fit-quality metric.

    This is the calibration loss WITHOUT the Tikhonov and Feller penalty terms, so
    it is the right number to feed the Popper kill condition (which thresholds on fit
    quality, not on regularization). `calibrate()` returns the full objective; use
    this for the kill log, the tick log, and the signal-reliability gate.
    """
    model_ivs = heston_implied_vols(cal.S, cal.strikes, cal.maturities, cal.r, cal.q, params)
    valid = ~jnp.isnan(model_ivs) & ~jnp.isnan(cal.market_ivs)
    w = jnp.where(valid, cal.weights, 0.0)
    w = w / (w.sum() + 1e-12)
    log_diff = jnp.log(model_ivs + 1e-8) - jnp.log(cal.market_ivs + 1e-8)
    safe_diff = jnp.where(valid, log_diff, 0.0)
    return float(jnp.sqrt(jnp.sum(w * safe_diff**2)))


@jax.jit
def _loss_and_grad(u, cal, prev_params):
    return jax.value_and_grad(calibration_loss)(u, cal, prev_params)


def calibrate(
    cal: CalibrationInput,
    initial_params: Optional[HestonParams] = None,
    prev_params: Optional[HestonParams] = None,
    n_steps: int = 500,
    learning_rate: float = 5e-3,
    verbose: bool = False,
) -> tuple[HestonParams, float]:
    """Run Adam gradient descent to calibrate Heston parameters.

    Returns (fitted_params, final_loss).

    initial_params: starting point for optimisation.
        Defaults to prev_params if available, otherwise a neutral starting point.
    prev_params: yesterday's parameters — used for Tikhonov regularisation.
    """
    if initial_params is None:
        if prev_params is not None:
            initial_params = prev_params
        else:
            # Neutral starting point — moderate vol regime
            initial_params = HestonParams(
                kappa=2.0,
                theta=0.04,   # ~20% long-run vol
                xi=0.3,       # 0.3² = 0.09 < 2*2*0.04 = 0.16 → Feller satisfied
                rho=-0.7,     # typical negative equity-vol correlation
                v0=0.04,      # ~20% spot vol
            )

    u = params_to_unconstrained(initial_params)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(u)

    best_loss = jnp.inf
    best_u = u

    for step in range(n_steps):
        loss, grad = _loss_and_grad(u, cal, prev_params)

        if jnp.isnan(loss) or jnp.isinf(loss):
            if verbose:
                print(f"  step {step}: NaN/Inf loss — stopping early")
            break

        if loss < best_loss:
            best_loss = loss
            best_u = u

        updates, opt_state = optimizer.update(grad, opt_state)
        u = optax.apply_updates(u, updates)

        if verbose and step % 100 == 0:
            p = unconstrained_to_params(u)
            print(
                f"  step {step:4d}: loss={loss:.6f}  "
                f"κ={p.kappa:.3f} θ={p.theta:.4f} ξ={p.xi:.3f} "
                f"ρ={p.rho:.3f} v₀={p.v0:.4f}"
            )

    fitted_params = unconstrained_to_params(best_u)
    # Hard Feller projection: the soft penalty can leave params marginally past the
    # boundary, and the signal layer silently zeroes ALL signals on any Feller violation.
    fitted_params = project_to_feller(fitted_params)
    return fitted_params, float(best_loss)
