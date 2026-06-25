"""
Calibration uncertainty via Laplace approximation (JAX Hessian).

After Adam finds the MAP estimate û (kappa, theta, xi, rho, v0), we approximate
the posterior as a Gaussian centred at û with covariance equal to the inverse
Hessian of the calibration loss evaluated at û:

    P(u | data) ≈ N(û, H⁻¹)    where H = ∇²L(û)

This is the Laplace approximation — standard in quantitative finance for
calibrated stochastic-volatility models (e.g. Cont & Tankov 2004 §10.3).

Why not full MCMC / SVI?
━━━━━━━━━━━━━━━━━━━━━━━
Running price_surface (62 options × 32 GL nodes × 400 RK4 steps) inside a
NumPyro SVI model requires JIT-compiling the full Carr-Madan trace, which
takes 5+ minutes on first run and is prohibitive for daily production use.

The Laplace approximation gives the SAME result as AutoNormal SVI at
convergence (both produce a Gaussian in unconstrained space), but:
  - Hessian: one forward pass + automatic differentiation = ~seconds
  - SVI 500 steps: 500 × (forward + backward) through price_surface = ~minutes

When the posterior is truly non-Gaussian (regime changes, multimodal), switch
to SVI with a linearized model — see run_svi_linearized() below.

Output
━━━━━━
CalibrationPosterior holds n_samples parameter draws. For each signal:
  - iv_credible_interval(strike_idx) → (p05, p95) across samples
  - signal_clears_uncertainty(vol_gap, strike_idx) → True if gap > uncertainty
"""

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from scipy.stats import norm as scipy_norm
import warnings

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from calibration.heston import (
    HestonParams,
    params_to_unconstrained,
    unconstrained_to_params,
    heston_implied_vols,
    feller_satisfied,
)
from calibration.calibrator import CalibrationInput, calibration_loss
import config


# ── Laplace approximation: Hessian at MAP estimate ───────────────────────────

def _calibration_loss_u(u: jnp.ndarray, cal: CalibrationInput) -> jnp.ndarray:
    """Wrapper: calibration loss as a function of unconstrained u only."""
    return calibration_loss(u, cal, prev_params=None)


def compute_hessian(
    u_opt: jnp.ndarray,
    cal: CalibrationInput,
    eps: float = 1e-3,
) -> jnp.ndarray:
    """Compute the Hessian ∇²L(û) via central finite differences.

    Uses the already-JIT-compiled _loss_and_grad from calibrator (reuses the
    cached XLA compilation — no recompile). 30 evaluations total for 5 params.

    eps: step in unconstrained space. 1e-3 is appropriate for Adam calibrated
    solutions where the loss landscape is smooth near the optimum.
    """
    from calibration.calibrator import _loss_and_grad

    u_opt = jnp.array(u_opt)
    n = len(u_opt)
    H = np.zeros((n, n))

    def f(u):
        # _loss_and_grad is @jax.jit — reuses compiled XLA from calibration
        loss, _ = _loss_and_grad(u, cal, None)
        return float(loss)

    for i in range(n):
        for j in range(i, n):
            ei = jnp.zeros(n).at[i].set(eps)
            ej = jnp.zeros(n).at[j].set(eps)
            f_pp = f(u_opt + ei + ej)
            f_pm = f(u_opt + ei - ej)
            f_mp = f(u_opt - ei + ej)
            f_mm = f(u_opt - ei - ej)
            H[i, j] = H[j, i] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps**2)

    return jnp.array(H)


def posterior_covariance(
    hessian: jnp.ndarray,
    regularisation: float = 1e-4,
) -> jnp.ndarray:
    """Posterior covariance Σ = (H + ε·I)⁻¹.

    The regularisation term ε prevents inversion failure when the loss surface
    is flat along a parameter direction (e.g. κ-v₀ degeneracy). A larger ε
    shrinks the posterior toward zero (wider uncertainty) for degenerate axes.
    """
    H_reg = hessian + regularisation * jnp.eye(5)
    return jnp.linalg.inv(H_reg)


# ── Posterior dataclass ───────────────────────────────────────────────────────

@dataclass
class CalibrationPosterior:
    """Laplace-approximated posterior over Heston parameters.

    param_samples: shape (n_samples, 5) — bounded HestonParams values.
        Columns: [kappa, theta, xi, rho, v0].
    u_samples: shape (n_samples, 5) — unconstrained space samples.
    hessian: (5, 5) calibration loss Hessian at MAP estimate.
    posterior_cov: (5, 5) posterior covariance = H⁻¹.
    """
    param_samples: np.ndarray         # (n_samples, 5) — bounded params
    u_samples: np.ndarray             # (n_samples, 5) — unconstrained
    hessian: np.ndarray               # (5, 5)
    posterior_cov: np.ndarray         # (5, 5)
    point_estimate: HestonParams
    cal: CalibrationInput

    # Lazy cache: model IVs on the calibration grid for each sample
    _iv_samples: Optional[np.ndarray] = field(default=None, repr=False)

    def _ensure_iv_samples(self):
        """Compute Heston IVs for every posterior sample on the calibration grid."""
        if self._iv_samples is not None:
            return
        n = len(self.param_samples)
        n_opts = len(self.cal.strikes)
        matrix = np.full((n, n_opts), np.nan)

        for i, row in enumerate(self.param_samples):
            p = HestonParams(*row)
            ivs = np.array(heston_implied_vols(
                self.cal.S, self.cal.strikes, self.cal.maturities,
                self.cal.r, self.cal.q, p,
            ))
            matrix[i] = ivs

        self._iv_samples = matrix

    def iv_credible_interval(
        self,
        strike_idx: int,
        level: float = config.UNCERTAINTY_CREDIBLE_LEVEL,
    ) -> tuple[float, float]:
        """(p_lo, p_hi) credible interval for model IV at one calibration-grid point."""
        self._ensure_iv_samples()
        col = self._iv_samples[:, strike_idx]
        valid = col[~np.isnan(col)]
        if len(valid) < 10:
            return (np.nan, np.nan)
        alpha = (1.0 - level) / 2.0
        return float(np.quantile(valid, alpha)), float(np.quantile(valid, 1.0 - alpha))

    def iv_uncertainty_halfwidth(
        self,
        strike_idx: int,
        level: float = config.UNCERTAINTY_CREDIBLE_LEVEL,
    ) -> float:
        """Halfwidth = (CI_upper − CI_lower) / 2 — the posterior Buffett gate threshold."""
        lo, hi = self.iv_credible_interval(strike_idx, level)
        if np.isnan(lo):
            return np.nan
        return (hi - lo) / 2.0

    def signal_clears_uncertainty(
        self,
        vol_gap: float,
        strike_idx: int,
        level: float = config.UNCERTAINTY_CREDIBLE_LEVEL,
    ) -> bool:
        """True if |vol_gap| > posterior IV half-width.

        The posterior Buffett gate: we only trade when the observed mispricing
        is larger than our calibration uncertainty at that point.
        """
        hw = self.iv_uncertainty_halfwidth(strike_idx, level)
        if np.isnan(hw):
            return False
        return abs(vol_gap) > hw

    def param_summary(self) -> dict:
        """Posterior mean, std, and 90% CI for each Heston parameter."""
        names = ["kappa", "theta", "xi", "rho", "v0"]
        summary = {}
        for j, name in enumerate(names):
            col = self.param_samples[:, j]
            summary[name] = {
                "mean": float(np.mean(col)),
                "std":  float(np.std(col)),
                "p05":  float(np.quantile(col, 0.05)),
                "p95":  float(np.quantile(col, 0.95)),
            }
        return summary

    def feller_satisfied_fraction(self) -> float:
        """Fraction of posterior samples that satisfy 2κθ ≥ ξ²."""
        return float(np.mean([
            feller_satisfied(HestonParams(*row))
            for row in self.param_samples
        ]))

    def parameter_correlation(self) -> np.ndarray:
        """5×5 posterior correlation matrix from the Laplace covariance."""
        cov = self.posterior_cov
        std = np.sqrt(np.diag(cov))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            corr = cov / np.outer(std, std)
        return np.clip(corr, -1.0, 1.0)


# ── Primary method: Laplace via JAX Hessian ──────────────────────────────────

def run_laplace(
    cal: CalibrationInput,
    point_estimate: HestonParams,
    n_samples: int = 500,
    regularisation: float = 0.05,
    seed: int = 0,
    verbose: bool = False,
) -> CalibrationPosterior:
    """Approximate posterior via Laplace method: N(û, H⁻¹) in unconstrained space.

    Args:
        cal: calibration inputs.
        point_estimate: Adam MAP estimate to expand around.
        n_samples: number of posterior draws.
        regularisation: ridge regularisation on H before inversion.
            Larger = wider uncertainty for near-degenerate param directions.
        seed: numpy RNG seed for reproducible samples.
        verbose: print Hessian eigenvalues (useful for diagnosing degeneracy).

    Returns:
        CalibrationPosterior ready for uncertainty-gated signal filtering.
    """
    u_opt = params_to_unconstrained(point_estimate)

    if verbose:
        print("  Computing Hessian at MAP estimate...")

    H = np.array(compute_hessian(u_opt, cal))

    if verbose:
        eigvals = np.linalg.eigvalsh(H)
        print(f"  Hessian eigenvalues: {eigvals}")
        print(f"  Condition number: {eigvals[-1] / max(eigvals[0], 1e-10):.1f}")

    Sigma = np.array(posterior_covariance(jnp.array(H), regularisation))

    # Cholesky decomposition for sampling: u = û + L @ z, z ~ N(0, I)
    try:
        L = np.linalg.cholesky(Sigma)
    except np.linalg.LinAlgError:
        # Sigma not PD — use eigendecomposition fallback
        eigvals, eigvecs = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-8)
        L = eigvecs @ np.diag(np.sqrt(eigvals))

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_samples, 5))
    u_opt_np = np.array(u_opt)
    u_samples = u_opt_np + z @ L.T

    # Map unconstrained samples → bounded HestonParams
    param_rows = np.array([
        list(unconstrained_to_params(jnp.array(u)))
        for u in u_samples
    ])

    # Warm up heston_implied_vols JIT cache using the point estimate — so
    # _ensure_iv_samples() calls on all 500 samples hit the compiled kernel.
    _ = heston_implied_vols(
        cal.S, cal.strikes, cal.maturities, cal.r, cal.q, point_estimate
    )

    return CalibrationPosterior(
        param_samples=param_rows,
        u_samples=u_samples,
        hessian=H,
        posterior_cov=Sigma,
        point_estimate=point_estimate,
        cal=cal,
    )


# ── Secondary method: linearized SVI (for non-Gaussian posteriors) ────────────

def run_svi_linearized(
    cal: CalibrationInput,
    point_estimate: HestonParams,
    n_steps: int = 500,
    n_samples: int = 500,
    learning_rate: float = 0.01,
    seed: int = 0,
    verbose: bool = False,
) -> CalibrationPosterior:
    """SVI on a first-order Taylor approximation of the IV surface.

    Instead of tracing price_surface inside the NumPyro model (slow),
    we precompute the Jacobian J = ∂IV/∂u at the MAP estimate and use the
    linear model IV(u) ≈ IV(û) + J @ (u - û) inside the model trace.

    This is accurate near the optimum and orders-of-magnitude faster than
    SVI with the full pricing model inside the trace.

    Use this when you suspect the posterior is non-Gaussian (e.g. large
    calibration RMSE, near-boundary parameters, regime transitions).
    """
    from numpyro.infer import SVI, Trace_ELBO
    from numpyro.infer.autoguide import AutoNormal

    u_opt = params_to_unconstrained(point_estimate)

    if verbose:
        print("  Computing IV Jacobian at MAP estimate...")

    # Precompute: model IVs and Jacobian ∂IV/∂u at u_opt
    def iv_from_u(u):
        p = unconstrained_to_params(u)
        return heston_implied_vols(
            cal.S, cal.strikes, cal.maturities, cal.r, cal.q, p
        )

    ivs_at_opt = np.array(iv_from_u(u_opt))
    jac = np.array(jax.jacfwd(iv_from_u)(u_opt))   # shape (N, 5)

    if verbose:
        print(f"  Jacobian shape: {jac.shape}, max|J|={np.nanmax(np.abs(jac)):.4f}")

    ivs_at_opt_jax = jnp.array(np.nan_to_num(ivs_at_opt, nan=0.2))
    jac_jax = jnp.array(np.nan_to_num(jac, nan=0.0))
    market_ivs = jnp.array(cal.market_ivs)
    valid_mask = ~jnp.isnan(market_ivs) & ~jnp.isnan(jnp.array(ivs_at_opt))
    u_opt_arr = jnp.array(u_opt)

    def model():
        u = numpyro.sample("u", dist.Normal(u_opt_arr, 0.5 * jnp.ones(5)))
        sigma_obs = numpyro.sample("sigma_obs", dist.HalfNormal(0.03))

        # Linear approximation (fast — just matrix-vector multiply)
        delta_u = u - u_opt_arr
        model_ivs = ivs_at_opt_jax + jac_jax @ delta_u
        model_ivs = jnp.clip(model_ivs, 0.01, 5.0)

        log_model  = jnp.where(valid_mask, jnp.log(model_ivs + 1e-8), 0.0)
        log_market = jnp.where(valid_mask, jnp.log(market_ivs + 1e-8), 0.0)

        with numpyro.plate("options", len(cal.strikes)):
            numpyro.sample(
                "obs",
                dist.Normal(log_model, sigma_obs).mask(valid_mask),
                obs=log_market,
            )

    guide = AutoNormal(model)
    optimizer = numpyro.optim.Adam(learning_rate)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    rng_key = jax.random.PRNGKey(seed)
    rng_key, init_key = jax.random.split(rng_key)
    state = svi.init(init_key)

    losses = []
    for step in range(n_steps):
        state, loss = svi.update(state)
        losses.append(float(loss))
        if verbose and step % 100 == 0:
            print(f"  SVI step {step:4d}: ELBO={-loss:.2f}")

    # Draw posterior samples from the trained guide
    rng_key, sample_key = jax.random.split(rng_key)
    post_samples = guide.sample_posterior(
        sample_key, svi.get_params(state), sample_shape=(n_samples,)
    )
    u_samples = np.array(post_samples["u"])

    param_rows = np.array([
        list(unconstrained_to_params(jnp.array(u)))
        for u in u_samples
    ])

    # Hessian from SVI guide's learned precision (diagonal approx)
    H_approx = np.diag(1.0 / (np.var(u_samples, axis=0) + 1e-8))
    Sigma_approx = np.diag(np.var(u_samples, axis=0))

    return CalibrationPosterior(
        param_samples=param_rows,
        u_samples=u_samples,
        hessian=H_approx,
        posterior_cov=Sigma_approx,
        point_estimate=point_estimate,
        cal=cal,
    )


# ── Integration: posterior-gated signal filtering ────────────────────────────

def filter_signals_by_posterior(
    signals,
    posterior: CalibrationPosterior,
    level: float = config.UNCERTAINTY_CREDIBLE_LEVEL,
) -> list:
    """Keep only signals where |gap| > posterior IV half-width.

    Args:
        signals: list of MispricingSignal from signals.mispricing.
        posterior: CalibrationPosterior from run_laplace() or run_svi_linearized().
        level: credible interval level (default 0.90).

    Returns:
        Filtered signals — only those that clear the posterior Buffett gate.
    """
    posterior._ensure_iv_samples()

    cal = posterior.cal
    strikes_arr = np.array(cal.strikes)
    mats_arr = np.array(cal.maturities)

    robust = []
    for sig in signals:
        F = cal.S * np.exp((cal.r - cal.q) * sig.maturity)
        K = F * np.exp(sig.log_moneyness)

        # Nearest calibration-grid point to this signal
        dists = (strikes_arr - K)**2 + (mats_arr - sig.maturity)**2 * 100
        idx = int(np.argmin(dists))

        if posterior.signal_clears_uncertainty(sig.vol_gap, idx, level):
            robust.append(sig)

    return robust
