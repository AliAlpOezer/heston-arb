"""
Heston model: characteristic function via RK4 ODE solver + Carr-Madan pricing.

Mathematical reference: Heston (1993), "A Closed-Form Solution for Options with
Stochastic Volatility with Applications to Bond and Currency Options."

Convention: ALL variances (kappa, theta, xi, v0) are in VARIANCE units (σ²),
not volatility units (σ). See CLAUDE.md §Mathematical Conventions.

Characteristic function implementation follows the formulation of
Lord & Kahl (2010) "Complex logarithms in Heston-like models" to avoid
the branch-cut discontinuity in the original Heston (1993) formulation.

ODE solver: custom fixed-step RK4 via jax.lax.scan. Diffrax's PIDController
uses lax.while_loop which triggers Equinox unvmap_max failures when nested
inside the two jax.vmap calls (price_surface → price_call → integrand).
lax.scan composes correctly inside jit+vmap+vmap.

Complex arithmetic: all Riccati ODE computation is expanded into real and
imaginary parts. No complex JAX arrays — avoids Diffrax complex dtype warnings
and any downstream type-promotion surprises.
"""

import jax
# Must be set before any JAX computation. float32 max ≈ exp(88); the 32-point
# Gauss-Laguerre nodes reach ~92, so phi·exp(v) overflows float32 → NaN IVs → loss=0.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from typing import NamedTuple

import config


class HestonParams(NamedTuple):
    kappa: float   # mean reversion speed
    theta: float   # long-term variance (variance units)
    xi:    float   # vol of vol
    rho:   float   # spot-variance correlation
    v0:    float   # initial variance (variance units)


# ── Parameter transforms: unconstrained ℝ → bounded parameter space ──────────

def params_to_unconstrained(p: HestonParams) -> jnp.ndarray:
    """Map bounded Heston params → unconstrained vector for gradient optimiser."""
    lo, hi = config.PARAM_BOUNDS["kappa"]
    kappa_u = jnp.log((p.kappa - lo) / (hi - p.kappa + 1e-10))

    lo, hi = config.PARAM_BOUNDS["theta"]
    theta_u = jnp.log((p.theta - lo) / (hi - p.theta + 1e-10))

    lo, hi = config.PARAM_BOUNDS["xi"]
    xi_u = jnp.log((p.xi - lo) / (hi - p.xi + 1e-10))

    # rho ∈ (-1, 1) via atanh
    rho_u = jnp.arctanh(p.rho)

    lo, hi = config.PARAM_BOUNDS["v0"]
    v0_u = jnp.log((p.v0 - lo) / (hi - p.v0 + 1e-10))

    return jnp.array([kappa_u, theta_u, xi_u, rho_u, v0_u])


def unconstrained_to_params(u: jnp.ndarray) -> HestonParams:
    """Map unconstrained vector → bounded HestonParams via sigmoid-like inverse."""
    def sigmoid_bounded(x, lo, hi):
        return lo + (hi - lo) * jax.nn.sigmoid(x)

    kappa = sigmoid_bounded(u[0], *config.PARAM_BOUNDS["kappa"])
    theta = sigmoid_bounded(u[1], *config.PARAM_BOUNDS["theta"])
    xi    = sigmoid_bounded(u[2], *config.PARAM_BOUNDS["xi"])
    rho   = jnp.tanh(u[3])
    v0    = sigmoid_bounded(u[4], *config.PARAM_BOUNDS["v0"])
    return HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)


# ── Feller condition ──────────────────────────────────────────────────────────

def feller_penalty(p: HestonParams) -> jnp.ndarray:
    """Soft penalty: 0 if Feller satisfied, positive if violated.

    Feller condition: 2κθ ≥ ξ²
    Penalty: max(0, ξ² − 2κθ)² — squared hinge loss.
    """
    violation = p.xi**2 - 2.0 * p.kappa * p.theta
    return jnp.maximum(0.0, violation) ** 2


def feller_satisfied(p: HestonParams) -> bool:
    """Hard check — use in evals, not in differentiable loss."""
    return bool(2.0 * p.kappa * p.theta >= p.xi**2)


# ── Heston Riccati ODE (fully real arithmetic) ───────────────────────────────
#
# Heston char fn: φ(u; T) = exp(C(u,T)·θ + D(u,T)·v₀ + i·u·log(F))
#
# C and D satisfy Riccati ODEs (Lord-Kahl 2010):
#   dD/dt = a + b·D + c·D²,   dC/dt = κ·D
# where (with u = u_r + i·u_i):
#   a = ½(u² + i·u)
#   b = ρ·ξ·i·u − κ
#   c = ½ξ²
#
# Expanded into real/imaginary components to avoid complex JAX dtypes:
#   a_R = ½(u_r² − u_i² − u_i)
#   a_I = ½·u_r·(2u_i + 1)
#   b_R = −κ − ρ·ξ·u_i,   b_I = ρ·ξ·u_r
#   c   = ½ξ²  (real)
#
# State vector: y = [Re(D), Im(D), Re(C), Im(C)] — real-valued throughout.

def _heston_odes(y: jnp.ndarray, args: tuple) -> jnp.ndarray:
    """RHS of Heston Riccati ODEs — pure real arithmetic.

    y    = [DR, DI, CR, CI]
    args = (u_r, u_i, kappa, theta, xi, rho)  — all real scalars
    """
    u_r, u_i, kappa, theta, xi, rho = args
    DR, DI, CR, CI = y[0], y[1], y[2], y[3]

    # Lord-Kahl (2010) Eq. 5: dD/dτ = −½(u²+iu) + bD + cD²
    # Note the NEGATIVE sign on a. Positive sign is wrong (D diverges rather
    # than converging to the stable Riccati root → negative call prices).
    a_R = -0.5 * (u_r**2 - u_i**2 - u_i)
    a_I = -0.5 * u_r * (2.0 * u_i + 1.0)
    b_R = -kappa - rho * xi * u_i
    b_I = rho * xi * u_r
    c   = 0.5 * xi**2

    dDR = a_R + b_R * DR - b_I * DI + c * (DR**2 - DI**2)
    dDI = a_I + b_R * DI + b_I * DR + 2.0 * c * DR * DI
    dCR = kappa * DR
    dCI = kappa * DI

    return jnp.array([dDR, dDI, dCR, dCI])


def _rk4_solve(u_r: float, u_i: float, T: float, kappa, theta, xi, rho) -> jnp.ndarray:
    """Fixed-step RK4 for Heston Riccati ODE. Returns y(T) = [DR, DI, CR, CI].

    Uses jax.lax.scan — composes correctly inside jit+vmap+vmap unlike
    Diffrax's PIDController (lax.while_loop).
    """
    n_steps = config.N_ODE_STEPS
    dt = T / n_steps
    args = (u_r, u_i, kappa, theta, xi, rho)

    def step(y, _):
        k1 = _heston_odes(y, args)
        k2 = _heston_odes(y + 0.5 * dt * k1, args)
        k3 = _heston_odes(y + 0.5 * dt * k2, args)
        k4 = _heston_odes(y + dt * k3, args)
        return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), None

    y0 = jnp.zeros(4)
    y_T, _ = jax.lax.scan(step, y0, None, length=n_steps)
    return y_T


def heston_char_fn(
    u_real: float,
    u_imag: float,
    T: float,
    p: HestonParams,
    log_F: float = 0.0,
) -> tuple:
    """Evaluate Heston characteristic function at u = u_real + i·u_imag, maturity T.

    Returns (phi_real, phi_imag) — real and imaginary parts separately.

    φ = exp(C·θ + D·v₀ + i·u·log F)
    i·u·log F: real part = −u_imag·log_F, imaginary part = u_real·log_F.
    """
    y_T = _rk4_solve(u_real, u_imag, T, p.kappa, p.theta, p.xi, p.rho)
    DR_T, DI_T, CR_T, CI_T = y_T[0], y_T[1], y_T[2], y_T[3]

    # exponent_R = Re(C·θ + D·v₀ + i·u·log_F)
    # exponent_I = Im(C·θ + D·v₀ + i·u·log_F)
    exp_R = CR_T * p.theta + DR_T * p.v0 - u_imag * log_F
    exp_I = CI_T * p.theta + DI_T * p.v0 + u_real * log_F

    mag   = jnp.exp(exp_R)
    phi_R = mag * jnp.cos(exp_I)
    phi_I = mag * jnp.sin(exp_I)
    return phi_R, phi_I


# ── Carr-Madan option pricing via Gauss-Laguerre quadrature ──────────────────

def _gauss_laguerre_nodes_weights(n: int):
    """Return nodes and weights for n-point Gauss-Laguerre quadrature on [0, ∞)."""
    import numpy as np
    nodes, weights = np.polynomial.laguerre.laggauss(n)
    return jnp.array(nodes), jnp.array(weights)


def price_call(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    n_quad: int = None,
    alpha: float = None,
) -> jnp.ndarray:
    """Price a European call via Carr-Madan (1999) with Gauss-Laguerre quadrature.

    C = e^{−αk−rT}/π · ∫₀^∞ Re[e^{−ivk} · ψ(v)] dv
    where k = log K, v is the real integration variable, and
      ψ(v) = e^{rT} · φ(v − (α+1)i) / [α²+α−v² + i(2α+1)v]

    u_shifted = v − (α+1)i  →  u_real=v, u_imag=−(α+1). Pure real arithmetic.
    """
    if n_quad is None:
        n_quad = config.N_QUADRATURE_POINTS
    if alpha is None:
        alpha = config.CARR_MADAN_ALPHA

    nodes, weights = _gauss_laguerre_nodes_weights(n_quad)

    log_K = jnp.log(K)
    log_F = jnp.log(S) + (r - q) * T
    u_imag = -(alpha + 1.0)
    erT    = jnp.exp(r * T)

    def integrand(v):
        phi_R, phi_I = heston_char_fn(v, u_imag, T, p, log_F=log_F)

        # denom = (α²+α−v²) + i·(2α+1)·v
        denom_R  = alpha**2 + alpha - v**2
        denom_I  = (2.0 * alpha + 1.0) * v
        denom_sq = denom_R**2 + denom_I**2 + 1e-30  # guard at v=0

        # psi = e^{rT} · phi / denom (complex division expanded)
        psi_R = erT * (phi_R * denom_R + phi_I * denom_I) / denom_sq
        psi_I = erT * (phi_I * denom_R - phi_R * denom_I) / denom_sq

        # Re[e^{−ivk} · psi] = cos(vk)·psi_R + sin(vk)·psi_I
        c = jnp.cos(v * log_K)
        s = jnp.sin(v * log_K)
        return c * psi_R + s * psi_I

    # Gauss-Laguerre weight correction: integrand lacks e^{−v} factor
    integrand_vals = jax.vmap(lambda v: integrand(v) * jnp.exp(v))(nodes)
    integral = jnp.sum(weights * integrand_vals)

    return jnp.exp(-alpha * log_K - r * T) / jnp.pi * integral


def price_put(
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
    p: HestonParams,
    **kwargs,
) -> jnp.ndarray:
    """European put via put-call parity: P = C − S·e^{−qT} + K·e^{−rT}."""
    call = price_call(S, K, T, r, q, p, **kwargs)
    return call - S * jnp.exp(-q * T) + K * jnp.exp(-r * T)


def price_surface(
    S: float,
    strikes: jnp.ndarray,
    maturities: jnp.ndarray,
    r: float,
    q: float,
    p: HestonParams,
) -> jnp.ndarray:
    """Price a flat list of (K, T) pairs. Returns call prices array."""
    def price_one(K, T):
        return price_call(S, K, T, r, q, p)

    return jax.vmap(price_one)(strikes, maturities)


# ── Black-Scholes IV extractor (Newton-Raphson, JAX) ─────────────────────────

def bs_call(S, K, T, r, q, sigma):
    """Black-Scholes call price."""
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    from jax.scipy.stats import norm
    return S * jnp.exp(-q * T) * norm.cdf(d1) - K * jnp.exp(-r * T) * norm.cdf(d2)


def implied_vol_newton(
    C_market: float,
    S: float,
    K: float,
    T: float,
    r: float,
    q: float,
) -> float:
    """Extract BS implied vol via Newton-Raphson (lax.scan).

    Returns NaN on convergence failure — never a fallback (CLAUDE.md convention).
    """
    sigma = jnp.array(config.IV_INITIAL_GUESS)

    def step(sigma, _):
        price = bs_call(S, K, T, r, q, sigma)
        d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
        from jax.scipy.stats import norm
        vega = S * jnp.exp(-q * T) * norm.pdf(d1) * jnp.sqrt(T)
        vega = jnp.maximum(vega, 1e-10)
        sigma_new = jnp.clip(sigma - (price - C_market) / vega, 1e-4, 10.0)
        return sigma_new, None

    sigma, _ = jax.lax.scan(step, sigma, None, length=config.IV_SOLVER_MAX_ITER)

    final_error = jnp.abs(bs_call(S, K, T, r, q, sigma) - C_market)
    return jnp.where(final_error < config.IV_SOLVER_TOL, sigma, jnp.nan)


def heston_implied_vols(
    S: float,
    strikes: jnp.ndarray,
    maturities: jnp.ndarray,
    r: float,
    q: float,
    p: HestonParams,
) -> jnp.ndarray:
    """Heston-model implied vols for a strip of (K, T) pairs."""
    call_prices = price_surface(S, strikes, maturities, r, q, p)

    def extract_iv(C, K, T):
        return implied_vol_newton(C, S, K, T, r, q)

    return jax.vmap(extract_iv)(call_prices, strikes, maturities)
