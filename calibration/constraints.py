"""
Heston parameter validation utilities.

Centralises all structural checks on HestonParams so the calibrator,
evals, and signal layer can call a single authoritative function rather
than each reimplementing the Feller condition or bounds check.
"""

import numpy as np
from calibration.heston import HestonParams, feller_satisfied
import config


def validate_params(p: HestonParams) -> tuple[bool, list[str]]:
    """Check a HestonParams against bounds AND Feller condition.

    Returns (ok, list_of_failure_reasons). Empty list means all checks pass.
    """
    failures = []
    bounds = config.PARAM_BOUNDS

    for name, value in zip(
        ["kappa", "theta", "xi", "rho", "v0"],
        [p.kappa, p.theta, p.xi, p.rho, p.v0],
    ):
        lo, hi = bounds[name]
        if not (lo <= value <= hi):
            failures.append(f"{name}={value:.6f} outside [{lo}, {hi}]")

    if not feller_satisfied(p):
        lhs = 2 * p.kappa * p.theta
        rhs = p.xi ** 2
        failures.append(
            f"Feller violated: 2*kappa*theta={lhs:.6f} < xi^2={rhs:.6f} "
            f"(deficit={rhs - lhs:.6f})"
        )

    return len(failures) == 0, failures


def validation_report(p: HestonParams) -> str:
    """Human-readable validation summary."""
    ok, failures = validate_params(p)
    lines = [
        f"  kappa={p.kappa:.4f}  theta={p.theta:.4f}  xi={p.xi:.4f}  "
        f"rho={p.rho:.4f}  v0={p.v0:.4f}",
        f"  Feller: 2kth={2*p.kappa*p.theta:.4f}  xi2={p.xi**2:.4f}  "
        f"margin={2*p.kappa*p.theta - p.xi**2:+.4f}",
        f"  Status: {'VALID' if ok else 'INVALID'}",
    ]
    if failures:
        for f in failures:
            lines.append(f"  [!] {f}")
    return "\n".join(lines)


def feller_margin(p: HestonParams) -> float:
    """How far the params are from violating Feller: 2κθ − ξ². Negative = violated."""
    return 2.0 * p.kappa * p.theta - p.xi ** 2


def project_to_feller(p: HestonParams, safety: float = 0.999) -> HestonParams:
    """Project params onto the Feller-satisfied region (2κθ ≥ ξ²) by capping ξ.

    The calibration's soft Feller penalty has near-zero gradient at the boundary, so the
    optimizer can settle marginally PAST it. The signal layer then silently drops every
    signal on any Feller violation — the dominant cause of "never opens a position". This
    hard projection caps ξ ≤ safety·√(2κθ) when violated (κ, θ, ρ, v0 unchanged), so the
    fitted params are always valid. A near-boundary fit moves by a hair; a clean fit is
    untouched. Returns floatified params (JSON-serialisable).
    """
    kappa, theta = float(p.kappa), float(p.theta)
    xi, rho, v0 = float(p.xi), float(p.rho), float(p.v0)
    lhs = 2.0 * kappa * theta
    if xi * xi > lhs:
        lo, hi = config.PARAM_BOUNDS["xi"]
        xi = float(np.clip(safety * np.sqrt(max(lhs, 0.0)), lo, hi))
    return HestonParams(kappa=kappa, theta=theta, xi=xi, rho=rho, v0=v0)


def clip_to_bounds(p: HestonParams) -> HestonParams:
    """Hard-clip all parameters to their configured bounds.

    Used only as an emergency fallback — the calibrator's softplus/tanh
    transforms should keep params in-bounds during optimisation. This is
    a last-resort before passing params to the pricer.
    """
    bounds = config.PARAM_BOUNDS

    def clip(name, value):
        lo, hi = bounds[name]
        return float(np.clip(value, lo, hi))

    return HestonParams(
        kappa=clip("kappa", p.kappa),
        theta=clip("theta", p.theta),
        xi=clip("xi", p.xi),
        rho=clip("rho", p.rho),
        v0=clip("v0", p.v0),
    )
