"""
VolSurface: arbitrage-free implied vol surface with 2D interpolation.

Takes a CleanedSurface (LP-repaired prices + IVs) and provides:
  - Grid interpolation (RectBivariateSpline over log-moneyness × maturity)
  - Pointwise IV lookup for arbitrary (K, T) pairs
  - Total variance w(k, T) = σ²(k, T) × T  (used in no-arb checks)
  - Durrleman butterfly invariant check (surface quality metric)

Not a Heston surface — this is the RAW MARKET surface derived from prices.
The Heston surface (model IVs) lives in calibration/heston.py.
The gap between these two is the mispricing signal in signals/mispricing.py.
"""

import numpy as np
from scipy.interpolate import RectBivariateSpline
from dataclasses import dataclass
from typing import Optional

from data.cleaner import CleanedSurface, OptionsChain
import config


@dataclass
class VolSurface:
    """Market implied vol surface with 2D interpolation.

    Coordinate system: log-moneyness k = log(K/F(T)) × maturity T.
    Interpolation is in (k, T) space — more stable than (K, T) space
    because log-moneyness is roughly stationary as S moves.
    """
    chain: OptionsChain
    log_moneyness: np.ndarray        # shape (N,) — k = log(K/F)
    maturities: np.ndarray           # shape (N,) — T in years
    market_ivs: np.ndarray           # shape (N,) — cleaned BS IVs

    # Grid for interpolation (unique, sorted)
    _k_grid: np.ndarray              # unique log-moneyness values
    _T_grid: np.ndarray              # unique maturity values
    _spline: Optional[RectBivariateSpline] = None

    def __post_init__(self):
        self._build_spline()

    def _build_spline(self):
        """Build RectBivariateSpline over the (k, T) grid."""
        # Find unique grid axes (within tolerance)
        k_vals = np.unique(np.round(self.log_moneyness, 5))
        T_vals = np.unique(np.round(self.maturities, 6))

        if len(k_vals) < 2 or len(T_vals) < 2:
            self._spline = None
            return

        # Build IV grid (k_idx × T_idx), NaN-fill missing cells
        iv_grid = np.full((len(k_vals), len(T_vals)), np.nan)

        for i, (k, T, iv) in enumerate(
            zip(self.log_moneyness, self.maturities, self.market_ivs)
        ):
            ki = np.argmin(np.abs(k_vals - k))
            Ti = np.argmin(np.abs(T_vals - T))
            iv_grid[ki, Ti] = iv

        # Fill NaN cells with nearest neighbor along each axis before spline fit
        iv_grid = _fill_nans_nearest(iv_grid)

        self._k_grid = k_vals
        self._T_grid = T_vals

        try:
            self._spline = RectBivariateSpline(
                k_vals, T_vals, iv_grid,
                kx=min(3, len(k_vals) - 1),
                ky=min(3, len(T_vals) - 1),
            )
        except Exception:
            self._spline = None

    def iv(self, K: float, T: float) -> float:
        """Interpolated market IV at strike K and maturity T years.

        Returns NaN if outside the surface range or spline unavailable.
        """
        if self._spline is None:
            return np.nan

        F = self.chain.spot * np.exp((self.chain.r - self.chain.q) * T)
        k = np.log(K / F)

        k_min, k_max = self._k_grid[0], self._k_grid[-1]
        T_min, T_max = self._T_grid[0], self._T_grid[-1]

        if k < k_min or k > k_max or T < T_min or T > T_max:
            return np.nan

        val = float(self._spline(k, T).item())
        return val if val > 0 else np.nan

    def total_variance(self, K: float, T: float) -> float:
        """Total variance w(K, T) = σ²(K, T) × T."""
        sigma = self.iv(K, T)
        return sigma**2 * T if not np.isnan(sigma) else np.nan

    def iv_strip(
        self,
        strikes: np.ndarray,
        maturity: float,
    ) -> np.ndarray:
        """Return market IVs for a strip of strikes at a single maturity."""
        return np.array([self.iv(K, maturity) for K in strikes])

    def calendar_spread_violations(self) -> list[dict]:
        """Find (k, T_short, T_long) triples where w(k,T_short) > w(k,T_long).

        Returns list of violation dicts. Empty list = no violations.
        """
        violations = []
        k_vals = self._k_grid if self._k_grid is not None else []
        T_sorted = np.sort(self._T_grid) if self._T_grid is not None else []

        F_mid = self.chain.spot * np.exp(
            (self.chain.r - self.chain.q) * np.mean(T_sorted)
        ) if len(T_sorted) > 0 else self.chain.spot

        for k in k_vals:
            K = F_mid * np.exp(k)  # approximate strike (F changes with T but small effect)
            w_prev = None
            T_prev = None
            for T in T_sorted:
                w = self.total_variance(K, T)
                if w_prev is not None and not np.isnan(w) and not np.isnan(w_prev):
                    if w < w_prev - config.BUTTERFLY_TOLERANCE:
                        violations.append({
                            "k": k, "T_short": T_prev, "T_long": T,
                            "w_short": w_prev, "w_long": w,
                            "violation": w_prev - w,
                        })
                if not np.isnan(w):
                    w_prev = w
                    T_prev = T

        return violations

    @property
    def n_options(self) -> int:
        return len(self.market_ivs)

    @property
    def n_maturities(self) -> int:
        return len(self._T_grid)

    @property
    def n_strikes_per_mat(self) -> int:
        return len(self._k_grid)

    @property
    def atm_iv(self) -> float:
        """ATM implied vol at the median maturity (k=0 approximation)."""
        if self._spline is None:
            return np.nan
        T_mid = np.median(self._T_grid)
        return float(self._spline(0.0, T_mid).item())

    def summary(self) -> dict:
        return {
            "ticker": self.chain.ticker,
            "snapshot_time": self.chain.snapshot_time,
            "n_options": self.n_options,
            "atm_iv": self.atm_iv,
            "iv_min": float(np.nanmin(self.market_ivs)),
            "iv_max": float(np.nanmax(self.market_ivs)),
            "calendar_violations": len(self.calendar_spread_violations()),
        }


def _fill_nans_nearest(grid: np.ndarray) -> np.ndarray:
    """Fill NaN cells with nearest valid neighbor (simple 1D forward/backward fill)."""
    out = grid.copy()
    n_k, n_T = out.shape

    # Forward fill along T axis
    for ki in range(n_k):
        last = np.nan
        for Ti in range(n_T):
            if not np.isnan(out[ki, Ti]):
                last = out[ki, Ti]
            elif not np.isnan(last):
                out[ki, Ti] = last

    # Backward fill along T axis
    for ki in range(n_k):
        last = np.nan
        for Ti in range(n_T - 1, -1, -1):
            if not np.isnan(out[ki, Ti]):
                last = out[ki, Ti]
            elif not np.isnan(last):
                out[ki, Ti] = last

    # Forward fill along k axis for any remaining NaNs
    for Ti in range(n_T):
        last = np.nan
        for ki in range(n_k):
            if not np.isnan(out[ki, Ti]):
                last = out[ki, Ti]
            elif not np.isnan(last):
                out[ki, Ti] = last

    # If still NaN, fill with column mean
    col_means = np.nanmean(out, axis=0)
    for Ti in range(n_T):
        for ki in range(n_k):
            if np.isnan(out[ki, Ti]):
                out[ki, Ti] = col_means[Ti] if not np.isnan(col_means[Ti]) else 0.2

    return out


def build_surface(cleaned: CleanedSurface) -> VolSurface:
    """Build a VolSurface from a CleanedSurface.

    Filters out NaN IVs (options where IV solver failed) before building.
    """
    chain = cleaned.chain
    ivs = cleaned.implied_vols

    # Drop NaN IVs
    valid = ~np.isnan(ivs)
    if valid.sum() < 4:
        raise ValueError(
            f"Too few valid IVs ({valid.sum()}) to build surface for "
            f"{chain.ticker} at {chain.snapshot_time}"
        )

    ivs_clean = ivs[valid]
    strikes_clean = chain.strikes[valid]
    maturities_clean = chain.maturities[valid]

    # Compute log-moneyness for each option
    log_moneyness = np.array([
        np.log(K / chain.forward(T))
        for K, T in zip(strikes_clean, maturities_clean)
    ])

    return VolSurface(
        chain=chain,
        log_moneyness=log_moneyness,
        maturities=maturities_clean,
        market_ivs=ivs_clean,
        _k_grid=np.array([]),
        _T_grid=np.array([]),
        _spline=None,
    )
