"""
Options data loader — abstract interface + implementations.

  SyntheticLoader  — generates test chains from known Heston params (no data needed)
  CBOELoader       — CBOE DataShop CSV files (see data/cboe.py)
  PolygonLoader    — Polygon.io real-time API (stub; not yet implemented)

Primary source: CBOE DataShop. Good for SPX/SPY historical backtesting.
Real-time extension: Polygon.io once backtesting validates the strategy.
"""

import numpy as np
import jax.numpy as jnp
from typing import Optional, Protocol, runtime_checkable

from calibration.heston import HestonParams, heston_implied_vols, price_call
from data.cleaner import OptionsChain
import config


# ── Loader protocol ───────────────────────────────────────────────────────────

@runtime_checkable
class OptionsLoader(Protocol):
    """Minimal interface any data source must implement."""

    def fetch(
        self,
        ticker: str,
        snapshot_time: str,
    ) -> OptionsChain:
        """Fetch a raw options chain snapshot.

        Args:
            ticker: underlying symbol (e.g. "SPY", "SPX").
            snapshot_time: ISO 8601 timestamp for the snapshot.

        Returns:
            OptionsChain with all available strikes and maturities.
            Quality filtering happens in data.cleaner, not here.
        """
        ...


# ── Synthetic loader ──────────────────────────────────────────────────────────

class SyntheticLoader:
    """Generate synthetic options chains from known Heston parameters.

    Use for:
    - End-to-end pipeline tests (no data subscription required)
    - Backtesting signal logic against known-truth surfaces
    - Evaluating calibrator cold-start performance

    The generated chain mimics realistic market structure:
      - 6 maturities: 1m, 2m, 3m, 6m, 1y, 2y
      - 11 strikes per maturity: ±30% log-moneyness
      - Bid-ask spread: flat 1% of mid (unrealistically tight — fine for tests)
      - Open interest: uniform 1000 (no liquidity weighting in weights)
    """

    def __init__(
        self,
        params: HestonParams,
        S: float = 100.0,
        r: float = 0.05,
        q: float = 0.02,
        bid_ask_frac: float = 0.01,
        noise_vol: float = 0.002,
        seed: int = 0,
    ):
        self.params = params
        self.S = S
        self.r = r
        self.q = q
        self.bid_ask_frac = bid_ask_frac
        self.noise_vol = noise_vol
        self.rng = np.random.default_rng(seed)

    def fetch(self, ticker: str = "SYNTH", snapshot_time: str = "2026-06-21T00:00:00Z") -> OptionsChain:
        ttms = np.array([1/12, 2/12, 3/12, 6/12, 1.0, 2.0])
        log_moneyness_grid = np.linspace(-0.30, 0.30, 11)

        S = self.S
        pairs = [(S * np.exp(k), T) for T in ttms for k in log_moneyness_grid]
        strikes = np.array([p[0] for p in pairs])
        maturities = np.array([p[1] for p in pairs])
        n = len(strikes)

        # Call prices from Heston pricer
        call_prices = np.array([
            float(price_call(S, float(K), float(T), self.r, self.q, self.params))
            for K, T in zip(strikes, maturities)
        ])

        # Drop negative or near-zero prices (extreme strikes fail IV solver)
        valid = call_prices > 1e-4
        strikes = strikes[valid]
        maturities = maturities[valid]
        call_prices = call_prices[valid]
        n = len(strikes)

        # Add small noise to simulate bid-ask midpoint uncertainty
        if self.noise_vol > 0:
            noise = self.rng.normal(0, self.noise_vol, n)
            call_prices = np.maximum(call_prices * (1 + noise), 1e-4)

        # Bid-ask spread: symmetric around mid
        half_spread = call_prices * self.bid_ask_frac / 2.0
        bid_prices = call_prices - half_spread
        ask_prices = call_prices + half_spread

        return OptionsChain(
            ticker=ticker,
            snapshot_time=snapshot_time,
            spot=S,
            r=self.r,
            q=self.q,
            strikes=strikes,
            maturities=maturities,
            mid_prices=call_prices,
            bid_prices=bid_prices,
            ask_prices=ask_prices,
            open_interest=np.full(n, 1000.0),
            option_type=np.array(["C"] * n),
        )


# ── CBOE DataShop stub ────────────────────────────────────────────────────────

class CBOELoader:
    """CBOE DataShop loader — implemented in data/cboe.py.

    Args:
        data_dir: directory of CBOE CSV files. See data/cboe.py for file
            naming conventions and column schema.
        r: risk-free rate. If None, uses spx_rates()/spy_rates() approximation.
        q: dividend yield. Same fallback behaviour.
        kwargs: forwarded to data.cboe.CBOELoader constructor.

    Usage:
        loader = CBOELoader("/path/to/datashop/spx/")
        chain = loader.fetch("SPX", "2024-03-15")
    """

    def __init__(self, data_dir: str, r: float = None, q: float = None, **kwargs):
        from data.cboe import CBOELoader as _CBOELoader
        self._impl = _CBOELoader(data_dir=data_dir, **kwargs)
        self._r = r
        self._q = q

    def fetch(self, ticker: str, snapshot_date: str, **kwargs) -> OptionsChain:
        from data.rates import get_rates
        r = self._r
        q = self._q
        if r is None or q is None:
            r_default, q_default = get_rates(ticker, snapshot_date)
            r = r if r is not None else r_default
            q = q if q is not None else q_default
        return self._impl.fetch(ticker, snapshot_date, r=r, q=q, **kwargs)


# ── Polygon.io stub ───────────────────────────────────────────────────────────

class PolygonLoader:
    """Polygon.io options chain loader — delegates to data/polygon.py.

    Args:
        api_key: Polygon API key (or set POLYGON_API_KEY env var).
        call_only: Only return call options (default True).
        min_open_interest: Minimum OI to include a contract (default 0).

    Warning: Polygon mid-price used as risk-neutral price is a 9-15% vol error
    on illiquid strikes (wide bid-ask). Always apply clean_chain() before use.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        from data.polygon import PolygonLoader as _PolygonLoader
        self._impl = _PolygonLoader(api_key=api_key, **kwargs)

    def fetch(self, ticker: str, snapshot_date: str,
              r: Optional[float] = None, q: Optional[float] = None) -> OptionsChain:
        return self._impl.fetch(ticker, snapshot_date, r=r, q=q)


# ── Factory ───────────────────────────────────────────────────────────────────

def get_loader(source: str = "synthetic", **kwargs) -> OptionsLoader:
    """Factory: return the appropriate loader by name.

    Args:
        source: "synthetic" | "cboe" | "polygon"
        **kwargs: forwarded to the loader constructor.
    """
    if source == "synthetic":
        params = kwargs.pop("params", HestonParams(kappa=2.0, theta=0.04, xi=0.3, rho=-0.7, v0=0.04))
        return SyntheticLoader(params=params, **kwargs)
    elif source == "cboe":
        if "data_dir" not in kwargs:
            raise ValueError("CBOELoader requires data_dir= argument.")
        return CBOELoader(**kwargs)
    elif source == "polygon":
        return PolygonLoader(**kwargs)
    elif source == "dolthub":
        from data.dolthub import DoltHubLoader
        return DoltHubLoader(**kwargs)
    else:
        raise ValueError(
            f"Unknown data source: {source!r}. Use 'synthetic', 'cboe', 'polygon', or 'dolthub'."
        )
