"""
Polygon.io options chain loader.

Fetches real-time (or end-of-day) options snapshots via the Polygon REST API.

Endpoint: GET /v3/snapshot/options/{underlyingAsset}
Docs: https://polygon.io/docs/options/get_v3_snapshot_options__underlyingasset

Data quality notes (from research):
- Mid-price as risk-neutral price causes 9-15% IV error on illiquid strikes.
  Always filter to tight-spread options (MAX_BID_ASK_FRAC) before calibration.
- Polygon quotes are T+0 real-time; CBOE DataShop is T+1 EOD.
- OI is updated daily (not intraday). Use for weighting only, not liquidity filter.
- IV reported by Polygon uses Black-Scholes; re-derive from prices (same as CBOE).

Rate limit: 100 requests/min for Starter plan. Use batch endpoints where possible.

Usage:
    loader = PolygonLoader(api_key="your_key")
    chain = loader.fetch("SPY", "2024-03-15T09:30:00")
"""

import time
import warnings
from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
import json
import math

from data.cleaner import OptionsChain
from data.rates import get_rates


_BASE = "https://api.polygon.io"
_RETRY_DELAYS = [1.0, 2.0, 5.0]  # seconds, 3 retries


@dataclass
class PolygonConfig:
    api_key: str
    call_only: bool = True          # only fetch call options
    min_open_interest: int = 0      # 0 = no filter
    max_ttm_days: int = 730
    min_ttm_days: int = 5
    max_bid_ask_frac: float = 0.50  # max (ask-bid)/mid
    timeout: int = 10               # request timeout seconds


class PolygonLoader:
    """Polygon.io options chain loader.

    Args:
        api_key: Polygon API key. Also reads POLYGON_API_KEY env var if not passed.
        call_only: If True (default), only return call options.
        min_open_interest: Minimum OI filter (0 = off).

    Example:
        loader = PolygonLoader(api_key="abc123")
        chain = loader.fetch("SPY", "2024-03-15")
        cleaned = clean_chain(chain)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        call_only: bool = True,
        min_open_interest: int = 0,
    ):
        import os
        key = api_key or os.environ.get("POLYGON_API_KEY")
        if not key:
            raise ValueError(
                "Polygon API key required. Pass api_key= or set POLYGON_API_KEY env var. "
                "Free tier available at https://polygon.io"
            )
        self._cfg = PolygonConfig(
            api_key=key,
            call_only=call_only,
            min_open_interest=min_open_interest,
        )

    def fetch(
        self,
        ticker: str,
        snapshot_date: str,
        r: Optional[float] = None,
        q: Optional[float] = None,
    ) -> OptionsChain:
        """Fetch an options chain snapshot.

        Args:
            ticker: Underlying symbol (e.g. "SPY", "SPX", "AAPL").
            snapshot_date: "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SS". The API returns
                the most recent snapshot available; for historical dates, use the
                /v3/snapshot endpoint with as_of parameter (requires Starter plan+).
            r: Risk-free rate (continuous). If None, fetches from FRED via data.rates.
            q: Dividend yield (continuous). Same fallback.

        Returns:
            OptionsChain ready for clean_chain().
        """
        snap_date = snapshot_date[:10]
        snap_time = snapshot_date if len(snapshot_date) > 10 else f"{snap_date}T16:00:00"

        if r is None or q is None:
            r_fetched, q_fetched = get_rates(ticker, snap_date)
            r = r if r is not None else r_fetched
            q = q if q is not None else q_fetched

        raw_contracts = self._fetch_snapshot(ticker, snap_date)
        chain = self._parse(raw_contracts, ticker, snap_time, r, q)
        if chain.spot <= 0:
            chain.spot = self.fetch_underlying_price(ticker) or 0.0
        return chain

    # ── Internal ──────────────────────────────────────────────────────────

    def _fetch_snapshot(self, ticker: str, snap_date: str) -> list[dict]:
        """Fetch all option contracts for the underlying. Handles pagination."""
        contracts = []
        url = (
            f"{_BASE}/v3/snapshot/options/{ticker}"
            f"?limit=250&apiKey={self._cfg.api_key}"
        )
        if snap_date:
            url += f"&as_of={snap_date}"

        while url:
            data = self._get(url)
            if data is None:
                break
            results = data.get("results", [])
            contracts.extend(results)
            url = data.get("next_url")
            if url and "apiKey" not in url:
                url += f"&apiKey={self._cfg.api_key}"

        return contracts

    def _get(self, url: str) -> Optional[dict]:
        """HTTP GET with retry logic."""
        req = Request(url, headers={"Accept": "application/json"})
        for attempt, delay in enumerate([0] + _RETRY_DELAYS):
            if delay:
                time.sleep(delay)
            try:
                with urlopen(req, timeout=self._cfg.timeout) as resp:
                    return json.loads(resp.read())
            except HTTPError as e:
                if e.code == 429:  # rate limit
                    if attempt < len(_RETRY_DELAYS):
                        continue
                    warnings.warn(f"[polygon] Rate limit hit after {attempt+1} retries")
                    return None
                elif e.code in (401, 403):
                    raise ValueError(
                        f"Polygon API authentication failed (HTTP {e.code}). "
                        "Check your API key and plan permissions."
                    )
                warnings.warn(f"[polygon] HTTP {e.code} for {url[:80]}...")
                return None
            except URLError as e:
                warnings.warn(f"[polygon] Network error: {e}")
                return None
        return None

    def _parse(
        self,
        contracts: list[dict],
        ticker: str,
        snap_time: str,
        r: float,
        q: float,
    ) -> OptionsChain:
        """Parse Polygon snapshot contracts into OptionsChain."""
        import numpy as np

        snap_date = snap_time[:10]
        today = date.fromisoformat(snap_date)

        strikes, maturities, mids, bids, asks, ois, opt_types = [], [], [], [], [], [], []

        for c in contracts:
            details = c.get("details", {})
            day = c.get("day", {})
            greeks_raw = c.get("greeks", {})

            # Option type filter
            opt_type = details.get("contract_type", "").upper()
            if opt_type == "CALL":
                opt_type = "C"
            elif opt_type == "PUT":
                opt_type = "P"
            else:
                continue
            if self._cfg.call_only and opt_type != "C":
                continue

            # Strike and expiry
            K = details.get("strike_price")
            exp_str = details.get("expiration_date")
            if K is None or exp_str is None:
                continue
            try:
                exp_date = date.fromisoformat(exp_str)
            except ValueError:
                continue

            ttm_days = (exp_date - today).days
            if ttm_days < self._cfg.min_ttm_days or ttm_days > self._cfg.max_ttm_days:
                continue

            T = ttm_days / 365.0

            # Prices: prefer day.close, fall back to last trade price
            close = day.get("close") or day.get("last_trade", {}).get("price")
            bid = day.get("bid") or 0.0
            ask = day.get("ask") or close or 0.0
            if close is None or close <= 0:
                continue

            mid = (bid + ask) / 2.0 if bid and ask else close

            # Bid-ask spread filter
            if bid > 0 and ask > 0:
                spread_frac = (ask - bid) / mid if mid > 0 else float("inf")
                if spread_frac > self._cfg.max_bid_ask_frac:
                    continue

            # Open interest
            oi = c.get("open_interest") or day.get("open_interest") or 0
            if self._cfg.min_open_interest > 0 and oi < self._cfg.min_open_interest:
                continue

            strikes.append(float(K))
            maturities.append(T)
            mids.append(mid)
            bids.append(bid)
            asks.append(ask)
            ois.append(float(oi))
            opt_types.append(opt_type)

        if not strikes:
            warnings.warn(
                f"[polygon] No valid contracts found for {ticker} at {snap_time}. "
                f"Returned empty chain."
            )
            return OptionsChain(
                ticker=ticker, snapshot_time=snap_time,
                spot=0.0, r=r, q=q,
                strikes=np.array([]), maturities=np.array([]),
                mid_prices=np.array([]), bid_prices=np.array([]),
                ask_prices=np.array([]), open_interest=np.array([]),
                option_type=np.array([]),
            )

        # Spot price: use underlying snapshot if available
        # The /v3/snapshot/options response embeds the underlying price
        # in the first contract's underlying_asset block (if present)
        spot = self._extract_spot(contracts)

        return OptionsChain(
            ticker=ticker,
            snapshot_time=snap_time,
            spot=spot,
            r=r,
            q=q,
            strikes=np.array(strikes),
            maturities=np.array(maturities),
            mid_prices=np.array(mids),
            bid_prices=np.array(bids),
            ask_prices=np.array(asks),
            open_interest=np.array(ois),
            option_type=np.array(opt_types),
        )

    def _extract_spot(self, contracts: list[dict]) -> float:
        """Extract underlying spot price from the contracts list."""
        for c in contracts:
            ua = c.get("underlying_asset", {})
            price = ua.get("price") or ua.get("last_trade", {}).get("price")
            if price and price > 0:
                return float(price)
        # Fallback: derive from ATM options (call = put + F*e^{-rT} - K*e^{-rT})
        # Return 0 if we can't determine it — cleaner will handle missing spot
        return 0.0

    def fetch_underlying_price(self, ticker: str) -> Optional[float]:
        """Fetch current spot price for the underlying via the stocks/aggs endpoint."""
        url = (
            f"{_BASE}/v2/aggs/ticker/{ticker}/prev"
            f"?adjusted=true&apiKey={self._cfg.api_key}"
        )
        data = self._get(url)
        if data and data.get("results"):
            return float(data["results"][0].get("c", 0))
        return None
