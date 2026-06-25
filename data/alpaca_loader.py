"""
Options chain loader backed by Alpaca Market Data API.

Uses Alpaca's OPRA or indicative feed to get real-time NBBO bid/ask quotes.
Preferred data source for live trading — data and execution from the same provider
eliminates cross-provider timestamp mismatches.

Feed tiers (set ALPACA_FEED env var or pass feed= to constructor):
  indicative  — free tier, near-real-time NBBO (suitable for most liquid names)
  opra        — full OPRA consolidated BBO, requires Unlimited data plan

get_option_chain() returns Dict[str, OptionsSnapshot] keyed by OCC symbol.
Each snapshot has: latest_quote (bid/ask), implied_volatility, greeks (delta/gamma/vega).
We rebuild bid/ask from NBBO and re-derive IV from prices (same as CBOE/Polygon pipeline).
"""

from __future__ import annotations

import os
import re
import warnings
from datetime import date
from typing import Optional

import numpy as np

from data.cleaner import OptionsChain
from data.rates import get_rates


_OCC_RE = re.compile(r'^([A-Z1-9]{1,6})(\d{6})([CP])(\d{8})$')


def _parse_occ(symbol: str) -> Optional[tuple]:
    """Parse OCC symbol into (underlying, expiry_str, opt_type, strike).

    Returns None on non-OCC input.
    expiry_str: "YYYY-MM-DD"
    """
    m = _OCC_RE.match(symbol)
    if not m:
        return None
    yymmdd = m.group(2)
    year = int("20" + yymmdd[:2])
    month = int(yymmdd[2:4])
    day = int(yymmdd[4:6])
    return (
        m.group(1),
        f"{year:04d}-{month:02d}-{day:02d}",
        m.group(3),
        int(m.group(4)) / 1000.0,
    )


class AlpacaLoader:
    """Options chain loader backed by Alpaca Market Data API.

    Fetches a full option chain snapshot via get_option_chain() which returns
    the latest trade, NBBO quote, and greeks for every listed contract.

    Args:
        api_key: Alpaca API key. Falls back to ALPACA_API_KEY env var.
        secret_key: Alpaca secret key. Falls back to ALPACA_SECRET_KEY env var.
        feed: 'indicative' (free, near-real-time) or 'opra' (full NBBO, paid plan).
        call_only: Return only call options. Default False — fetch both calls and puts
            so calibration can see the wings and identify skew (rho) and vol-of-vol (xi);
            the cleaner keeps the OTM side per strike and derives put IVs via parity.
        min_open_interest: Minimum OI filter. 0 = off (OI not in chain snapshot).
        max_bid_ask_frac: Drop contracts where (ask-bid)/mid > this value.
        min_ttm_days: Drop contracts expiring sooner than this.
        max_ttm_days: Drop contracts expiring later than this.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        feed: str = 'indicative',
        call_only: bool = False,
        min_open_interest: int = 0,
        max_bid_ask_frac: float = 0.50,
        min_ttm_days: int = 5,
        max_ttm_days: int = 730,
    ):
        try:
            from alpaca.data.historical.option import OptionHistoricalDataClient
        except ImportError:
            raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")

        key = api_key or os.environ.get("ALPACA_API_KEY")
        secret = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise ValueError(
                "Alpaca keys required. Pass api_key/secret_key or set "
                "ALPACA_API_KEY and ALPACA_SECRET_KEY in .env"
            )

        self._key = key
        self._secret = secret
        self._client = OptionHistoricalDataClient(key, secret)
        self._feed_str = feed
        self._call_only = call_only
        self._min_oi = min_open_interest
        self._max_ba_frac = max_bid_ask_frac
        self._min_ttm_days = min_ttm_days
        self._max_ttm_days = max_ttm_days

    def fetch(
        self,
        ticker: str,
        snapshot_date: str,
        r: Optional[float] = None,
        q: Optional[float] = None,
    ) -> OptionsChain:
        """Fetch NBBO options chain snapshot.

        Args:
            ticker: Underlying symbol ("SPY", "SPX", etc.)
            snapshot_date: "YYYY-MM-DD" — used for TTM and rate lookup.
            r: Risk-free rate (continuous). None = auto-fetch from FRED/table.
            q: Dividend yield (continuous). None = auto-fetch.

        Returns:
            OptionsChain where bid_prices/ask_prices are real-time NBBO.
        """
        from alpaca.data.requests import OptionChainRequest
        from alpaca.data.enums import OptionsFeed

        snap_date = snapshot_date[:10]
        today = date.fromisoformat(snap_date)
        snap_time = f"{snap_date}T16:00:00"

        if r is None or q is None:
            r_auto, q_auto = get_rates(ticker, snap_date)
            r = r if r is not None else r_auto
            q = q if q is not None else q_auto

        feed = OptionsFeed.OPRA if self._feed_str == "opra" else OptionsFeed.INDICATIVE

        spot = self._fetch_spot(ticker)

        try:
            req = OptionChainRequest(
                underlying_symbol=ticker,
                feed=feed,
            )
            chain_data = self._client.get_option_chain(req)
        except Exception as e:
            err_str = str(e)
            if "401" in err_str:
                warnings.warn(
                    f"[alpaca-loader] 401 on options chain for {ticker}. "
                    "Alpaca options data requires options trading to be enabled on your account. "
                    "Visit app.alpaca.markets → Account → Options Trading to apply."
                )
            else:
                warnings.warn(f"[alpaca-loader] get_option_chain failed for {ticker}: {e}")
            return _empty_chain(ticker, snap_time, r, q, spot=spot)

        return self._parse(chain_data, ticker, snap_time, today, spot, r, q)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _fetch_spot(self, ticker: str) -> float:
        """Fetch underlying spot price.

        Uses NBBO mid when both bid and ask are positive.
        Falls back to bid-only (pre-market / after-hours) or last trade price.
        """
        try:
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest, StockLatestBarRequest
            sc = StockHistoricalDataClient(self._key, self._secret)
            resp = sc.get_stock_latest_quote(StockLatestQuoteRequest(symbol_or_symbols=ticker))
            q = resp[ticker]
            bid = float(q.bid_price or 0)
            ask = float(q.ask_price or 0)
            if bid > 0 and ask > 0:
                return (bid + ask) / 2.0
            if bid > 0:
                return bid
            # Fall back to latest bar close
            bar_resp = sc.get_stock_latest_bar(StockLatestBarRequest(symbol_or_symbols=ticker))
            close = float(bar_resp[ticker].close or 0)
            return close
        except Exception as e:
            warnings.warn(f"[alpaca-loader] Could not fetch spot for {ticker}: {e}")
            return 0.0

    def _parse(
        self,
        chain_data: dict,
        ticker: str,
        snap_time: str,
        today: date,
        spot: float,
        r: float,
        q: float,
    ) -> OptionsChain:
        strikes, maturities, mids, bids, asks, ois, opt_types = [], [], [], [], [], [], []

        for occ_symbol, snapshot in chain_data.items():
            parsed = _parse_occ(occ_symbol)
            if parsed is None:
                continue
            _, expiry_str, opt_type, strike = parsed

            if self._call_only and opt_type != "C":
                continue

            try:
                exp_date = date.fromisoformat(expiry_str)
            except ValueError:
                continue

            ttm_days = (exp_date - today).days
            if ttm_days < self._min_ttm_days or ttm_days > self._max_ttm_days:
                continue

            lq = getattr(snapshot, "latest_quote", None)
            if lq is None:
                continue

            bid = float(getattr(lq, "bid_price", 0) or 0)
            ask = float(getattr(lq, "ask_price", 0) or 0)
            if bid <= 0 or ask <= 0 or ask < bid:
                continue

            mid = (bid + ask) / 2.0
            spread_frac = (ask - bid) / mid if mid > 0 else float("inf")
            if spread_frac > self._max_ba_frac:
                continue

            T = ttm_days / 365.0
            strikes.append(strike)
            maturities.append(T)
            mids.append(mid)
            bids.append(bid)
            asks.append(ask)
            ois.append(0.0)
            opt_types.append(opt_type)

        if not strikes:
            warnings.warn(f"[alpaca-loader] No valid contracts for {ticker} at {snap_time}")
            return _empty_chain(ticker, snap_time, r, q, spot=spot)

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


def _empty_chain(ticker, snap_time, r, q, spot=0.0) -> OptionsChain:
    return OptionsChain(
        ticker=ticker, snapshot_time=snap_time, spot=spot, r=r, q=q,
        strikes=np.array([]), maturities=np.array([]),
        mid_prices=np.array([]), bid_prices=np.array([]),
        ask_prices=np.array([]), open_interest=np.array([]),
        option_type=np.array([]),
    )
