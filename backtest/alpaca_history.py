"""
Historical SPY option-chain reconstruction from Alpaca trade bars.

WHY THIS EXISTS (read before trusting the output):
  - Polygon's plan has NO historical options data (NOT_AUTHORIZED on every endpoint).
  - Alpaca has historical *trade* bars (OHLCV) back to ~2024-02-06 (OPRA start), but
    NO historical bid/ask quotes on this plan, and its contracts-reference endpoint only
    lists currently-active contracts (expired chains cannot be enumerated).

So we reconstruct each day's chain by brute-forcing OCC symbols over a strike x expiry
grid and fetching daily bars; the option "price" is the trade CLOSE for that day.

HARD CAVEATS baked into any result built on this loader:
  - Price = last trade close, NOT mid. No bid/ask -> transaction cost cannot be measured,
    only modeled. Illiquid wing strikes trade rarely -> sparse and stale closes.
  - A strike with no trade that day is simply absent (survivorship toward liquid strikes).
These caveats are acceptable for the gap-DECOMPOSITION diagnostic (which asks whether the
model chases the market) but NOT for a clean P&L claim. Do not over-read dollar P&L here.
"""

import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import requests

from data.rates import get_rates

# NOTE: we deliberately do NOT import data.cleaner / data.surface here. They pull in
# cvxpy (LP arbitrage repair), whose native solver DLLs segfault when loaded into the
# same process as scipy+requests on this Windows box. The gap-decomposition diagnostic
# needs market IVs, not LP repair, so we extract IVs locally with scipy only.

_DATA = "https://data.alpaca.markets"


@dataclass
class HistChain:
    """Reconstructed historical chain. Same fields the pipeline's OptionsChain exposes,
    but standalone (no cvxpy dependency). Prices are trade closes (no bid/ask)."""
    ticker: str
    snapshot_time: str
    spot: float
    r: float
    q: float
    strikes: np.ndarray
    maturities: np.ndarray        # years, ACT/365
    prices: np.ndarray            # trade close
    option_type: np.ndarray       # 'C' / 'P'

    def forward(self, T: float) -> float:
        return self.spot * np.exp((self.r - self.q) * T)

    def market_ivs(self) -> np.ndarray:
        """Black-Scholes IVs from trade closes (puts mapped to calls via parity).
        NaN where the price is below intrinsic or the solver fails — never a fallback
        (matches cleaner._extract_implied_vols semantics)."""
        from scipy.stats import norm
        from scipy.optimize import brentq

        S, r, q = self.spot, self.r, self.q
        out = np.full(len(self.prices), np.nan)
        for i in range(len(self.prices)):
            price = float(self.prices[i])
            K = float(self.strikes[i])
            T = float(self.maturities[i])
            F = self.forward(T)
            if self.option_type[i] == "P":               # put -> call via parity
                price = price + F * np.exp(-r * T) - K * np.exp(-r * T)
            intrinsic = max(F * np.exp(-r * T) - K * np.exp(-r * T), 0.0)
            if price <= intrinsic + 1e-8:
                continue
            def err(sigma, K=K, T=T, price=price):
                d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
                d2 = d1 - sigma * np.sqrt(T)
                return (S * np.exp(-q * T) * norm.cdf(d1)
                        - K * np.exp(-r * T) * norm.cdf(d2)) - price
            try:
                out[i] = brentq(err, 1e-4, 10.0, xtol=1e-8)
            except (ValueError, RuntimeError):
                pass
        return out


def _occ(ticker: str, expiry: date, opt_type: str, strike: float) -> str:
    """Build an OCC option symbol, e.g. SPY240419C00500000 (strike*1000, 8 digits)."""
    return f"{ticker}{expiry:%y%m%d}{opt_type}{int(round(strike * 1000)):08d}"


def _nearest_friday(d: date) -> date:
    """Snap a date forward to the nearest Friday (SPY always lists Friday weeklies)."""
    return d + timedelta(days=(4 - d.weekday()) % 7)


def _session_dates(start: str, end: str) -> list[str]:
    """All weekday dates in [start, end] inclusive (holidays filter themselves out:
    days with no SPY stock bar are dropped downstream)."""
    d0 = date.fromisoformat(start)
    d1 = date.fromisoformat(end)
    out = []
    d = d0
    while d <= d1:
        if d.weekday() < 5:
            out.append(d.isoformat())
        d += timedelta(days=1)
    return out


class AlpacaHistoryLoader:
    """Reconstructs historical SPY OptionsChain snapshots from Alpaca trade bars.

    Drop-in for the backtest pipeline: .fetch(ticker, date) -> OptionsChain, same shape
    CBOELoader/PolygonLoader produce. bid/ask are set to the trade close (no quote data),
    open_interest=0 (unavailable), so liquidity-weighting stays disabled exactly as in
    the live loop (audit finding #11) — we test the pipeline as it actually runs.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        target_tenor_days: tuple[int, ...] = (14, 30, 45, 60, 90),
        strike_band: float = 0.12,      # +/- fraction of spot for the strike grid
        strike_step: float = 5.0,       # SPY $5 strikes are reliably listed & liquid
        fred_api_key: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ["ALPACA_API_KEY"]
        self.secret_key = secret_key or os.environ["ALPACA_SECRET_KEY"]
        self.target_tenor_days = target_tenor_days
        self.strike_band = strike_band
        self.strike_step = strike_step
        self.fred_api_key = fred_api_key or os.environ.get("FRED_API_KEY")
        self._h = {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    # ── HTTP helpers ──────────────────────────────────────────────────────────
    def _get(self, url: str, params: dict) -> dict:
        r = requests.get(url, headers=self._h, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def _spot(self, ticker: str, day: str) -> Optional[float]:
        """SPY stock close for `day` (the day's spot). None if no session that day."""
        nxt = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        j = self._get(f"{_DATA}/v2/stocks/bars", {
            "symbols": ticker, "timeframe": "1Day", "start": day, "end": nxt, "limit": 5,
        })
        bars = j.get("bars", {}).get(ticker, [])
        for b in bars:
            if b["t"][:10] == day:
                return float(b["c"])
        return None

    def _option_closes(self, symbols: list[str], day: str) -> dict[str, float]:
        """Daily close per option symbol that traded on `day`. Missing => absent."""
        nxt = (date.fromisoformat(day) + timedelta(days=1)).isoformat()
        out: dict[str, float] = {}
        # Alpaca caps the symbols query string; chunk conservatively.
        for i in range(0, len(symbols), 100):
            chunk = symbols[i:i + 100]
            page_token = None
            while True:
                params = {
                    "symbols": ",".join(chunk), "timeframe": "1Day",
                    "start": day, "end": nxt, "limit": 1000,
                }
                if page_token:
                    params["page_token"] = page_token
                j = self._get(f"{_DATA}/v1beta1/options/bars", params)
                for sym, bars in j.get("bars", {}).items():
                    for b in bars:
                        if b["t"][:10] == day:
                            out[sym] = float(b["c"])
                page_token = j.get("next_page_token")
                if not page_token:
                    break
        return out

    # ── Chain reconstruction ────────────────────────────────────────────────────
    def _expiries(self, day: date) -> list[date]:
        seen, out = set(), []
        for t in self.target_tenor_days:
            e = _nearest_friday(day + timedelta(days=t))
            if e not in seen:
                seen.add(e)
                out.append(e)
        return out

    def _strikes(self, spot: float) -> list[float]:
        lo = np.floor(spot * (1 - self.strike_band) / self.strike_step) * self.strike_step
        hi = np.ceil(spot * (1 + self.strike_band) / self.strike_step) * self.strike_step
        return list(np.arange(lo, hi + self.strike_step, self.strike_step))

    def fetch(self, ticker: str, snapshot_date: str) -> HistChain:
        """Reconstruct an OptionsChain for `ticker` on `snapshot_date` (YYYY-MM-DD).

        Raises ValueError if no session that day or too few contracts traded.
        """
        day = snapshot_date[:10]
        spot = self._spot(ticker, day)
        if spot is None:
            raise ValueError(f"No {ticker} session on {day} (holiday/weekend?)")

        d = date.fromisoformat(day)
        expiries = self._expiries(d)
        strikes = self._strikes(spot)

        # Build the candidate symbol grid (calls + puts), remember each symbol's metadata.
        meta: dict[str, tuple[float, float, str]] = {}  # sym -> (strike, T_years, type)
        symbols = []
        for e in expiries:
            ttm_days = (e - d).days
            if ttm_days <= 0:
                continue
            T = ttm_days / 365.0
            for K in strikes:
                for ot in ("C", "P"):
                    sym = _occ(ticker, e, ot, K)
                    meta[sym] = (K, T, ot)
                    symbols.append(sym)

        closes = self._option_closes(symbols, day)
        if len(closes) < 10:
            raise ValueError(f"{day}: only {len(closes)} contracts traded — too sparse")

        strikes_a, mats_a, prices_a, types_a = [], [], [], []
        for sym, px in closes.items():
            K, T, ot = meta[sym]
            strikes_a.append(K)
            mats_a.append(T)
            prices_a.append(px)
            types_a.append(ot)

        r, q = get_rates(ticker, day, self.fred_api_key)

        return HistChain(
            ticker=ticker,
            snapshot_time=f"{day}T20:00:00Z",   # ~market close ET
            spot=spot,
            r=r,
            q=q,
            strikes=np.array(strikes_a),
            maturities=np.array(mats_a),
            prices=np.array(prices_a),
            option_type=np.array(types_a),
        )

    def trading_days(self, ticker: str, start: str, end: str) -> list[str]:
        """Dates in [start,end] with a real SPY session (one stock-bar request)."""
        j = self._get(f"{_DATA}/v2/stocks/bars", {
            "symbols": ticker, "timeframe": "1Day", "start": start, "end": end, "limit": 1000,
        })
        return [b["t"][:10] for b in j.get("bars", {}).get(ticker, [])]


if __name__ == "__main__":
    # Bucket-1 self-test: pull a few real days and sanity-check IVs.
    loader = AlpacaHistoryLoader()
    days = loader.trading_days("SPY", "2024-04-01", "2024-04-05")
    print(f"sessions found: {days}")
    for day in days[:3]:
        chain = loader.fetch("SPY", day)
        ivs = chain.market_ivs()
        ok = ivs[~np.isnan(ivs)]
        print(
            f"{day}: spot={chain.spot:.2f} r={chain.r:.4f} q={chain.q:.4f} "
            f"contracts={len(chain.strikes)} valid_IV={len(ok)} "
            f"IV[min/med/max]={ok.min():.3f}/{np.median(ok):.3f}/{ok.max():.3f}"
        )
