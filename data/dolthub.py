"""
DoltHub free options loader — post-no-preference/options.

WHY THIS EXISTS (read before trusting the output):
  - CBOE DataShop / OptionMetrics / ORATS all cost money; no university WRDS access.
  - DoltHub `post-no-preference/options` is a FREE, daily-updated database of US
    option chains with bid/ask, implied vol, and greeks, queryable over a public
    SQL-over-HTTP API (no key). Coverage runs from ~2019 to present.
    Repo:  https://www.dolthub.com/repositories/post-no-preference/options

HARD CAVEATS baked into any result built on this loader (verified against the live API):
  - SPY ONLY. The SPX cash index is NOT in the dataset — only the SPY ETF among
    S&P 500 instruments. fetch("SPX", ...) raises.
  - NO open interest. The table has no OI column, so open_interest is set to 0
    ("unavailable") — exactly like AlpacaHistoryLoader. Liquidity weighting must
    fall back to uniform/spread-only (compute_weights collapses on all-zero OI;
    see backtest/gap_decomposition.py:76 for the uniform-weight convention).
  - NO underlying spot. The table has no underlying-price column. Spot is DERIVED
    per expiry via put-call parity  F = K + e^{rT}(C_mid - P_mid),  S = F·e^{-(r-q)T}
    (median across matched strikes), then written into underlying_bid/underlying_ask.
    underlying_bid == underlying_ask (we have no underlying quote → zero spread).
  - SPARSE chains. ~100 contracts/day (a near-the-money slice), NOT a full chain.
    "covers many options, not all." Thin wings; usable for a directional Heston
    surface fit, weak for tail calibration.
  - `vol` (implied_volatility) is source-provided of unknown provenance. The Heston
    pipeline re-derives IV from mid-prices in the cleaner anyway; we carry `vol`
    through only to populate the CBOE-format `implied_volatility` column.
  - Recorded `date` is not every trading day, especially pre-2022. Use
    trading_days() to discover which dates actually exist before iterating.

These caveats make the data fine for a free, quick-and-dirty backtest sanity check.
They do NOT make it production-grade NBBO. Do not over-read absolute dollar P&L.

Output format: the normaliser emits the exact CBOE DataShop CSV schema the rest of
the pipeline expects —
    quote_datetime, root, expiration, strike, option_type,
    bid, ask, open_interest, underlying_bid, underlying_ask, implied_volatility
so to_cboe_csv() writes files the existing CBOELoader reads directly. fetch() parses
inline (cvxpy-free) and returns the same OptionsChain shape.

Typical workflow:
    loader = DoltHubLoader()
    days   = loader.trading_days("SPY", "2023-01-01", "2023-12-31")
    chain  = loader.fetch("SPY", days[0])          # -> OptionsChain (calls only)
    # or dump CBOE-format CSVs for the existing CBOELoader / archival:
    loader.to_cboe_csv("SPY", "2023-06-15", "out/SPY_20230615.csv")
"""

import time
import urllib.parse
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import requests

from data.rates import get_rates

# NOTE: we deliberately do NOT import data.cleaner / data.cboe here. They import
# cvxpy (LP arbitrage repair), whose native solver DLLs segfault (0xC0000005) on
# import on this Windows box — same reason backtest/alpaca_history.py avoids them.
# So we mirror cleaner.OptionsChain locally and do the (simple) parsing inline.
# The fields below match data.cleaner.OptionsChain exactly, so this object is a
# drop-in for any pipeline stage that duck-types the chain (e.g. gap_decomposition).

_API = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options"
_PAGE = 1000  # rows per SQL page; we paginate via LIMIT/OFFSET defensively


@dataclass
class OptionsChain:
    """Mirror of data.cleaner.OptionsChain (kept cvxpy-free). Same field names so it
    is interchangeable wherever the pipeline duck-types the chain."""
    ticker: str
    snapshot_time: str
    spot: float
    r: float
    q: float
    strikes: np.ndarray
    maturities: np.ndarray        # years, ACT/365
    mid_prices: np.ndarray        # (bid + ask) / 2
    bid_prices: np.ndarray
    ask_prices: np.ndarray
    open_interest: np.ndarray
    option_type: np.ndarray       # 'C' / 'P'

    def forward(self, T: float) -> float:
        return self.spot * np.exp((self.r - self.q) * T)

    def log_moneyness(self) -> np.ndarray:
        F = np.vectorize(self.forward)(self.maturities)
        return np.log(self.strikes / F)

    def market_ivs(self) -> np.ndarray:
        """Black-Scholes IVs from MID prices (puts mapped to calls via parity).
        NaN where mid is below intrinsic or the solver fails — never a fallback
        (matches cleaner / AlpacaHistoryLoader.HistChain semantics). This lets a
        DoltHub chain plug straight into backtest.gap_decomposition._calibrate_session.
        Uses mid = (bid+ask)/2 — the real two-sided quote, not a trade close."""
        from scipy.stats import norm
        from scipy.optimize import brentq

        S, r, q = self.spot, self.r, self.q
        out = np.full(len(self.mid_prices), np.nan)
        for i in range(len(self.mid_prices)):
            price = float(self.mid_prices[i])
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


def _spy_rates_fallback(day: str) -> tuple[float, float]:
    """Last-resort r/q if FRED lookup fails (offline). Coarse, by era — SPY q ≈ 1.6%."""
    year = int(str(day)[:4])
    r = (0.053 if year >= 2024 else 0.045 if year >= 2023 else 0.025 if year >= 2022
         else 0.005 if year >= 2020 else 0.024 if year >= 2019 else 0.020)
    return r, 0.016


class DoltHubLoader:
    """Load SPY option-chain snapshots from the free DoltHub options database.

    Drop-in for the backtest pipeline: .fetch(ticker, date) -> OptionsChain, the same
    shape CBOELoader/AlpacaHistoryLoader produce. DoltHub rows are normalised into the
    CBOE DataShop CSV schema, then parsed inline (no cvxpy import; see module note).

    Args:
        branch: DoltHub branch (default "master" — the repo's default branch).
        r, q:   risk-free rate / dividend yield. If None, spy_rates(date) is used.
                Needed for the put-call-parity spot derivation and the IV solver.
        call_only: return only calls in fetch() (calibration uses calls; the cleaner
                   maps puts via parity). The CBOE CSV always contains both.
        timeout, max_retries: HTTP behaviour. The SQL API throws
                "context deadline exceeded" on full-table scans; date+symbol filters
                hit the leading primary key and are fast, but we retry a few times.
    """

    def __init__(
        self,
        branch: str = "master",
        r: Optional[float] = None,
        q: Optional[float] = None,
        call_only: bool = True,
        timeout: int = 30,
        max_retries: int = 4,
    ):
        self.branch = branch
        self._r = r
        self._q = q
        self.call_only = call_only
        self.timeout = timeout
        self.max_retries = max_retries

    # ── SQL-over-HTTP ────────────────────────────────────────────────────────────

    def _query(self, sql: str) -> list[dict]:
        """Run one SQL statement; return rows as dicts. Retries on deadline errors."""
        url = f"{_API}/{self.branch}?q={urllib.parse.quote(sql)}"
        last_msg = ""
        for attempt in range(self.max_retries):
            resp = requests.get(url, timeout=self.timeout)
            resp.raise_for_status()
            j = resp.json()
            if j.get("query_execution_status") == "Success":
                return j.get("rows", [])
            last_msg = j.get("query_execution_message", "")
            if "deadline" in last_msg.lower():
                time.sleep(1.0 + attempt)  # back off; full scans are slow
                continue
            raise RuntimeError(f"DoltHub query failed: {last_msg}\nSQL: {sql}")
        raise RuntimeError(
            f"DoltHub query timed out after {self.max_retries} tries: {last_msg}\nSQL: {sql}"
        )

    def _query_day(self, ticker: str, day: str) -> list[dict]:
        """All option_chain rows for one symbol on one recorded date (paginated)."""
        rows: list[dict] = []
        offset = 0
        while True:
            sql = (
                "SELECT date, expiration, strike, call_put, bid, ask, vol "
                "FROM option_chain "
                f"WHERE act_symbol = '{ticker}' AND date = '{day}' "
                f"ORDER BY expiration, strike, call_put LIMIT {_PAGE} OFFSET {offset}"
            )
            page = self._query(sql)
            rows.extend(page)
            if len(page) < _PAGE:
                break
            offset += _PAGE
        return rows

    # ── Public: date discovery ────────────────────────────────────────────────────

    @staticmethod
    def candidate_days(start: str, end: str) -> list[str]:
        """Business days (Mon–Fri) in [start, end] — instant, no network. Holidays and
        unrecorded dates are weeded out downstream (has_day/fetch return empty)."""
        return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)]

    def has_day(self, ticker: str, day: str) -> bool:
        """True if `ticker` has any rows on `day` (one fast COUNT — exact-date PK seek)."""
        rows = self._query(
            f"SELECT COUNT(*) n FROM option_chain WHERE act_symbol = '{ticker}' AND date = '{day}'"
        )
        return bool(rows) and int(rows[0]["n"]) > 0

    def trading_days(self, ticker: str, start: str, end: str, verbose: bool = False) -> list[str]:
        """Recorded dates in [start, end] that actually have `ticker` rows.

        Probes each business day with a fast COUNT (one query per day) — DISTINCT over a
        range is NOT usable here: `date` is the leading primary key but `act_symbol` is
        only the 2nd key column, so a range scan reads every symbol's rows and the SQL
        API times out. Coverage is sparser pre-2022. For multi-year spans, cache the
        result; or skip discovery entirely and iterate candidate_days() with try/except.
        """
        out = []
        for day in self.candidate_days(start, end):
            if self.has_day(ticker, day):
                out.append(day)
                if verbose:
                    print(f"  {day} ok")
        return out

    # ── Normalisation: DoltHub rows -> CBOE DataShop DataFrame ──────────────────────

    def _cboe_dataframe(self, ticker: str, day: str, r: float, q: float) -> pd.DataFrame:
        """Build a CBOE-format DataFrame for one day, deriving spot via put-call parity."""
        rows = self._query_day(ticker, day)
        if not rows:
            raise ValueError(
                f"No {ticker} rows on {day} in DoltHub (holiday, or date not recorded). "
                f"Use trading_days() to list available dates."
            )

        df = pd.DataFrame(rows)
        for c in ("strike", "bid", "ask", "vol"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["call_put"] = df["call_put"].astype(str).str.strip().str.lower()
        df["mid"] = (df["bid"] + df["ask"]) / 2.0

        spot = self._spot_from_parity(df, day, r, q)

        out = pd.DataFrame({
            "quote_datetime": f"{day} 16:00:00",
            "root": ticker,
            "expiration": df["expiration"],
            "strike": df["strike"],
            "option_type": np.where(df["call_put"].str.startswith("c"), "C", "P"),
            "bid": df["bid"],
            "ask": df["ask"],
            "open_interest": 0,            # not in dataset — uniform/spread weighting only
            "underlying_bid": spot,        # derived spot; no real underlying quote
            "underlying_ask": spot,
            "implied_volatility": df["vol"],  # source-provided; pipeline re-derives IV
        })
        # Keep only rows with a usable two-sided quote.
        out = out[(out["bid"].notna()) & (out["ask"].notna()) & (out["ask"] >= out["bid"])]
        if len(out) == 0:
            raise ValueError(f"No two-sided quotes for {ticker} on {day}.")
        return out.reset_index(drop=True)

    @staticmethod
    def _spot_from_parity(df: pd.DataFrame, day: str, r: float, q: float) -> float:
        """Derive underlying spot from put-call parity, median across matched strikes.

        F = K + e^{rT}(C_mid - P_mid)  per (expiration, strike) pair with both legs;
        S = median over expiries of  median_K(F) · e^{-(r-q)T}.
        """
        day_d = pd.Timestamp(day)
        spots = []
        for expiry, grp in df.groupby("expiration"):
            T = max((pd.Timestamp(expiry) - day_d).days, 0) / 365.0
            if T <= 0:
                continue
            calls = grp[grp["call_put"].str.startswith("c")].set_index("strike")["mid"]
            puts = grp[grp["call_put"].str.startswith("p")].set_index("strike")["mid"]
            common = calls.index.intersection(puts.index)
            if len(common) == 0:
                continue
            K = common.to_numpy(dtype=float)
            fwd = K + np.exp(r * T) * (calls.loc[common].to_numpy() - puts.loc[common].to_numpy())
            spots.append(float(np.median(fwd)) * np.exp(-(r - q) * T))
        if not spots:
            raise ValueError(
                f"Cannot derive spot for {day}: no matched call/put pairs for parity. "
                f"Pass r/q and a spot override, or pick a denser date."
            )
        return float(np.median(spots))

    # ── Public: fetch / export ──────────────────────────────────────────────────────

    def _rates(self, ticker: str, day: str) -> tuple[float, float]:
        if self._r is not None and self._q is not None:
            return self._r, self._q
        try:
            r, q = get_rates(ticker, day)
        except Exception:
            r, q = _spy_rates_fallback(day)
        return (self._r if self._r is not None else r,
                self._q if self._q is not None else q)

    def fetch(self, ticker: str, snapshot_date: str) -> OptionsChain:
        """Fetch one day's SPY chain as an OptionsChain.

        Raises ValueError for SPX (not in dataset) or empty/parity-less dates.
        """
        ticker = ticker.upper()
        if ticker != "SPY":
            raise ValueError(
                f"DoltHub post-no-preference/options only has SPY among S&P 500 "
                f"instruments; {ticker!r} (e.g. SPX) is not in the dataset."
            )
        r, q = self._rates(ticker, snapshot_date)
        df = self._cboe_dataframe(ticker, snapshot_date, r, q)

        if self.call_only:
            df = df[df["option_type"] == "C"]
        day_d = pd.Timestamp(snapshot_date)
        ttm = (pd.to_datetime(df["expiration"]) - day_d).dt.days.clip(lower=0) / 365.0
        mid = (df["bid"] + df["ask"]) / 2.0
        valid = (df["strike"] > 0) & (ttm > 0) & (mid > 0) & (df["bid"] >= 0)
        if valid.sum() == 0:
            raise ValueError(f"No valid {ticker} contracts on {snapshot_date} after filtering.")
        df, ttm, mid = df[valid], ttm[valid], mid[valid]

        return OptionsChain(
            ticker=ticker,
            snapshot_time=f"{snapshot_date}T16:00:00Z",
            spot=float(df["underlying_bid"].iloc[0]),
            r=r, q=q,
            strikes=df["strike"].to_numpy(dtype=np.float64),
            maturities=ttm.to_numpy(dtype=np.float64),
            mid_prices=mid.to_numpy(dtype=np.float64),
            bid_prices=df["bid"].to_numpy(dtype=np.float64),
            ask_prices=df["ask"].to_numpy(dtype=np.float64),
            open_interest=np.zeros(len(df), dtype=np.float64),  # unavailable
            option_type=df["option_type"].to_numpy(),
        )

    def to_cboe_csv(self, ticker: str, snapshot_date: str, out_path: str) -> str:
        """Write one day's chain as a CBOE DataShop-format CSV. Returns out_path."""
        ticker = ticker.upper()
        if ticker != "SPY":
            raise ValueError(f"Only SPY is available; {ticker!r} is not in the dataset.")
        r, q = self._rates(ticker, snapshot_date)
        df = self._cboe_dataframe(ticker, snapshot_date, r, q)
        df.to_csv(out_path, index=False)
        return out_path


if __name__ == "__main__":
    # Self-test: discover a week of SPY dates, check spot derivation + chain shape,
    # and dump one CBOE-format CSV.
    loader = DoltHubLoader()
    days = loader.trading_days("SPY", "2023-06-12", "2023-06-16", verbose=True)
    print(f"available SPY dates: {days}")
    for day in days[:3]:
        chain = loader.fetch("SPY", day)
        print(
            f"{day}: spot~{chain.spot:.2f} r={chain.r:.4f} q={chain.q:.4f} "
            f"calls={len(chain.strikes)} "
            f"K[min/max]={chain.strikes.min():.0f}/{chain.strikes.max():.0f} "
            f"T[min/max]={chain.maturities.min():.3f}/{chain.maturities.max():.3f}"
        )
    if days:
        path = loader.to_cboe_csv("SPY", days[0], f"SPY_{days[0].replace('-', '')}.csv")
        print(f"wrote CBOE-format CSV: {path}")
