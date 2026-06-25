"""
CBOE DataShop loader.

CBOE DataShop (https://datashop.cboe.com) delivers option snapshots as CSVs.
This module parses those files into OptionsChain objects for the Heston pipeline.

Supported products: SPX, SPXW, SPY, VIX, individual equities.
The column schema is consistent across products since ~2020; earlier files
may need the column_aliases dict extended.

Key departures from raw CBOE data:
  1. CBOE's own IV column is DROPPED. It uses a binomial tree with discrete
     dividends and differs from our BS IV by 0-5% on dividend-paying stocks.
     We re-derive IV from mid-prices via the cleaner's Newton-Raphson solver.
  2. IV == 0 rows are dropped (config.CBOE_DROP_ZERO_IV). CBOE sets IV=0 when
     the option is deep ITM, the model fails, or price > 850% vol threshold.
  3. r (risk-free rate) and q (dividend yield) are NOT in CBOE data. Pass them
     explicitly or use the rate_fetcher utilities below.

Typical workflow:
    loader = CBOELoader(data_dir="/path/to/datashop/spx/")
    chain  = loader.fetch("SPX", "2024-03-15", r=0.053, q=0.015)
    cleaned = clean_chain(chain)
    surface = build_surface(cleaned)
"""

import os
import re
import numpy as np
import pandas as pd
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Union

from data.cleaner import OptionsChain
import config


# ── Column name aliases (DataShop schema has changed slightly over time) ──────

_COLUMN_ALIASES = {
    # quote timestamp
    "quote_datetime":   ["quote_datetime", "quotedate", "quote_date", "datetime", "timestamp"],
    # underlying
    "root":             ["root", "symbol", "underlying", "underlying_symbol"],
    # option contract
    "expiration":       ["expiration", "expirdate", "expiry", "expiration_date"],
    "strike":           ["strike", "strike_price", "strikeprice"],
    "option_type":      ["option_type", "putcall", "cp_flag", "call_put", "type"],
    # prices
    "bid":              ["bid", "bid_price", "best_bid"],
    "ask":              ["ask", "ask_price", "best_ask"],
    "mid":              ["mid", "mid_price"],          # optional — computed if absent
    # volume / OI
    "open_interest":    ["open_interest", "openinterest", "oi"],
    "volume":           ["volume", "trade_volume", "vol"],
    # spot
    "underlying_bid":   ["underlying_bid", "stock_price", "spot", "underbid", "ulying_bid"],
    "underlying_ask":   ["underlying_ask", "underask",   "ulying_ask"],
    # CBOE's own IV (dropped after load)
    "implied_volatility": ["implied_volatility", "impliedvol", "iv", "ivol"],
}


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Return {canonical_name: actual_column_name} for all columns found in df."""
    cols_lower = {c.lower(): c for c in df.columns}
    resolved = {}
    for canonical, aliases in _COLUMN_ALIASES.items():
        for alias in aliases:
            if alias.lower() in cols_lower:
                resolved[canonical] = cols_lower[alias.lower()]
                break
    return resolved


# ── Date / time utilities ─────────────────────────────────────────────────────

def _parse_date(s) -> date:
    """Parse CBOE date strings: YYYY-MM-DD, MM/DD/YYYY, YYYYMMDD."""
    if isinstance(s, (date, datetime)):
        return s.date() if isinstance(s, datetime) else s
    s = str(s).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date: {s!r}")


def _ttm_years(quote_dt: date, expiry_dt: date) -> float:
    """ACT/365 time to maturity in years."""
    return max((expiry_dt - quote_dt).days, 0) / 365.0


# ── Spot price extraction ─────────────────────────────────────────────────────

def _extract_spot(df: pd.DataFrame, col_map: dict[str, str]) -> float:
    """Best available spot price from the chain snapshot."""
    # Prefer mid of underlying bid/ask
    if "underlying_bid" in col_map and "underlying_ask" in col_map:
        bids = pd.to_numeric(df[col_map["underlying_bid"]], errors="coerce")
        asks = pd.to_numeric(df[col_map["underlying_ask"]], errors="coerce")
        mid = ((bids + asks) / 2.0).dropna()
        if len(mid) > 0:
            return float(mid.median())

    if "underlying_bid" in col_map:
        vals = pd.to_numeric(df[col_map["underlying_bid"]], errors="coerce").dropna()
        if len(vals) > 0:
            return float(vals.median())

    raise ValueError(
        "Cannot extract spot price from CBOE data. "
        "Expected columns: underlying_bid, underlying_ask. "
        f"Got: {list(df.columns)}"
    )


# ── Core loader ───────────────────────────────────────────────────────────────

class CBOELoader:
    """Load CBOE DataShop option chain snapshots from CSV files.

    Args:
        data_dir: directory containing CBOE CSV files. Files are matched by
            ticker and date using common CBOE naming conventions:
              - {ROOT}_{YYYYMMDD}.csv
              - {YYYYMMDD}_{ROOT}.csv
              - {ROOT}_{YYYY-MM-DD}.csv
              - Any CSV in data_dir that contains ROOT and date in the filename.
            If data_dir is None, pass file_path directly to fetch().

        call_only: if True, return only call options (our calibration uses calls
            only; puts can be added via put-call parity in the cleaner). Default True.

        drop_zero_iv: drop rows where CBOE's own IV column equals 0 (these are
            deep ITM, model failures, or capped vols). Default: config.CBOE_DROP_ZERO_IV.

        min_open_interest: pre-filter before the cleaner's filter_chain(). Use 0
            to let the cleaner handle it; use a positive value for early reduction
            of very large files (SPX has 10k+ rows per snapshot).
    """

    def __init__(
        self,
        data_dir: Optional[Union[str, Path]] = None,
        call_only: bool = True,
        drop_zero_iv: bool = config.CBOE_DROP_ZERO_IV,
        min_open_interest: int = 0,
    ):
        self.data_dir = Path(data_dir) if data_dir else None
        self.call_only = call_only
        self.drop_zero_iv = drop_zero_iv
        self.min_open_interest = min_open_interest

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch(
        self,
        ticker: str,
        snapshot_date: Union[str, date],
        r: float,
        q: float,
        snapshot_time: str = "09:35:00",
        file_path: Optional[Union[str, Path]] = None,
    ) -> OptionsChain:
        """Load and parse one CBOE DataShop snapshot.

        Args:
            ticker: underlying root symbol (e.g. "SPX", "SPY").
            snapshot_date: trading date of the snapshot.
            r: risk-free rate (continuous, annualised). CBOE doesn't provide this —
               use the 3-month OIS or T-bill rate on the snapshot date.
            q: continuous dividend yield. For SPX: ~1.3-1.5%. For SPY: ~1.4%.
               Use a trailing 12-month dividend yield or options-implied q.
            snapshot_time: intraday time to label the snapshot (ISO HH:MM:SS).
               CBOE EOD files typically use 16:00:00.
            file_path: explicit file path override (skips data_dir search).

        Returns:
            OptionsChain ready for clean_chain().
        """
        snap_date = _parse_date(snapshot_date)
        snap_str = f"{snap_date.isoformat()}T{snapshot_time}Z"

        if file_path is not None:
            df = self._read_csv(Path(file_path))
        elif self.data_dir is not None:
            df = self._find_and_read(ticker, snap_date)
        else:
            raise ValueError("Provide either data_dir at construction or file_path to fetch().")

        return self._parse(df, ticker, snap_date, snap_str, r, q)

    def fetch_from_dataframe(
        self,
        df: pd.DataFrame,
        ticker: str,
        snapshot_date: Union[str, date],
        r: float,
        q: float,
        snapshot_time: str = "16:00:00",
    ) -> OptionsChain:
        """Parse an already-loaded DataFrame (useful when caller manages file I/O)."""
        snap_date = _parse_date(snapshot_date)
        snap_str = f"{snap_date.isoformat()}T{snapshot_time}Z"
        return self._parse(df, ticker, snap_date, snap_str, r, q)

    # ── File I/O ───────────────────────────────────────────────────────────────

    def _read_csv(self, path: Path) -> pd.DataFrame:
        """Read a CBOE CSV, handling common encoding and separator variants."""
        # Try comma first, then pipe (some DataShop exports use pipe)
        for sep in (",", "|", "\t"):
            try:
                df = pd.read_csv(
                    path, sep=sep, low_memory=False,
                    encoding="latin-1",    # CBOE files sometimes have non-ASCII
                )
                if df.shape[1] > 3:       # sanity check — real files have many cols
                    return df
            except Exception:
                continue
        raise ValueError(f"Cannot read CBOE file: {path}")

    def _find_and_read(self, ticker: str, snap_date: date) -> pd.DataFrame:
        """Search data_dir for a file matching ticker and date."""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"CBOE data directory not found: {self.data_dir}")

        date_variants = [
            snap_date.strftime("%Y%m%d"),
            snap_date.strftime("%Y-%m-%d"),
            snap_date.strftime("%m%d%Y"),
        ]
        ticker_upper = ticker.upper()

        candidates = []
        for path in self.data_dir.rglob("*.csv"):
            stem = path.stem.upper()
            has_ticker = ticker_upper in stem
            has_date = any(dv in stem for dv in date_variants)
            if has_ticker and has_date:
                candidates.append(path)

        if not candidates:
            # Fallback: look for any CSV in a date-named subdirectory
            for date_var in date_variants:
                sub = self.data_dir / date_var
                if sub.exists():
                    for path in sub.glob(f"*{ticker_upper}*.csv"):
                        candidates.append(path)

        if not candidates:
            raise FileNotFoundError(
                f"No CBOE file found for {ticker} on {snap_date} in {self.data_dir}. "
                f"Expected naming: {{ROOT}}_{{YYYYMMDD}}.csv or similar."
            )

        # Prefer exact match; use first if multiple
        candidates.sort(key=lambda p: len(p.stem))
        return self._read_csv(candidates[0])

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse(
        self,
        df: pd.DataFrame,
        ticker: str,
        snap_date: date,
        snap_str: str,
        r: float,
        q: float,
    ) -> OptionsChain:
        col = _resolve_columns(df)
        _require(col, ["expiration", "strike", "option_type", "bid", "ask"])

        # ── Option type filter ──────────────────────────────────────────────
        opt_type_col = col["option_type"]
        df[opt_type_col] = df[opt_type_col].astype(str).str.strip().str.upper()

        if self.call_only:
            df = df[df[opt_type_col].isin(["C", "CALL"])].copy()
            if len(df) == 0:
                raise ValueError(
                    f"No call options found after filtering. "
                    f"option_type values seen: {df[opt_type_col].unique()[:10]}"
                )

        # ── Spot price ──────────────────────────────────────────────────────
        spot = _extract_spot(df, col)

        # ── Numeric columns ─────────────────────────────────────────────────
        bid  = pd.to_numeric(df[col["bid"]], errors="coerce")
        ask  = pd.to_numeric(df[col["ask"]], errors="coerce")

        if "mid" in col:
            mid = pd.to_numeric(df[col["mid"]], errors="coerce")
        else:
            mid = (bid + ask) / 2.0

        strike = pd.to_numeric(df[col["strike"]], errors="coerce")

        if "open_interest" in col:
            oi = pd.to_numeric(df[col["open_interest"]], errors="coerce").fillna(0)
        else:
            # No OI column — use volume as proxy; fall back to ones
            if "volume" in col:
                oi = pd.to_numeric(df[col["volume"]], errors="coerce").fillna(0)
            else:
                oi = pd.Series(np.ones(len(df)), index=df.index)

        # ── Expiry → TTM ────────────────────────────────────────────────────
        expiry_raw = df[col["expiration"]]
        expiry_dates = expiry_raw.apply(_parse_date)
        ttm = expiry_dates.apply(lambda e: _ttm_years(snap_date, e))

        # ── CBOE IV filter ──────────────────────────────────────────────────
        if self.drop_zero_iv and "implied_volatility" in col:
            cboe_iv = pd.to_numeric(df[col["implied_volatility"]], errors="coerce")
            keep_iv = (cboe_iv.isna()) | (cboe_iv > 0)
        else:
            keep_iv = pd.Series(True, index=df.index)

        # ── Early OI filter (optional, for large files) ─────────────────────
        if self.min_open_interest > 0:
            keep_oi = oi >= self.min_open_interest
        else:
            keep_oi = pd.Series(True, index=df.index)

        # ── Validity mask ───────────────────────────────────────────────────
        valid = (
            bid.notna() & ask.notna() & mid.notna() &
            strike.notna() & (strike > 0) &
            ttm.notna() & (ttm > 0) &
            (mid > 0) & (bid >= 0) & (ask >= bid) &
            keep_iv & keep_oi
        )

        if valid.sum() == 0:
            raise ValueError(
                f"No valid options after CBOE parsing for {ticker} on {snap_date}. "
                f"DataFrame shape: {df.shape}. Check column mapping: {col}"
            )

        # ── Assemble OptionsChain ───────────────────────────────────────────
        opt_types = df[opt_type_col][valid].map(
            lambda x: "C" if x in ("C", "CALL") else "P"
        )

        return OptionsChain(
            ticker=ticker,
            snapshot_time=snap_str,
            spot=spot,
            r=r,
            q=q,
            strikes=strike[valid].to_numpy(dtype=np.float64),
            maturities=ttm[valid].to_numpy(dtype=np.float64),
            mid_prices=mid[valid].to_numpy(dtype=np.float64),
            bid_prices=bid[valid].to_numpy(dtype=np.float64),
            ask_prices=ask[valid].to_numpy(dtype=np.float64),
            open_interest=oi[valid].to_numpy(dtype=np.float64),
            option_type=opt_types.to_numpy(),
        )


def _require(col_map: dict, required: list[str]):
    missing = [c for c in required if c not in col_map]
    if missing:
        raise ValueError(
            f"Required CBOE columns not found: {missing}. "
            f"Resolved columns: {col_map}. "
            f"Extend _COLUMN_ALIASES if your DataShop export uses different names."
        )


# ── Rate helpers ──────────────────────────────────────────────────────────────

def spx_rates(snapshot_date: Union[str, date]) -> tuple[float, float]:
    """Approximate r and q for SPX on a given date.

    These are reasonable defaults — override with real market rates for
    production. r ≈ 3-month OIS rate; q ≈ SPX dividend yield.

    For serious backtesting: fetch r from FRED (DGS3MO series) and q from
    the CBOE implied dividend yield or S&P 500 trailing yield.
    """
    d = _parse_date(snapshot_date)
    year = d.year

    # Rough US risk-free rate by period (3-month OIS proxy)
    if year >= 2024:
        r = 0.053   # ~5.3% (post-hike plateau)
    elif year >= 2023:
        r = 0.045   # ~4.5%
    elif year >= 2022:
        r = 0.025   # rising cycle
    elif year >= 2021:
        r = 0.005   # near-zero
    elif year >= 2020:
        r = 0.005
    elif year >= 2019:
        r = 0.024
    else:
        r = 0.020   # pre-2019 default

    q = 0.015   # SPX trailing dividend yield (rough — 1.3-1.7% historically)
    return r, q


def spy_rates(snapshot_date: Union[str, date]) -> tuple[float, float]:
    """Approximate r and q for SPY. q slightly higher than SPX (~1.5%)."""
    r, _ = spx_rates(snapshot_date)
    q = 0.016
    return r, q
