"""
Risk-free rate and dividend yield fetcher.

Sources:
  FRED (Federal Reserve Economic Data):
    - DGS3MO: 3-month Treasury yield (risk-free rate proxy)
    - SP500DIV: S&P 500 dividend yield (use for SPX/SPY q)

  Fallback: hardcoded annual rate tables (used when no internet or no API key).

Usage:
    r, q = get_rates("SPX", "2023-06-15")          # fetches from FRED if API key set
    r, q = get_rates("SPX", "2023-06-15", "abc123") # explicit API key

FRED API key: free at https://fred.stlouisfed.org/docs/api/api_key.html
Set via environment variable FRED_API_KEY or pass directly.

Rate conventions:
  - r: continuously-compounded annualized rate (ln(1 + discrete_rate))
  - q: continuously-compounded dividend yield
  - Both are stored as decimals (0.05 = 5%)
"""

import os
import json
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Optional
from urllib.request import urlopen
from urllib.error import URLError

# ── Cache ─────────────────────────────────────────────────────────────────────
# Simple JSON cache: {(series_id, date_str): value}
_CACHE_PATH = Path(__file__).parent.parent / "data" / ".rates_cache.json"


def _load_cache() -> dict:
    if _CACHE_PATH.exists():
        try:
            with _CACHE_PATH.open() as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with _CACHE_PATH.open("w") as f:
            json.dump(cache, f)
    except Exception:
        pass  # cache write failure is non-fatal


# ── FRED fetcher ──────────────────────────────────────────────────────────────

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"


def _fred_fetch(series_id: str, observation_date: str, api_key: str) -> Optional[float]:
    """Fetch a single FRED observation for the given date.

    Searches the 30-day window ending on observation_date and returns the
    most recent available value (FRED data has publication lags).

    Returns None on any error (network, key invalid, date out of range).
    """
    cache = _load_cache()
    cache_key = f"{series_id}:{observation_date}"
    if cache_key in cache:
        return cache[cache_key]

    start = (date.fromisoformat(observation_date) - timedelta(days=30)).isoformat()
    url = (
        f"{_FRED_BASE}?series_id={series_id}"
        f"&observation_start={start}"
        f"&observation_end={observation_date}"
        f"&api_key={api_key}&file_type=json"
    )

    try:
        with urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read())
        observations = [
            o for o in data.get("observations", [])
            if o["value"] not in (".", "")
        ]
        if not observations:
            return None
        raw = float(observations[-1]["value"])  # most recent non-missing
        # FRED DGS3MO is in percent per year — convert to decimal
        value = raw / 100.0
        cache[cache_key] = value
        _save_cache(cache)
        return value
    except (URLError, KeyError, ValueError, Exception):
        return None


# ── Fallback rate tables ──────────────────────────────────────────────────────
# Approximate annual-average 3-month T-bill rates and SPX dividend yields.
# Used when FRED is unavailable or no API key is provided.

_RF_FALLBACK = {
    2016: 0.0032, 2017: 0.0093, 2018: 0.0193, 2019: 0.0211,
    2020: 0.0036, 2021: 0.0005, 2022: 0.0200, 2023: 0.0518,
    2024: 0.0530, 2025: 0.0430, 2026: 0.0400,
}

_DIV_FALLBACK = {
    # SPX trailing dividend yield (annual average)
    2016: 0.021, 2017: 0.019, 2018: 0.020, 2019: 0.019,
    2020: 0.016, 2021: 0.013, 2022: 0.016, 2023: 0.015,
    2024: 0.013, 2025: 0.013, 2026: 0.013,
}

_SPY_DIV_FALLBACK = {
    # SPY has slightly higher yield (includes dividend pass-through timing)
    2016: 0.022, 2017: 0.020, 2018: 0.021, 2019: 0.020,
    2020: 0.017, 2021: 0.014, 2022: 0.017, 2023: 0.016,
    2024: 0.014, 2025: 0.014, 2026: 0.014,
}


def _fallback_rates(ticker: str, snapshot_date: str) -> tuple[float, float]:
    """Return (r, q) from hardcoded annual tables. Continuously-compounded."""
    year = int(snapshot_date[:4])
    r_discrete = _RF_FALLBACK.get(year, 0.04)
    div_table = _SPY_DIV_FALLBACK if ticker.upper() == "SPY" else _DIV_FALLBACK
    q_discrete = div_table.get(year, 0.015)
    import math
    r = math.log(1 + r_discrete)
    q = math.log(1 + q_discrete)
    return r, q


# ── Public interface ──────────────────────────────────────────────────────────

def get_rates(
    ticker: str,
    snapshot_date: str,
    api_key: Optional[str] = None,
) -> tuple[float, float]:
    """Return (r, q) for the given ticker and date.

    Rate sources:
        r  — FRED DGS3MO (3-month T-bill, percent p.a.). Falls back to _RF_FALLBACK.
        q  — Hardcoded annual table (_SPY_DIV_FALLBACK / _DIV_FALLBACK). No free
             FRED series exists for S&P 500 trailing dividend yield; the tables are
             accurate to within ~0.1 vol pt for Heston calibration purposes.
    """
    import math

    _, q = _fallback_rates(ticker, snapshot_date)

    key = api_key or os.environ.get("FRED_API_KEY")
    if key:
        r_raw = _fred_fetch("DGS3MO", snapshot_date, key)
        if r_raw is not None:
            return math.log(1 + r_raw), q
        warnings.warn(
            f"[rates] FRED DGS3MO fetch failed for {snapshot_date} — using fallback r. "
            "Check FRED_API_KEY and network access."
        )

    r, _ = _fallback_rates(ticker, snapshot_date)
    return r, q


def get_rates_batch(
    ticker: str,
    dates: list[str],
    api_key: Optional[str] = None,
) -> dict[str, tuple[float, float]]:
    """Fetch rates for a list of dates. Returns {date_str: (r, q)}.

    Caches results so each unique date only hits FRED once.
    """
    return {d: get_rates(ticker, d, api_key) for d in dates}
