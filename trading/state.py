"""
Live trading state — persists between loop iterations.

Stored in data/live_state.json. Loaded at startup, saved after every tick.

State contains:
  - open_positions: list of active vol-arb positions
  - current_hedge:  net shares of underlying held for delta hedge
  - prev_params:    last calibrated HestonParams (warm-start for next calibration)
  - last_tick_time: ISO timestamp of the last completed tick
  - session_pnl:    cumulative P&L since the state file was created

Thread safety: single-process, single-thread only. No locking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from calibration.heston import HestonParams


_DEFAULT_STATE_PATH = Path("data/live_state.json")


@dataclass
class LivePosition:
    """An open vol-arb position in the live portfolio."""
    ticker: str
    entry_date: str           # ISO date "YYYY-MM-DD"
    entry_time: str           # ISO timestamp of entry
    strike: float
    maturity: float           # years remaining at entry
    expiry: str               # "YYYY-MM-DD" — used for option contract routing
    direction: str            # "buy" or "sell"
    qty: int                  # number of contracts (Kelly-sized)
    entry_market_iv: float
    entry_model_iv: float
    entry_vol_gap: float
    entry_spot: float         # spot at entry (for vega/gamma reference)
    entry_premium: float = 0.0  # capital deployed at entry ($): premium paid (long) or
                                # estimated initial margin (short). Sums to the book's
                                # deployed capital for the MAX_PORTFOLIO_CAPITAL check.
    entry_fill_price: float = 0.0  # per-share fill price for MTM; set from the broker's
                                   # real avg_entry_price on reconcile (limit price until).
    right: str = "C"            # "C" or "P" — the OTM instrument actually traded (put
                                # when strike<forward, call otherwise). Legacy recs = "C".
    filled: bool = True         # True once the broker confirms the fill. Live entries set
                                # this False until reconciliation sees the position held;
                                # unfilled positions are not marked or hedged. (Legacy=True.)
    age_days: int = 0
    cumulative_pnl: float = 0.0
    option_order_id: Optional[str] = None   # broker order ID for entry
    exited: bool = False
    exit_date: Optional[str] = None
    exit_reason: Optional[str] = None
    exit_order_id: Optional[str] = None


@dataclass
class LiveState:
    """Full persistent state for the live trading loop."""
    open_positions: list[LivePosition] = field(default_factory=list)
    current_hedge: int = 0               # net shares of underlying (signed)
    prev_params: Optional[dict] = None   # HestonParams as dict for JSON
    last_tick_time: Optional[str] = None
    session_pnl: float = 0.0
    realized_pnl: float = 0.0            # cumulative P&L booked from closed positions
    n_ticks: int = 0
    consec_fail_ticks: int = 0                  # consecutive ticks with RMSE > fail threshold
    intraday_halt_date: Optional[str] = None    # date the intraday breaker paused new entries
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    # ── Param helpers ──────────────────────────────────────────────────

    def get_prev_params(self) -> Optional[HestonParams]:
        if self.prev_params is None:
            return None
        return HestonParams(**self.prev_params)

    def set_prev_params(self, p: HestonParams) -> None:
        self.prev_params = {
            "kappa": float(p.kappa),
            "theta": float(p.theta),
            "xi": float(p.xi),
            "rho": float(p.rho),
            "v0": float(p.v0),
        }

    # ── Position helpers ───────────────────────────────────────────────

    @property
    def active_positions(self) -> list[LivePosition]:
        return [p for p in self.open_positions if not p.exited]

    def position_key(self, p: LivePosition) -> tuple:
        return (p.ticker, round(p.strike, 1), p.expiry, p.direction)

    def has_position(self, ticker: str, strike: float, expiry: str, direction: str) -> bool:
        key = (ticker, round(strike, 1), expiry, direction)
        return any(self.position_key(p) == key for p in self.active_positions)

    def total_delta_exposure(self) -> float:
        """Sum of position deltas × qty (absolute, not net — for sizing reference)."""
        return sum(abs(p.qty) * p.entry_vol_gap for p in self.active_positions)

    # ── Persistence ────────────────────────────────────────────────────

    def save(self, path: Path = _DEFAULT_STATE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "open_positions": [asdict(p) for p in self.open_positions],
            "current_hedge": self.current_hedge,
            "prev_params": self.prev_params,
            "last_tick_time": self.last_tick_time,
            "session_pnl": self.session_pnl,
            "realized_pnl": self.realized_pnl,
            "n_ticks": self.n_ticks,
            "consec_fail_ticks": self.consec_fail_ticks,
            "intraday_halt_date": self.intraday_halt_date,
            "created_at": self.created_at,
        }
        with path.open("w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path = _DEFAULT_STATE_PATH) -> LiveState:
        if not path.exists():
            return cls()
        with path.open() as f:
            data = json.load(f)
        positions = [LivePosition(**p) for p in data.get("open_positions", [])]
        return cls(
            open_positions=positions,
            current_hedge=data.get("current_hedge", 0),
            prev_params=data.get("prev_params"),
            last_tick_time=data.get("last_tick_time"),
            session_pnl=data.get("session_pnl", 0.0),
            realized_pnl=data.get("realized_pnl", 0.0),
            n_ticks=data.get("n_ticks", 0),
            consec_fail_ticks=data.get("consec_fail_ticks", 0),
            intraday_halt_date=data.get("intraday_halt_date"),
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
        )

    @classmethod
    def reset(cls, path: Path = _DEFAULT_STATE_PATH) -> LiveState:
        """Create a fresh state, archiving the old one if it exists."""
        if path.exists():
            archive = path.with_suffix(f".{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.bak")
            path.rename(archive)
        state = cls()
        state.save(path)
        return state
