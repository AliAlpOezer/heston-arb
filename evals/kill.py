"""
Popper kill-condition runner.

Records daily calibration RMSE to a JSON log and halts the strategy after
POPPER_KILL_DAYS consecutive days above CALIBRATION_FAIL_RMSE.

Usage (daily cron):
    python -m evals.kill --ticker SPX --date 2024-03-15 --rmse 0.042

Usage (check current status):
    python -m evals.kill --status

The log is stored at: data/kill_log.json
Each entry: {"date": "YYYY-MM-DD", "ticker": str, "rmse": float, "killed": bool}

Killed state: once kill is triggered, the flag is written to data/kill_flag.json.
The live trading loop must check KillCondition.is_halted() before sizing any position.
"""

import json
import argparse
from dataclasses import dataclass, field, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import config

_LOG_PATH = Path(__file__).parent.parent / "data" / "kill_log.json"
_FLAG_PATH = Path(__file__).parent.parent / "data" / "kill_flag.json"


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class DailyRecord:
    date: str           # YYYY-MM-DD
    ticker: str
    rmse: float
    failed: bool        # rmse > CALIBRATION_FAIL_RMSE


@dataclass
class KillState:
    halted: bool
    triggered_on: Optional[str]     # date of 5th consecutive failure
    triggered_ticker: Optional[str]
    consecutive_failures: int
    reason: str


# ── Log I/O ───────────────────────────────────────────────────────────────────

def _load_log() -> list[DailyRecord]:
    if not _LOG_PATH.exists():
        return []
    with _LOG_PATH.open() as f:
        raw = json.load(f)
    return [DailyRecord(**r) for r in raw]


def _save_log(records: list[DailyRecord]) -> None:
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("w") as f:
        json.dump([asdict(r) for r in records], f, indent=2)


def _load_flag() -> KillState:
    if not _FLAG_PATH.exists():
        return KillState(
            halted=False,
            triggered_on=None,
            triggered_ticker=None,
            consecutive_failures=0,
            reason="",
        )
    with _FLAG_PATH.open() as f:
        return KillState(**json.load(f))


def _save_flag(state: KillState) -> None:
    _FLAG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _FLAG_PATH.open("w") as f:
        json.dump(asdict(state), f, indent=2)


# ── Kill condition logic ───────────────────────────────────────────────────────

class KillCondition:
    """
    Stateful kill-condition tracker. Wraps the JSON log file.

    Invariants:
    - Once halted=True, only reset() clears it (requires manual intervention).
    - A day's RMSE is aggregated as the WORST (max) value recorded for that date.
    - Consecutive-failure count resets to zero on the first passing day.
    """

    def __init__(
        self,
        log_path: Path = _LOG_PATH,
        flag_path: Path = _FLAG_PATH,
        kill_days: int = config.POPPER_KILL_DAYS,
        fail_rmse: float = config.CALIBRATION_FAIL_RMSE,
    ):
        self._log_path = log_path
        self._flag_path = flag_path
        self.kill_days = kill_days
        self.fail_rmse = fail_rmse

    # ── Public interface ───────────────────────────────────────────────────

    def record(self, ticker: str, snapshot_date: str, rmse: float) -> KillState:
        """Append a daily calibration result and return the updated kill state.

        Args:
            ticker: e.g. "SPX"
            snapshot_date: "YYYY-MM-DD"
            rmse: calibration RMSE (log-IV space)

        Returns:
            KillState with halted=True if kill condition triggered.
        """
        records = self._load()

        # Daily aggregate: keep the WORST (max) RMSE seen for this date+ticker. Under
        # an intraday loop (e.g. 5-min ticks) this means a day that calibrated badly on
        # most ticks but had one good final tick still counts as a failed day. (Was
        # last-write-wins, which let a single good tick mask a bad day from the kill.)
        existing = next(
            (r for r in records if r.date == snapshot_date and r.ticker == ticker), None
        )
        if existing is not None:
            existing.rmse = max(existing.rmse, rmse)
            existing.failed = existing.rmse > self.fail_rmse
        else:
            records.append(DailyRecord(
                date=snapshot_date, ticker=ticker, rmse=rmse,
                failed=rmse > self.fail_rmse,
            ))
        self._save(records)

        state = self._evaluate(records, ticker)
        self._save_flag(state)
        return state

    def is_halted(self) -> bool:
        """Returns True if the kill condition has been triggered."""
        return self._load_flag().halted

    def status(self) -> KillState:
        """Return the current kill state without recording anything."""
        return self._load_flag()

    def reset(self, reason: str = "") -> None:
        """Clear kill flag. Requires manual intervention — log the reason."""
        state = KillState(
            halted=False,
            triggered_on=None,
            triggered_ticker=None,
            consecutive_failures=0,
            reason=f"Reset: {reason}",
        )
        self._save_flag(state)
        print(f"[kill] Kill flag cleared. Reason: {reason}")

    def tail(self, n: int = 10) -> list[DailyRecord]:
        """Return the last n records."""
        return self._load()[-n:]

    # ── File I/O (uses instance paths, not module-level globals) ──────────

    def _load(self) -> list[DailyRecord]:
        if not self._log_path.exists():
            return []
        with self._log_path.open() as f:
            raw = json.load(f)
        return [DailyRecord(**r) for r in raw]

    def _save(self, records: list[DailyRecord]) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("w") as f:
            json.dump([asdict(r) for r in records], f, indent=2)

    def _load_flag(self) -> KillState:
        if not self._flag_path.exists():
            return KillState(halted=False, triggered_on=None, triggered_ticker=None,
                             consecutive_failures=0, reason="")
        with self._flag_path.open() as f:
            return KillState(**json.load(f))

    def _save_flag(self, state: KillState) -> None:
        self._flag_path.parent.mkdir(parents=True, exist_ok=True)
        with self._flag_path.open("w") as f:
            json.dump(asdict(state), f, indent=2)

    # ── Internal ──────────────────────────────────────────────────────────

    def _evaluate(self, records: list[DailyRecord], ticker: str) -> KillState:
        """Recompute kill state from the full log."""
        # Filter to this ticker, sorted by date
        ticker_records = sorted(
            [r for r in records if r.ticker == ticker],
            key=lambda r: r.date,
        )

        if not ticker_records:
            return KillState(halted=False, triggered_on=None, triggered_ticker=None,
                             consecutive_failures=0, reason="")

        # Count consecutive failures from the tail
        consecutive = 0
        for rec in reversed(ticker_records):
            if rec.failed:
                consecutive += 1
            else:
                break

        if consecutive >= self.kill_days:
            triggered_on = ticker_records[-1].date
            reason = (
                f"{consecutive} consecutive calibration failures for {ticker} "
                f"(RMSE > {self.fail_rmse:.3f}). Last date: {triggered_on}. "
                f"Manual review required before resuming."
            )
            return KillState(
                halted=True,
                triggered_on=triggered_on,
                triggered_ticker=ticker,
                consecutive_failures=consecutive,
                reason=reason,
            )

        return KillState(
            halted=False,
            triggered_on=None,
            triggered_ticker=None,
            consecutive_failures=consecutive,
            reason="",
        )


# ── Default singleton ─────────────────────────────────────────────────────────

_default = KillCondition()


def record(ticker: str, snapshot_date: str, rmse: float) -> KillState:
    """Record a daily calibration result (module-level convenience wrapper)."""
    return _default.record(ticker, snapshot_date, rmse)


def is_halted() -> bool:
    """Check if strategy is halted (module-level convenience wrapper)."""
    return _default.is_halted()


# ── CLI ───────────────────────────────────────────────────────────────────────

def _print_state(state: KillState) -> None:
    status = "[HALTED]" if state.halted else "[OK]"
    print(f"Status: {status}")
    print(f"  Consecutive failures: {state.consecutive_failures} / {config.POPPER_KILL_DAYS}")
    if state.halted:
        print(f"  Triggered on: {state.triggered_on} ({state.triggered_ticker})")
        print(f"  Reason: {state.reason}")


def _print_tail(kc: KillCondition, n: int = 10) -> None:
    records = kc.tail(n)
    if not records:
        print("  (no records)")
        return
    print(f"  {'Date':<12} {'Ticker':<8} {'RMSE':>8}  {'Status'}")
    print(f"  {'-'*12} {'-'*8} {'-'*8}  {'-'*8}")
    for r in records:
        mark = "FAIL" if r.failed else "pass"
        print(f"  {r.date:<12} {r.ticker:<8} {r.rmse:>8.4f}  {mark}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Popper kill-condition runner")
    sub = parser.add_subparsers(dest="cmd")

    rec = sub.add_parser("record", help="Record a daily calibration result")
    rec.add_argument("--ticker", required=True)
    rec.add_argument("--date", required=True, help="YYYY-MM-DD")
    rec.add_argument("--rmse", required=True, type=float)

    sub.add_parser("status", help="Print current kill state and recent log")

    rst = sub.add_parser("reset", help="Clear kill flag (manual intervention)")
    rst.add_argument("--reason", default="Manual reset", help="Reason for clearing")

    args = parser.parse_args()

    kc = KillCondition()

    if args.cmd == "record":
        state = kc.record(args.ticker, args.date, args.rmse)
        print(f"[kill] Recorded: {args.ticker} {args.date} RMSE={args.rmse:.4f} "
              f"{'FAIL' if args.rmse > config.CALIBRATION_FAIL_RMSE else 'pass'}")
        _print_state(state)
        if state.halted:
            print("\n*** STRATEGY HALTED — manual review required ***\n")
            raise SystemExit(1)

    elif args.cmd == "status":
        state = kc.status()
        _print_state(state)
        print("\nRecent log:")
        _print_tail(kc)

    elif args.cmd == "reset":
        kc.reset(args.reason)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
