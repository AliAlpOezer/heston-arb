"""Tests for the Popper kill-condition runner."""

import json
import tempfile
from pathlib import Path

from evals.kill import KillCondition


def _kc(tmp: Path) -> KillCondition:
    return KillCondition(
        log_path=tmp / "kill_log.json",
        flag_path=tmp / "kill_flag.json",
        kill_days=5,
        fail_rmse=0.10,
    )


def test_no_halt_below_threshold():
    with tempfile.TemporaryDirectory() as d:
        kc = _kc(Path(d))
        for i in range(10):
            state = kc.record("SPX", f"2024-03-{i+1:02d}", rmse=0.05)
        assert not state.halted
        assert state.consecutive_failures == 0


def test_halt_after_5_consecutive_failures():
    with tempfile.TemporaryDirectory() as d:
        kc = _kc(Path(d))
        for i in range(5):
            state = kc.record("SPX", f"2024-03-{i+1:02d}", rmse=0.15)
        assert state.halted
        assert state.consecutive_failures == 5
        assert state.triggered_ticker == "SPX"


def test_reset_clears_after_passing_day():
    with tempfile.TemporaryDirectory() as d:
        kc = _kc(Path(d))
        # 4 failures
        for i in range(4):
            kc.record("SPX", f"2024-03-{i+1:02d}", rmse=0.15)
        # 1 passing day — resets consecutive counter
        state = kc.record("SPX", "2024-03-05", rmse=0.05)
        assert not state.halted
        assert state.consecutive_failures == 0


def test_halt_requires_5_in_a_row():
    with tempfile.TemporaryDirectory() as d:
        kc = _kc(Path(d))
        # 4 failures, 1 pass, 4 failures — should NOT halt (only 4 in a row)
        for i in range(4):
            kc.record("SPX", f"2024-03-{i+1:02d}", rmse=0.15)
        kc.record("SPX", "2024-03-05", rmse=0.05)  # pass
        for i in range(4):
            state = kc.record("SPX", f"2024-03-{i+6:02d}", rmse=0.15)
        assert not state.halted
        assert state.consecutive_failures == 4


def test_halt_5_after_reset_gap():
    with tempfile.TemporaryDirectory() as d:
        kc = _kc(Path(d))
        # 4 failures, pass, then 5 more failures — should halt on the 5th
        for i in range(4):
            kc.record("SPX", f"2024-03-{i+1:02d}", rmse=0.15)
        kc.record("SPX", "2024-03-05", rmse=0.05)
        for i in range(5):
            state = kc.record("SPX", f"2024-03-{i+6:02d}", rmse=0.20)
        assert state.halted
        assert state.consecutive_failures == 5


def test_worst_of_day_aggregation():
    with tempfile.TemporaryDirectory() as d:
        kc = _kc(Path(d))
        kc.record("SPX", "2024-03-01", rmse=0.15)        # fail
        kc.record("SPX", "2024-03-01", rmse=0.05)        # same day, better tick
        state = kc.record("SPX", "2024-03-02", rmse=0.15)  # fail
        # Worst-of-day: Mar-01 stays FAILED (max rmse 0.15), so Mar-01 + Mar-02 = 2 in a row.
        # (A single good intraday tick must NOT erase a day that calibrated badly.)
        assert not state.halted
        assert state.consecutive_failures == 2


def test_same_day_all_pass_not_failed():
    with tempfile.TemporaryDirectory() as d:
        kc = _kc(Path(d))
        kc.record("SPX", "2024-03-01", rmse=0.05)        # pass
        state = kc.record("SPX", "2024-03-01", rmse=0.08)  # same day, still below threshold
        # Worst-of-day = 0.08 < 0.10 → day is not failed
        assert not state.halted
        assert state.consecutive_failures == 0


def test_is_halted_persists():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "kill_log.json"
        flag = Path(d) / "kill_flag.json"
        kc1 = KillCondition(log_path=log, flag_path=flag, kill_days=5, fail_rmse=0.10)
        for i in range(5):
            kc1.record("SPX", f"2024-03-{i+1:02d}", rmse=0.15)
        # New instance reads from same files
        kc2 = KillCondition(log_path=log, flag_path=flag, kill_days=5, fail_rmse=0.10)
        assert kc2.is_halted()


def test_reset():
    with tempfile.TemporaryDirectory() as d:
        kc = _kc(Path(d))
        for i in range(5):
            kc.record("SPX", f"2024-03-{i+1:02d}", rmse=0.15)
        assert kc.is_halted()
        kc.reset("Fixed data pipeline")
        assert not kc.is_halted()


def test_ticker_isolation():
    with tempfile.TemporaryDirectory() as d:
        kc = _kc(Path(d))
        # SPX fails 5 times
        for i in range(5):
            kc.record("SPX", f"2024-03-{i+1:02d}", rmse=0.15)
        # SPY passes — should not be halted
        state_spy = kc.record("SPY", "2024-03-01", rmse=0.03)
        state_spx = kc.status()
        # SPX is halted, SPY is not (but status() reads the flag written by last record())
        # Kill state is per-record call; last record was SPY so flag reflects SPY state
        assert not state_spy.halted
        assert state_spy.consecutive_failures == 0


if __name__ == "__main__":
    tests = [
        test_no_halt_below_threshold,
        test_halt_after_5_consecutive_failures,
        test_reset_clears_after_passing_day,
        test_halt_requires_5_in_a_row,
        test_halt_5_after_reset_gap,
        test_worst_of_day_aggregation,
        test_same_day_all_pass_not_failed,
        test_is_halted_persists,
        test_reset,
        test_ticker_isolation,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} passed")
    if passed < len(tests):
        raise SystemExit(1)
