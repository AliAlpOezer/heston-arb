"""Unit tests for evals.health — one per outcome class, no network/I/O.

Run: pytest evals/test_health.py -q
"""

import config
from evals.health import evaluate

# A fresh "seen" baseline so records with tick > N are treated as new.
SEEN = {"last_tick": 0, "first_run_date": None, "halt_active": False}


def _kinds(records, state=SEEN):
    res = evaluate(records, state)
    return [e.kind for e in res.events], res.state


def _ok(tick, date="2026-06-25", **kw):
    rec = {"tick": tick, "date": date, "status": "ok", "spot": 745.0,
           "rmse": 0.03, "feller_ok": True, "laplace_ok": True,
           "intraday_paused": False, "n_entered": 0, "n_exited": 0,
           "open_positions": 0, "tick_pnl": 0.0, "kappa": 4.0, "theta": 0.05,
           "xi": 0.6, "rho": -0.9, "v0": 0.02}
    rec.update(kw)
    return rec


# ── cold start / dedup ──────────────────────────────────────────────────────────

def test_cold_start_emits_nothing_and_baselines():
    recs = [_ok(1), _ok(2)]
    res = evaluate(recs, None)              # no prior state
    assert res.events == []
    assert res.state["last_tick"] == 2      # everything marked seen

def test_old_records_do_not_reemit():
    recs = [_ok(1, n_entered=1, entries=[{"direction": "buy", "qty": 1, "strike": 745,
            "expiry": "2026-07-25", "vol_gap": 0.02}])]
    kinds, _ = _kinds(recs, {"last_tick": 1, "first_run_date": "2026-06-25",
                             "halt_active": False})
    assert kinds == []                       # tick 1 not > last_tick 1


# ── silent + first run ──────────────────────────────────────────────────────────

def test_market_closed_is_silent():
    kinds, st = _kinds([{"tick": 1, "date": "2026-06-25", "status": "skip_closed"}])
    assert kinds == []
    assert st["first_run_date"] is None      # closed ticks don't claim the day

def test_first_run_fires_once_per_date():
    recs = [{"tick": 1, "date": "2026-06-25", "status": "skip_closed"},
            _ok(2), _ok(3)]                  # 2 is first non-closed
    kinds, _ = _kinds(recs)
    assert kinds.count("first_run") == 1
    assert kinds[0] == "first_run"

def test_first_run_each_new_day():
    recs = [_ok(1, date="2026-06-24"), _ok(2, date="2026-06-25")]
    kinds, _ = _kinds(recs)
    assert kinds.count("first_run") == 2


# ── trades ───────────────────────────────────────────────────────────────────────

def test_buy_and_sell_events():
    recs = [_ok(1, n_entered=1, entries=[{"direction": "buy", "qty": 2, "strike": 745,
                "expiry": "2026-07-25", "vol_gap": 0.03}]),
            _ok(2, n_exited=1, session_pnl=5.0, exits=[{"direction": "buy", "qty": 2,
                "strike": 745, "expiry": "2026-07-25", "reason": "gap_closed", "pnl": 5.0}])]
    kinds, _ = _kinds(recs)
    assert "buy" in kinds and "sell" in kinds


# ── critical paths ────────────────────────────────────────────────────────────────

def test_fetch_error_is_crit_with_record():
    res = evaluate([{"tick": 1, "date": "2026-06-25", "status": "err_fetch",
                     "error": "boom"}], SEEN)
    ev = [e for e in res.events if e.kind == "err_fetch"][0]
    assert ev.tier == "crit" and ev.attach_record

def test_calib_error_is_crit():
    kinds, _ = _kinds([{"tick": 1, "date": "2026-06-25", "status": "err_calib",
                        "error": "nan"}])
    assert "err_calib" in kinds

def test_kill_then_resume():
    recs = [{"tick": 1, "date": "2026-06-25", "status": "kill_triggered",
             "reason": "5d", "rmse": 0.2}, _ok(2)]
    kinds, st = _kinds(recs)
    assert "kill_triggered" in kinds and "resumed" in kinds
    assert st["halt_active"] is False

def test_halt_dedup_until_resume():
    # already halted: first 'halted' alerts, second does not.
    recs = [{"tick": 1, "date": "2026-06-25", "status": "halted"},
            {"tick": 2, "date": "2026-06-25", "status": "halted"}]
    kinds, st = _kinds(recs)
    assert kinds.count("halted") == 1 and st["halt_active"] is True


# ── warn paths ──────────────────────────────────────────────────────────────────

def test_rmse_spike_needs_floor_and_jump():
    # baseline ~0.03; a jump to 0.04 stays below SIGNAL_MAX_RMSE floor -> no spike.
    recs = [_ok(i, rmse=0.03) for i in range(1, 6)] + [_ok(6, rmse=0.04)]
    kinds, _ = _kinds(recs)
    assert "rmse_spike" not in kinds
    # a jump above the floor and >2.5x baseline -> spike.
    recs2 = [_ok(i, rmse=0.03) for i in range(1, 6)] + [_ok(6, rmse=0.19)]
    kinds2, _ = _kinds(recs2)
    assert "rmse_spike" in kinds2

def test_gate_suppressed_warn():
    kinds, _ = _kinds([_ok(1, rmse=0.15, rmse_gate_suppressed=True)])
    assert "rmse_gate_suppressed" in kinds

def test_feller_flip():
    recs = [_ok(1, feller_ok=True), _ok(2, feller_ok=False)]
    kinds, _ = _kinds(recs)
    assert "feller_flip" in kinds

def test_laplace_failed():
    kinds, _ = _kinds([_ok(1, laplace_ok=False)])
    assert "laplace_failed" in kinds

def test_intraday_breaker():
    kinds, _ = _kinds([_ok(1, intraday_paused=True)])
    assert "intraday_breaker" in kinds

def test_gate_problem_signals_cleared_but_zero_traded():
    rec = _ok(1, n_entered=0, n_signals_raw=10,
              gap_diag={"n_cleared_uncertainty": 4, "median_gap_over_hw": 1.8})
    kinds, _ = _kinds([rec])
    assert "gate_problem" in kinds

def test_no_gate_problem_when_traded():
    rec = _ok(1, n_entered=2, n_signals_raw=10,
              gap_diag={"n_cleared_uncertainty": 4, "median_gap_over_hw": 1.8},
              entries=[{"direction": "buy", "qty": 1, "strike": 745,
                        "expiry": "2026-07-25", "vol_gap": 0.03}])
    kinds, _ = _kinds([rec])
    assert "gate_problem" not in kinds
