"""
Loop-health eval — classify per-tick outcomes into alertable events.

Pure logic, no I/O, no network: feed it the parsed tick-log records (oldest→newest)
plus the prior notifier state, get back a list of HealthEvent + the updated state.
trading.notifier owns the file reads, Telegram transport, and state persistence.

Single source of truth is the tick log (data/tick_log.jsonl). Every tick writes one
record tagged with `status` (see trading.loop._emit_outcome); legacy records with no
`status` are treated as successful ("ok") ticks.

Severity tiers
  info  — routine, worth knowing: first run of day, entries, exits, resume.
  warn  — degraded but not halted: RMSE spike / gate-suppressed surface, Feller flip,
          uncertainty gate lost, intraday breaker, "signals cleared gate but 0 traded".
  crit  — action required, attaches the raw record: Popper kill, halt, fetch/calib error.

Dedup is via the returned state (last_tick, first_run_date, halt_active): only records
newer than last_tick emit, first-run fires once per date, halt/resume fire on transition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Optional

import config

# ── Tunables ───────────────────────────────────────────────────────────────────
# A spike must be both a big jump vs recent fit AND cross the fit-quality gate, so we
# don't cry wolf on normal RMSE wiggle while the fit is still healthy (< SIGNAL_MAX_RMSE).
RMSE_SPIKE_MULT = 2.5
RMSE_SPIKE_FLOOR = config.SIGNAL_MAX_RMSE        # 0.10
RMSE_HIST_WINDOW = 10                            # recent ok ticks for the median baseline

# "Signals cleared the uncertainty gate but nothing traded" — the gate-problem tell.
# Per the live-bound-by-uncertainty-gate finding, no-trade is the posterior gate; if
# signals clear it yet 0 enter, the downstream (cost gate / sizing) is the suspect.
GATE_PROBLEM_MIN_CLEARED = 1

_OK = "ok"
_SILENT_STATUSES = {"skip_closed"}               # market-closed no-ops never alert

CRIT_STATUSES = {"err_fetch", "err_calib", "kill_triggered", "halted"}
WARN_STATUSES = {"err_spot", "err_thin_chain"}


@dataclass
class HealthEvent:
    tier: str            # "info" | "warn" | "crit"
    kind: str            # machine label, e.g. "first_run", "buy", "rmse_spike"
    title: str           # one-line headline
    message: str         # full body for the notification
    tick: int = 0
    attach_record: bool = False   # crit: include the raw tick record in the message


@dataclass
class EvalResult:
    events: list = field(default_factory=list)
    state: dict = field(default_factory=dict)


def _status(rec: dict) -> str:
    return rec.get("status") or _OK


def _fmt_params(rec: dict) -> str:
    return (f"κ={rec.get('kappa')} θ={rec.get('theta')} ξ={rec.get('xi')} "
            f"ρ={rec.get('rho')} v0={rec.get('v0')}")


def _entry_lines(rec: dict, key: str) -> str:
    """Render per-contract detail if the loop logged it, else fall back to the count."""
    items = rec.get(key)
    n = rec.get("n_entered" if key == "entries" else "n_exited", 0)
    if isinstance(items, list) and items:
        lines = []
        for it in items:
            d = (it.get("direction") or "").upper()
            gap = it.get("vol_gap")
            gap_s = f" gap={gap:+.3f}" if isinstance(gap, (int, float)) else ""
            pnl = it.get("pnl")
            pnl_s = f" pnl={pnl:+.4f}" if isinstance(pnl, (int, float)) else ""
            reason = it.get("reason")
            reason_s = f" ({reason})" if reason else ""
            lines.append(f"  • {d} {it.get('qty', '?')}x {it.get('strike', '?')} "
                         f"exp={it.get('expiry', '?')}{gap_s}{pnl_s}{reason_s}")
        return "\n".join(lines)
    return f"  • {n} contract(s) (no per-contract detail logged)"


def evaluate(records: list, state: Optional[dict] = None) -> EvalResult:
    """Classify the tick log into alertable events.

    Args:
        records: parsed tick-log records, oldest→newest. May mix legacy (no `status`)
                 and new records; legacy are treated as status="ok".
        state:   prior notifier state {last_tick, first_run_date, halt_active}.
                 None/empty = first ever run (everything ≤ a synthetic baseline is skipped
                 so we don't replay the whole history as alerts).

    Returns:
        EvalResult(events, state). `state` is the new dedup state to persist.
    """
    state = dict(state or {})
    # On a cold start (no prior state) treat the entire existing log as already-seen,
    # so we alert only on ticks produced *after* the notifier is first wired in.
    if "last_tick" not in state:
        last_tick = max((r.get("tick", 0) for r in records), default=0)
        first_run_date = records[-1].get("date") if records else None
        halt_active = _status(records[-1]) in {"halted", "kill_triggered"} if records else False
        return EvalResult(events=[], state={
            "last_tick": last_tick,
            "first_run_date": first_run_date,
            "halt_active": halt_active,
        })

    last_tick = state.get("last_tick", 0)
    first_run_date = state.get("first_run_date")
    halt_active = bool(state.get("halt_active", False))

    events: list = []
    ok_rmse_hist: list = []        # rolling rmse of ok ticks seen so far (for spike baseline)
    prev_ok_feller: Optional[bool] = None

    new_last_tick = last_tick

    for rec in records:
        tick = rec.get("tick", 0)
        status = _status(rec)
        is_new = tick > last_tick
        date = rec.get("date")

        # ── rolling context (built from ALL records, old + new) ──────────────
        baseline = median(ok_rmse_hist[-RMSE_HIST_WINDOW:]) if ok_rmse_hist else None
        feller_before = prev_ok_feller
        if status == _OK and isinstance(rec.get("rmse"), (int, float)):
            ok_rmse_hist.append(float(rec["rmse"]))
        if status == _OK and "feller_ok" in rec:
            prev_ok_feller = bool(rec["feller_ok"])

        if not is_new:
            continue
        new_last_tick = max(new_last_tick, tick)

        # ── market closed: never alert; do not advance first_run/halt ────────
        if status in _SILENT_STATUSES:
            continue

        # ── first run of the day (first non-closed tick of a new date) ───────
        if date and date != first_run_date:
            first_run_date = date
            events.append(HealthEvent(
                tier="info", kind="first_run", tick=tick,
                title=f"First run — {date}",
                message=(f"🔔 First tick of {date} (tick {tick}). "
                         f"Loop is live.\n"
                         f"spot={rec.get('spot', '?')} status={status}"),
            ))

        # ── critical statuses ────────────────────────────────────────────────
        if status == "err_fetch":
            events.append(HealthEvent(
                tier="crit", kind="err_fetch", tick=tick, attach_record=True,
                title="Data/API fetch failed",
                message=f"🚨 Chain fetch error (tick {tick}):\n{rec.get('error', '?')}",
            ))
            continue
        if status == "err_calib":
            events.append(HealthEvent(
                tier="crit", kind="err_calib", tick=tick, attach_record=True,
                title="Calibration failed",
                message=f"🚨 Calibration error (tick {tick}):\n{rec.get('error', '?')}",
            ))
            continue
        if status == "kill_triggered":
            halt_active = True
            events.append(HealthEvent(
                tier="crit", kind="kill_triggered", tick=tick, attach_record=True,
                title="Popper kill triggered — strategy halted",
                message=(f"⛔ KILL TRIGGERED (tick {tick}). New entries halted.\n"
                         f"reason={rec.get('reason', '?')} rmse={rec.get('rmse', '?')}\n"
                         f"Reset with `python -m evals.kill reset`."),
            ))
            continue
        if status == "halted":
            # Kill flag was already set on entry. Alert once on the transition only.
            if not halt_active:
                halt_active = True
                events.append(HealthEvent(
                    tier="crit", kind="halted", tick=tick, attach_record=True,
                    title="Loop halted (kill flag set)",
                    message=(f"⛔ Loop is halted by the kill flag (tick {tick}). "
                             f"No calibration/orders until reset."),
                ))
            continue

        # ── warn statuses ────────────────────────────────────────────────────
        if status == "err_spot":
            events.append(HealthEvent(
                tier="warn", kind="err_spot", tick=tick,
                title="Spot unavailable",
                message=f"⚠️ Spot price unavailable (tick {tick}, spot={rec.get('spot', '?')}).",
            ))
            continue
        if status == "err_thin_chain":
            events.append(HealthEvent(
                tier="warn", kind="err_thin_chain", tick=tick,
                title="Thin options chain",
                message=(f"⚠️ Only {rec.get('n_clean', '?')} clean options "
                         f"(raw {rec.get('n_raw', '?')}) — tick skipped (tick {tick})."),
            ))
            continue

        # ── ok tick: resume, trades, then health warnings ────────────────────
        if halt_active:
            halt_active = False
            events.append(HealthEvent(
                tier="info", kind="resumed", tick=tick,
                title="Strategy resumed",
                message=f"✅ Loop resumed — first healthy tick after halt (tick {tick}).",
            ))

        n_entered = int(rec.get("n_entered", 0) or 0)
        n_exited = int(rec.get("n_exited", 0) or 0)
        if n_entered > 0:
            events.append(HealthEvent(
                tier="info", kind="buy", tick=tick,
                title=f"{n_entered} position(s) entered",
                message=(f"🟢 ENTERED {n_entered} position(s) (tick {tick}):\n"
                         f"{_entry_lines(rec, 'entries')}\n"
                         f"open={rec.get('open_positions', '?')} "
                         f"tick_pnl={rec.get('tick_pnl', 0)}"),
            ))
        if n_exited > 0:
            events.append(HealthEvent(
                tier="info", kind="sell", tick=tick,
                title=f"{n_exited} position(s) exited",
                message=(f"🔵 EXITED {n_exited} position(s) (tick {tick}):\n"
                         f"{_entry_lines(rec, 'exits')}\n"
                         f"open={rec.get('open_positions', '?')} "
                         f"tick_pnl={rec.get('tick_pnl', 0)} "
                         f"session_pnl={rec.get('session_pnl', 0)}"),
            ))

        # health warnings (only meaningful on a calibrated tick)
        rmse = rec.get("rmse")
        if rec.get("rmse_gate_suppressed"):
            events.append(HealthEvent(
                tier="warn", kind="rmse_gate_suppressed", tick=tick,
                title="Fit-quality gate suppressed the surface",
                message=(f"⚠️ RMSE {rmse} > SIGNAL_MAX_RMSE {config.SIGNAL_MAX_RMSE} "
                         f"(tick {tick}) — whole surface gated out, no signals. "
                         f"Fit is too poor to trust.\n{_fmt_params(rec)}"),
            ))
        elif (isinstance(rmse, (int, float)) and baseline is not None
              and rmse > RMSE_SPIKE_FLOOR and rmse > RMSE_SPIKE_MULT * baseline):
            events.append(HealthEvent(
                tier="warn", kind="rmse_spike", tick=tick,
                title="RMSE spike",
                message=(f"⚠️ RMSE jumped to {rmse} (recent median {baseline:.4f}, "
                         f"×{rmse / baseline:.1f}) at tick {tick}. Fit degrading.\n"
                         f"{_fmt_params(rec)}"),
            ))

        if feller_before is not None and "feller_ok" in rec:
            if bool(rec["feller_ok"]) != feller_before:
                now_ok = bool(rec["feller_ok"])
                events.append(HealthEvent(
                    tier="warn", kind="feller_flip", tick=tick,
                    title=f"Feller condition {'restored' if now_ok else 'violated'}",
                    message=(f"⚠️ Feller 2κθ≥ξ² flipped {feller_before}→{now_ok} "
                             f"(tick {tick}). Variance-path validity changed.\n"
                             f"{_fmt_params(rec)}"),
                ))

        if "laplace_ok" in rec and not rec["laplace_ok"]:
            events.append(HealthEvent(
                tier="warn", kind="laplace_failed", tick=tick,
                title="Uncertainty gate unavailable",
                message=(f"⚠️ Laplace posterior failed (tick {tick}) — trading WITHOUT "
                         f"the uncertainty gate this tick."),
            ))

        if rec.get("intraday_paused"):
            events.append(HealthEvent(
                tier="warn", kind="intraday_breaker", tick=tick,
                title="Intraday breaker active",
                message=(f"⚠️ Intraday breaker engaged (tick {tick}) — "
                         f"{config.POPPER_KILL_TICKS} consecutive RMSE>"
                         f"{config.CALIBRATION_FAIL_RMSE} ticks; new entries paused today."),
            ))

        # gate problem: signals cleared the uncertainty gate but nothing was entered.
        gd = rec.get("gap_diag")
        cleared = gd.get("n_cleared_uncertainty") if isinstance(gd, dict) else None
        if (isinstance(cleared, int) and cleared >= GATE_PROBLEM_MIN_CLEARED
                and n_entered == 0 and not rec.get("rmse_gate_suppressed")
                and not rec.get("intraday_paused")):
            ratio = gd.get("median_gap_over_hw")
            ratio_s = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "?"
            events.append(HealthEvent(
                tier="warn", kind="gate_problem", tick=tick,
                title="Signals cleared uncertainty gate but 0 traded",
                message=(f"⚠️ {cleared} signal(s) cleared the uncertainty gate yet 0 "
                         f"entered (tick {tick}). gap/halfwidth median={ratio_s}. "
                         f"Cost gate or sizing may be over-filtering."),
            ))

    new_state = {
        "last_tick": new_last_tick,
        "first_run_date": first_run_date,
        "halt_active": halt_active,
    }
    return EvalResult(events=events, state=new_state)
