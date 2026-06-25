"""
Telegram notifier — turns loop-health events into push messages.

Pipeline: read the tick log (single source of truth) + prior notifier state →
evals.health.evaluate() → send each event to Telegram → persist the new state.
Runs as a step in the live-tick workflow (one invocation per tick).

State (data/notifier_state.json) lives on the same branch as the tick log and is
committed alongside it, so dedup survives across CI runs:
  last_tick       — only records newer than this emit (no replaying history)
  first_run_date  — first-run-of-day fires once per date
  halt_active     — kill/halt fires on transition; resume fires when it clears
  warn_throttle   — {kind: last_tick}: a persistent warn alerts once per WARN_THROTTLE
                    window, not every 5 minutes

Credentials come from env (never committed):
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

CLI:
  python -m trading.notifier --tick-log data/tick_log.jsonl --state data/notifier_state.json
  python -m trading.notifier --get-chat-id        # print chat ids from getUpdates
  python -m trading.notifier --test               # send a connectivity test message
  python -m trading.notifier ... --dry-run        # classify + print, send nothing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from evals.health import evaluate, HealthEvent

# A persistent fault (e.g. fit-quality gate suppressed) recurs every tick. Alert once
# per this many ticks (~1h at the 5-min cadence) instead of spamming. INFO/CRIT never throttle.
WARN_THROTTLE_TICKS = 12

_API = "https://api.telegram.org/bot{token}/{method}"


def _out(s: str) -> None:
    """Print that survives a non-UTF-8 console (Windows cp1252) — emojis become '?'."""
    try:
        print(s)
    except UnicodeEncodeError:
        sys.stdout.buffer.write((s + "\n").encode("utf-8", "replace"))


# ── Telegram transport ──────────────────────────────────────────────────────────

def _call(token: str, method: str, params: dict) -> dict:
    url = _API.format(token=token, method=method)
    data = urllib.parse.urlencode(params).encode()
    req = urllib.request.Request(url, data=data)
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())


def send_message(token: str, chat_id: str, text: str) -> dict:
    # Telegram hard-caps a message at 4096 chars; truncate defensively.
    if len(text) > 4000:
        text = text[:3990] + "\n…(truncated)"
    return _call(token, "sendMessage", {"chat_id": chat_id, "text": text})


def get_chat_ids(token: str) -> list:
    """Return [(chat_id, name, last_text)] from pending getUpdates (for setup)."""
    res = _call(token, "getUpdates", {})
    out = []
    for u in res.get("result", []):
        msg = u.get("message") or u.get("channel_post") or {}
        chat = msg.get("chat", {})
        if chat:
            name = chat.get("username") or chat.get("title") or chat.get("first_name", "")
            out.append((chat.get("id"), name, msg.get("text", "")))
    return out


# ── State + records I/O ───────────────────────────────────────────────────────

def _load_records(path: Path) -> list:
    if not path.exists():
        return []
    records = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue   # tolerate a partially-written trailing line
    return records


def _load_state(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))


# ── Main run ──────────────────────────────────────────────────────────────────

def _format(event: HealthEvent, record: Optional[dict]) -> str:
    text = event.message
    if event.attach_record and record is not None:
        text += "\n\nraw tick:\n" + json.dumps(record, indent=2)
    return text


def run(tick_log: Path, state_path: Path, token: Optional[str], chat_id: Optional[str],
        dry_run: bool = False) -> int:
    """Classify new tick-log records and push events. Returns number of messages sent."""
    records = _load_records(tick_log)
    prior = _load_state(state_path)
    result = evaluate(records, prior)
    new_state = result.state

    # carry the warn-throttle ledger across runs (health is agnostic to transport)
    throttle = dict((prior or {}).get("warn_throttle", {}))
    by_tick = {r.get("tick"): r for r in records}

    sent = 0
    for ev in result.events:
        if ev.tier == "warn":
            last = throttle.get(ev.kind)
            if last is not None and ev.tick - last < WARN_THROTTLE_TICKS:
                continue   # same fault, still inside the throttle window
            throttle[ev.kind] = ev.tick

        text = _format(ev, by_tick.get(ev.tick))
        if dry_run or not (token and chat_id):
            _out(f"[notifier] {ev.tier.upper():4} {ev.kind} (t{ev.tick})\n{text}\n")
        else:
            try:
                resp = send_message(token, chat_id, text)
                if not resp.get("ok"):
                    print(f"[notifier] Telegram rejected {ev.kind}: {resp}", file=sys.stderr)
                    continue
            except Exception as e:   # never let a notify failure break the CI job
                print(f"[notifier] send failed for {ev.kind}: {e}", file=sys.stderr)
                continue
        sent += 1

    new_state["warn_throttle"] = throttle
    _save_state(state_path, new_state)
    return sent


def main() -> None:
    p = argparse.ArgumentParser(description="Heston-arb Telegram health notifier")
    p.add_argument("--tick-log", default="data/tick_log.jsonl")
    p.add_argument("--state", default="data/notifier_state.json")
    p.add_argument("--token", default=os.environ.get("TELEGRAM_BOT_TOKEN"))
    p.add_argument("--chat-id", default=os.environ.get("TELEGRAM_CHAT_ID"))
    p.add_argument("--dry-run", action="store_true",
                   help="Classify and print events; send nothing.")
    p.add_argument("--get-chat-id", action="store_true",
                   help="Print chat ids from getUpdates (message the bot first), then exit.")
    p.add_argument("--test", action="store_true",
                   help="Send a connectivity test message and exit.")
    args = p.parse_args()

    if args.get_chat_id:
        if not args.token:
            sys.exit("TELEGRAM_BOT_TOKEN not set.")
        ids = get_chat_ids(args.token)
        if not ids:
            print("No updates. Send a message to the bot, then retry.")
        for cid, name, text in ids:
            print(f"chat_id={cid}  name={name}  last_msg={text!r}")
        return

    if args.test:
        if not (args.token and args.chat_id):
            sys.exit("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID required.")
        resp = send_message(args.token, args.chat_id,
                            "✅ heston-arb notifier connected.")
        print("ok" if resp.get("ok") else f"failed: {resp}")
        return

    sent = run(Path(args.tick_log), Path(args.state), args.token, args.chat_id,
               dry_run=args.dry_run)
    print(f"[notifier] {sent} message(s) sent.")


if __name__ == "__main__":
    main()
