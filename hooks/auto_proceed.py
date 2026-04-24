#!/usr/bin/env python3
"""Stop hook: auto-feed "proceed" when a plan has unchecked items.

Activation is read from ~/.claude/mcp-skill-hub/config.json:
    {"auto_proceed": true, "auto_proceed_max": 20}

(Env vars SKILL_HUB_AUTO_PROCEED / SKILL_HUB_MAX_PROCEEDS still override.)

State: ~/.claude/mcp-skill-hub/state/auto_proceed.json tracks per-session
counter so runaway loops are capped (default 20).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdict_cache  # noqa: E402

LOG = Path.home() / ".claude" / "mcp-skill-hub" / "logs" / "hook-debug.log"
STATE = Path.home() / ".claude" / "mcp-skill-hub" / "state" / "auto_proceed.json"
CONFIG = Path.home() / ".claude" / "mcp-skill-hub" / "config.json"
PLANS_DIR = Path.home() / ".claude" / "plans"
DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"
RECENT_MARKER_MAX_AGE_S = 60 * 60  # 60 minutes
RECENT_MARKER_LITERAL = "<!-- auto-proceed -->"


def load_config() -> dict:
    try:
        return json.loads(CONFIG.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _in_window(now_hour: int, start: int, end: int) -> bool:
    """Window honors wrap-around (e.g. 23 -> 7 covers 23,0,1,2,3,4,5,6)."""
    if start == end:
        return False
    if start < end:
        return start <= now_hour < end
    return now_hour >= start or now_hour < end


def is_enabled(cfg: dict) -> bool:
    env = os.environ.get("SKILL_HUB_AUTO_PROCEED")
    if env is not None:
        return env == "1"
    if cfg.get("auto_proceed", False):
        return True
    window = cfg.get("auto_proceed_window")
    if isinstance(window, dict):
        try:
            start = int(window.get("start_hour", 23))
            end = int(window.get("end_hour", 7))
        except (TypeError, ValueError):
            return False
        return _in_window(datetime.now().hour, start, end)
    return False


def max_proceeds(cfg: dict) -> int:
    env = os.environ.get("SKILL_HUB_MAX_PROCEEDS")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            pass
    return int(cfg.get("auto_proceed_max", 20))


def log(msg: str) -> None:
    try:
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LOG, "a") as f:
            tag = verdict_cache.task_tag()
            f.write(f"[{datetime.now():%H:%M:%S}] AUTO_PROCEED {tag}{msg}\n")
    except OSError:
        pass


def load_state() -> dict:
    if not STATE.exists():
        return {}
    try:
        return json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(data: dict) -> None:
    try:
        STATE.parent.mkdir(parents=True, exist_ok=True)
        STATE.write_text(json.dumps(data, indent=2))
    except OSError as e:
        log(f"state_save_error  {e}")


_ARCHIVED_NAME_PREFIXES = ("_archived-", "_done-", "_superseded-")


def is_archived_plan(plan_path: Path) -> bool:
    """Skip plans flagged as archived — by filename prefix or first-line marker."""
    name = plan_path.name
    if name.startswith(_ARCHIVED_NAME_PREFIXES):
        return True
    try:
        with plan_path.open("r") as fh:
            first = fh.readline()
    except OSError:
        return False
    return "[ARCHIVED" in first or "[DONE" in first or "[SUPERSEDED" in first


def has_unchecked_items(plan_path: Path) -> bool:
    if is_archived_plan(plan_path):
        return False
    try:
        text = plan_path.read_text()
    except OSError:
        return False
    return bool(re.search(r"^\s*-\s*\[\s\]", text, re.MULTILINE))


def newest_active_plan() -> Path | None:
    if not PLANS_DIR.exists():
        return None
    plans = sorted(PLANS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in plans:
        if has_unchecked_items(p):
            return p
    return None


def has_open_task_for_session(session_id: str) -> bool:
    """Return True if an open task exists for this session in the skill-hub DB."""
    if not session_id or not DB_PATH.exists():
        return False
    try:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            row = conn.execute(
                "SELECT 1 FROM tasks WHERE status='open' AND session_id=? "
                "ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()
    except sqlite3.Error:
        return False


_QUESTION_PATTERNS = [
    re.compile(r"\(a\).{0,200}\(b\)", re.IGNORECASE | re.DOTALL),
    re.compile(
        r"\b(want(?: me)? to|should i|shall i|do you want me to|confirm|pause here"
        r"|keep going|push through|proceed\?|continue\?)\b",
        re.IGNORECASE,
    ),
]
# Intentionally no bare trailing-? pattern — that matches every conversational
# question and causes runaway auto-proceed loops (dozens of Stop-hook fires/session).


def last_message_is_clarifying_question(data: dict) -> bool:
    """Detect a mid-task pause question in the last assistant message.

    Matches multi-choice offers ((a)/(b)) and explicit hedged asks
    ("want me to", "should I", "proceed?"). The bare trailing-? pattern
    is excluded: almost every summary message ends with '?' and auto-feeding
    those causes runaway loops (dozens of Stop-hook fires per session).
    """
    msg = (data.get("last_assistant_message") or "").strip()
    if not msg:
        return False
    tail = msg[-1500:]
    for pat in _QUESTION_PATTERNS:
        if pat.search(tail):
            return True
    return False


def recent_marker_plan() -> Path | None:
    """Return a plan modified in the last 60min containing the marker literal."""
    if not PLANS_DIR.exists():
        return None
    now = time.time()
    plans = sorted(PLANS_DIR.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in plans:
        try:
            if now - p.stat().st_mtime > RECENT_MARKER_MAX_AGE_S:
                continue
            if is_archived_plan(p):
                continue
            if RECENT_MARKER_LITERAL in p.read_text():
                return p
        except OSError:
            continue
    return None


def main() -> int:
    cfg = load_config()
    if not is_enabled(cfg):
        return 0
    cap = max_proceeds(cfg)

    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return 0

    # Avoid infinite Stop-hook recursion.
    if data.get("stop_hook_active"):
        log("skip  reason=stop_hook_active")
        return 0

    session_id = data.get("session_id") or "unknown"

    # Multi-signal activation: plan checklist | open DB task | recent marker.
    signal = "none"
    source_label = ""
    plan = newest_active_plan()
    if plan is not None:
        signal = "plan_checklist"
        source_label = f"plan {plan.name} has unchecked items"
    elif has_open_task_for_session(session_id):
        signal = "open_task"
        source_label = f"open task for session {session_id}"
    else:
        marker_plan = recent_marker_plan()
        if marker_plan is not None:
            signal = "recent_marker"
            plan = marker_plan
            source_label = f"plan {marker_plan.name} has auto-proceed marker"
        # clarifying_question as standalone signal removed: scanning the full
        # message body for keywords produces false positives (e.g. a message
        # explaining the patterns will match its own examples). Without an
        # active plan or task, auto-proceed should not fire.

    log(f"signal={signal}  session={session_id}")
    if signal == "none":
        log(f"skip  reason=no_active_signal  session={session_id}")
        return 0

    state = load_state()
    plan_key = str(plan) if plan is not None else f"task:{session_id}"
    sess = state.setdefault(session_id, {"count": 0, "plan": plan_key})
    if sess["count"] >= cap:
        log(f"skip  reason=max_proceeds_reached  session={session_id}  count={sess['count']}")
        return 0

    sess["count"] += 1
    sess["plan"] = plan_key
    sess["last"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)

    log(f"PROCEED  session={session_id}  count={sess['count']}  signal={signal}")

    # If the chrome-intents queue has pending items, remind Claude to drain
    # them via the chrome-devtools MCP before continuing.
    extras = []
    intents_path = (
        Path.home() / ".claude" / "mcp-skill-hub" / "state"
        / "chrome_intents.jsonl"
    )
    if intents_path.exists():
        try:
            pending = 0
            for raw in intents_path.read_text().splitlines():
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if obj.get("status") == "pending":
                    pending += 1
            if pending:
                extras.append(
                    f"{pending} chrome intent(s) pending — drain the queue "
                    f"via the chrome-devtools MCP "
                    f"(see /intents in the dashboard)."
                )
        except OSError:
            pass

    reason = f"proceed (auto — {sess['count']}/{cap}, {source_label})"
    out: dict = {"decision": "block", "reason": reason}
    if extras:
        out["systemMessage"] = " | ".join(extras)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
