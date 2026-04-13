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
import sys
from datetime import datetime
from pathlib import Path

LOG = Path.home() / ".claude" / "mcp-skill-hub" / "logs" / "hook-debug.log"
STATE = Path.home() / ".claude" / "mcp-skill-hub" / "state" / "auto_proceed.json"
CONFIG = Path.home() / ".claude" / "mcp-skill-hub" / "config.json"
PLANS_DIR = Path.home() / ".claude" / "plans"


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
            f.write(f"[{datetime.now():%H:%M:%S}] AUTO_PROCEED {msg}\n")
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


def has_unchecked_items(plan_path: Path) -> bool:
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

    plan = newest_active_plan()
    if plan is None:
        log(f"skip  reason=no_active_plan  session={session_id}")
        return 0

    state = load_state()
    sess = state.setdefault(session_id, {"count": 0, "plan": str(plan)})
    if sess["count"] >= cap:
        log(f"skip  reason=max_proceeds_reached  session={session_id}  count={sess['count']}")
        return 0

    sess["count"] += 1
    sess["plan"] = str(plan)
    sess["last"] = datetime.now().isoformat(timespec="seconds")
    save_state(state)

    log(f"PROCEED  session={session_id}  count={sess['count']}  plan={plan.name}")
    out = {
        "decision": "block",
        "reason": f"proceed (auto — {sess['count']}/{cap}, plan {plan.name} has unchecked items)",
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
