"""A3 — Plugin hook registry and dispatcher.

Plugins declare handlers in their plugin.json:

    "hooks": {
        "on_session_start":   "hooks/on_session_start.py",
        "on_session_end":     "hooks/on_session_end.py",
        "on_tool_call":       "hooks/on_tool_call.py",
        "on_skill_activated": {"script": "hooks/on_skill.py", "async": true}
    }

Each script is invoked as a subprocess: stdin receives a JSON payload describing
the event; stdout should emit JSON (ignored unless non-empty). Timeout is 10s
per sync handler; async handlers fire-and-forget on a background thread.

Handler failures NEVER propagate — they are logged and ignored.

Kill-switch: set ``PLUGIN_HOOKS_DISABLED=1`` to skip all dispatch.

See docs/plugin-extension-points.md for the canonical event payload shapes.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

from .plugin_registry import iter_enabled_plugins

_log = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 10


def _resolve_handler(plugin: dict[str, Any], event: str) -> tuple[Path, bool] | None:
    hooks = plugin["manifest"].get("hooks") or {}
    entry = hooks.get(event)
    if not entry:
        return None
    if isinstance(entry, str):
        script, async_mode = entry, False
    elif isinstance(entry, dict):
        script = entry.get("script", "")
        async_mode = bool(entry.get("async", False))
    else:
        return None
    if not script:
        return None
    path = (plugin["path"] / script).resolve()
    if not path.exists():
        _log.warning("plugin hook missing: %s %s -> %s", plugin["name"], event, path)
        return None
    return path, async_mode


def _run_handler(plugin_name: str, script: Path, payload: dict[str, Any]) -> dict[str, Any] | None:
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=_TIMEOUT_SECONDS,
            cwd=str(script.parent.parent),
        )
    except subprocess.TimeoutExpired:
        _log.warning("plugin hook timed out: %s %s", plugin_name, script.name)
        return None
    except Exception as exc:  # noqa: BLE001 — handler isolation
        _log.warning("plugin hook failed to start: %s %s (%s)", plugin_name, script.name, exc)
        return None
    if proc.returncode != 0:
        _log.warning(
            "plugin hook non-zero exit: %s %s rc=%s stderr=%s",
            plugin_name,
            script.name,
            proc.returncode,
            (proc.stderr or "")[:400],
        )
    out = (proc.stdout or "").strip()
    if not out:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {"raw": out[:4000]}


def dispatch(event: str, payload: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """Fire ``event`` to every enabled plugin that registered a handler.

    Returns the collected results from sync handlers (async handlers return
    nothing immediately). Never raises.
    """
    if os.environ.get("PLUGIN_HOOKS_DISABLED") == "1":
        return []
    payload = dict(payload or {})
    payload.setdefault("event", event)
    results: list[dict[str, Any]] = []
    for plugin in iter_enabled_plugins():
        resolved = _resolve_handler(plugin, event)
        if not resolved:
            continue
        script, async_mode = resolved
        if async_mode:
            t = threading.Thread(
                target=_run_handler,
                args=(plugin["name"], script, payload),
                daemon=True,
                name=f"plugin-hook-{plugin['name']}-{event}",
            )
            t.start()
            continue
        out = _run_handler(plugin["name"], script, payload)
        if out is not None:
            results.append({"plugin": plugin["name"], "event": event, "result": out})
    return results
