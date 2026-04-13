"""Enforcement — apply the router verdict to Claude's model selection.

Confidence-tiered, asymmetric:
  - Upgrades (→ opus/plan) above threshold: silent hard switch (write settings.json)
  - Downgrades (→ haiku/no-plan) above threshold: one-shot session confirmation required
  - Below threshold: soft suggestion only (injected as systemMessage text)
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

# Model priority for comparison (higher = more capable)
_MODEL_PRIORITY = {"haiku": 0, "sonnet": 1, "opus": 2}

# Temp-file prefix for session downgrade-confirmation tracking
_DOWNGRADE_FLAG_PREFIX = "claude-router-downgrade-"


def _session_flag(session_id: str) -> str:
    return os.path.join(tempfile.gettempdir(), f"{_DOWNGRADE_FLAG_PREFIX}{session_id}")


def _read_current_model() -> str:
    """Return the model currently written in settings.json, or '' if absent."""
    try:
        settings = json.loads(SETTINGS_PATH.read_text())
        return str(settings.get("model", ""))
    except (OSError, json.JSONDecodeError):
        return ""


def _write_model(model: str) -> bool:
    """Overwrite only the 'model' key in settings.json. Returns True on success."""
    try:
        settings: dict[str, Any] = {}
        if SETTINGS_PATH.exists():
            try:
                settings = json.loads(SETTINGS_PATH.read_text())
            except (json.JSONDecodeError, OSError):
                pass
        settings["model"] = model
        SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(settings, indent=2))
        return True
    except OSError:
        return False


def _is_upgrade(from_model: str, to_model: str) -> bool:
    return _MODEL_PRIORITY.get(to_model, 1) > _MODEL_PRIORITY.get(from_model, 1)


def _downgrade_confirmed_this_session(session_id: str) -> bool:
    flag = _session_flag(session_id)
    return os.path.exists(flag)


def _mark_downgrade_confirmed(session_id: str) -> None:
    flag = _session_flag(session_id)
    try:
        with open(flag, "w") as f:
            f.write("")
    except OSError:
        pass


def apply(
    verdict_model: str,
    plan_mode: bool,
    confidence: float,
    session_id: str,
    cfg: dict[str, Any],
) -> tuple[str, str]:
    """Apply the verdict, returning (enforcement_action, optional_message).

    enforcement_action:
      "hard_switch" — model written to settings.json silently
      "suggest"     — soft suggestion text returned
      "none"        — nothing done (confident downgrade already confirmed)
    """
    threshold: float = float(cfg.get("router_hard_switch_threshold", 0.9))
    current_model = _read_current_model() or "sonnet"  # assume sonnet if unknown

    # Normalise alias in case settings has a full ID
    for alias in ("haiku", "sonnet", "opus"):
        if alias in current_model:
            current_model = alias
            break

    is_up = _is_upgrade(current_model, verdict_model)
    is_same = current_model == verdict_model
    is_down = not is_up and not is_same

    if is_same:
        return "none", ""

    if confidence < threshold:
        # Below threshold: soft suggestion only
        direction = "↑ upgrade" if is_up else "↓ downgrade"
        msg = (
            f"Router suggests {direction} to {verdict_model}"
            + (" + plan mode" if plan_mode else "")
            + f" (confidence={confidence:.2f} — below threshold for auto-switch). "
            "Use /model to change if you agree."
        )
        return "suggest", msg

    # Above threshold
    if is_up:
        # Silent hard switch — upgrades are safe
        _write_model(verdict_model)
        return "hard_switch", ""

    # Downgrade — only auto-apply after first session confirmation
    if not session_id or _downgrade_confirmed_this_session(session_id):
        _write_model(verdict_model)
        return "hard_switch", ""

    # First downgrade this session: ask once
    _mark_downgrade_confirmed(session_id)
    msg = (
        f"Router recommends downshifting to {verdict_model} for this prompt "
        f"(confidence={confidence:.2f}). Model NOT changed automatically — "
        "use /model haiku if you agree. Future downgrades this session will be silent."
    )
    return "suggest", msg
