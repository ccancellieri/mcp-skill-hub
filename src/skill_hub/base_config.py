"""Canonical mcp-skill-hub hook configuration + an idempotent settings merger.

Claude Code (and other tools) periodically rewrite ``~/.claude/settings.json``
and drop the skill-hub hooks, silently removing this MCP from the request loop
(no router, no session enforcer, no compression). This module defines the
known-good base hook configuration and merges it back into settings.json
**without clobbering unrelated user settings**.

It powers:
  * ``/hub-install-hooks``  — re-apply the base hooks (alias ``/hub-enforce``)
  * ``/hub-install-hooks check``  — report which skill-hub hooks are missing
  * ``python -m skill_hub.base_config [check|--dry-run]``

The merge is idempotent: hooks are matched by their script *basename*, so
re-running never duplicates an entry, and a clobbered settings.json is repaired
by appending only the missing skill-hub hook groups. Other tools' hooks
(codegraph, ~/.claude/hooks/*, model-tier enforcement) are left untouched.

Ref: https://code.claude.com/docs/en/hooks
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path.home() / ".claude" / "settings.json"


def hooks_dir() -> Path:
    """Absolute path to this repo's ``hooks/`` directory.

    ``SKILL_HUB_HOOKS_DIR`` overrides it (used by tests and non-standard
    checkouts). Otherwise it is resolved relative to this file:
    ``src/skill_hub/base_config.py`` → ``<repo>/hooks``.
    """
    override = os.environ.get("SKILL_HUB_HOOKS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "hooks"


# The skill-hub-owned hooks, in the exact shape Claude Code expects. Each entry
# becomes one hook group: {"hooks": [{"type": "command", "command": <abs path>,
# ...}]}. ``if`` (a tool matcher like "Bash(*)") and ``matcher`` are optional.
# This list is the single source of truth for what "the MCP in the loop" means.
_HOOKS: tuple[dict[str, Any], ...] = (
    {"event": "PreCompact", "script": "precompact.sh", "timeout": 10,
     "statusMessage": "Snapshotting routing state..."},
    {"event": "UserPromptSubmit", "script": "session-start-enforcer.sh", "timeout": 5,
     "statusMessage": "Checking session start protocol..."},
    {"event": "UserPromptSubmit", "script": "intercept-task-commands.sh", "timeout": 45,
     "statusMessage": "Skill Hub: enriching context..."},
    {"event": "UserPromptSubmit", "script": "prompt-router.sh", "timeout": 20,
     "statusMessage": "Routing prompt..."},
    {"event": "Stop", "script": "session-end.sh", "timeout": 45,
     "statusMessage": "Saving session memory..."},
    {"event": "Stop", "script": "auto-proceed.sh", "timeout": 5,
     "statusMessage": "Checking for plan continuation..."},
    {"event": "PreToolUse", "script": "auto-approve.sh", "if": "Bash(*)", "timeout": 5,
     "statusMessage": "Checking allow-list..."},
    {"event": "PostToolUse", "script": "post-tool-observer.sh", "if": "Bash(*)", "timeout": 5,
     "statusMessage": "Recording approved command..."},
    {"event": "PostToolUseFailure", "script": "post-tool-observer.sh", "if": "Bash(*)", "timeout": 5,
     "statusMessage": "Recording failed command..."},
    {"event": "StopFailure", "script": "stop-failure.sh", "timeout": 5,
     "statusMessage": "Logging API error..."},
    {"event": "SessionEnd", "script": "session-end-real.sh", "timeout": 30,
     "statusMessage": "Closing session..."},
    {"event": "PostCompact", "script": "postcompact.sh", "timeout": 30,
     "statusMessage": "Optimising memory..."},
    {"event": "SubagentStart", "script": "subagent-observer.sh", "timeout": 5,
     "statusMessage": "Logging subagent start..."},
    {"event": "SubagentStop", "script": "subagent-observer.sh", "timeout": 5,
     "statusMessage": "Logging subagent stop..."},
)


def _build_hook(spec: dict[str, Any]) -> dict[str, Any]:
    """Render a single hook-command dict with an absolute command path."""
    hook: dict[str, Any] = {
        "type": "command",
        "command": str(hooks_dir() / spec["script"]),
    }
    if "if" in spec:
        hook["if"] = spec["if"]
    if "timeout" in spec:
        hook["timeout"] = spec["timeout"]
    if "statusMessage" in spec:
        hook["statusMessage"] = spec["statusMessage"]
    return hook


def _first_token(command: str) -> str:
    """The command path, ignoring any trailing args."""
    return command.split(None, 1)[0] if command else ""


def _script_basenames_in_event(settings: dict[str, Any], event: str) -> set[str]:
    """All ``*.sh`` basenames already wired for ``event`` in settings."""
    names: set[str] = set()
    for group in settings.get("hooks", {}).get(event, []) or []:
        for h in group.get("hooks", []) or []:
            tok = _first_token(h.get("command", ""))
            if tok.endswith(".sh"):
                names.add(Path(tok).name)
    return names


def base_hooks() -> dict[str, list[dict[str, Any]]]:
    """The base hook block as Claude Code's ``hooks`` mapping (event → groups)."""
    out: dict[str, list[dict[str, Any]]] = {}
    for spec in _HOOKS:
        out.setdefault(spec["event"], []).append({"hooks": [_build_hook(spec)]})
    return out


def check(settings: dict[str, Any]) -> list[str]:
    """Return ``event:script`` labels for every skill-hub hook NOT installed."""
    missing: list[str] = []
    for spec in _HOOKS:
        if spec["script"] not in _script_basenames_in_event(settings, spec["event"]):
            missing.append(f"{spec['event']}:{spec['script']}")
    return missing


def merge(settings: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Add any missing skill-hub hooks to ``settings`` (mutates and returns it).

    Idempotent: a hook already present (matched by script basename within its
    event) is skipped, so re-running adds nothing. Existing groups, matchers,
    and non-skill-hub hooks are preserved untouched.

    Returns ``(settings, added)`` where ``added`` is the list of
    ``event:script`` labels that were appended.
    """
    added: list[str] = []
    hooks = settings.setdefault("hooks", {})
    for spec in _HOOKS:
        event = spec["event"]
        if spec["script"] in _script_basenames_in_event(settings, event):
            continue
        hooks.setdefault(event, []).append({"hooks": [_build_hook(spec)]})
        added.append(f"{event}:{spec['script']}")
    return settings, added


def missing_scripts_on_disk() -> list[str]:
    """Base hooks whose script file is absent from ``hooks_dir()`` — a config
    that points at non-existent scripts is worse than none."""
    hd = hooks_dir()
    seen: set[str] = set()
    absent: list[str] = []
    for spec in _HOOKS:
        name = spec["script"]
        if name in seen:
            continue
        seen.add(name)
        if not (hd / name).exists():
            absent.append(name)
    return absent


def install(
    settings_path: Path = SETTINGS_PATH,
    *,
    dry_run: bool = False,
    backup: bool = True,
) -> dict[str, Any]:
    """Re-apply the base hooks into ``settings_path`` idempotently.

    Returns a report dict: ``{settings_path, existed, added, already_present,
    missing_scripts, backup_path, dry_run}``. Writes a ``.bak`` beside the
    settings file before modifying it (unless ``backup=False`` or nothing
    changed). Never raises on a missing settings file — it is created.
    """
    settings_path = Path(settings_path)
    existed = settings_path.exists()
    settings: dict[str, Any] = {}
    if existed:
        try:
            settings = json.loads(settings_path.read_text(encoding="utf-8"))
            if not isinstance(settings, dict):
                settings = {}
        except (OSError, ValueError):
            settings = {}

    before_missing = check(settings)
    merged, added = merge(settings)

    report: dict[str, Any] = {
        "settings_path": str(settings_path),
        "existed": existed,
        "added": added,
        "already_present": [m for m in (
            f"{s['event']}:{s['script']}" for s in _HOOKS
        ) if m not in before_missing],
        "missing_scripts": missing_scripts_on_disk(),
        "backup_path": None,
        "dry_run": dry_run,
    }

    if dry_run or not added:
        return report

    if backup and existed:
        bak = settings_path.with_suffix(settings_path.suffix + ".bak")
        try:
            bak.write_text(settings_path.read_text(encoding="utf-8"), encoding="utf-8")
            report["backup_path"] = str(bak)
        except OSError:
            pass

    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return report


def format_report(report: dict[str, Any]) -> str:
    """Human-readable summary for the CLI / slash command."""
    lines = ["=== mcp-skill-hub hook install ==="]
    lines.append(f"settings: {report['settings_path']}"
                 + ("" if report["existed"] else " (created)"))
    if report["missing_scripts"]:
        lines.append("⚠ base hook scripts missing on disk: "
                     + ", ".join(report["missing_scripts"]))
    if report["dry_run"]:
        if report["added"]:
            lines.append(f"Would add {len(report['added'])} hook(s):")
            lines.extend(f"  + {a}" for a in report["added"])
        else:
            lines.append("All skill-hub hooks already present — nothing to do.")
        return "\n".join(lines)
    if report["added"]:
        lines.append(f"Re-applied {len(report['added'])} missing hook(s):")
        lines.extend(f"  + {a}" for a in report["added"])
        if report["backup_path"]:
            lines.append(f"Backup written: {report['backup_path']}")
        lines.append("⚠ Restart Claude Code for the hooks to take effect.")
    else:
        lines.append("All skill-hub hooks already present — nothing to do.")
    return "\n".join(lines)


def _main(argv: list[str]) -> int:
    args = [a.lower() for a in argv]
    if "check" in args:
        try:
            settings = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            settings = {}
        miss = check(settings)
        if miss:
            print("Missing skill-hub hooks:")
            for m in miss:
                print(f"  - {m}")
            print("\nRun `python -m skill_hub.base_config` to re-apply.")
            return 1
        print("All skill-hub hooks present.")
        return 0
    dry = "--dry-run" in args or "dry-run" in args
    report = install(dry_run=dry)
    print(format_report(report))
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    raise SystemExit(_main(sys.argv[1:]))
