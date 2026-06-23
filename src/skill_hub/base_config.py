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
(codegraph, hooks in other dirs, model-tier enforcement) are left untouched.

Beyond hooks, this module also manages:
  * MCP-server registration in ``~/.claude.json``  (``check_mcp`` / ``merge_mcp``)
  * A sentinel "base roles" block in ``~/.claude/CLAUDE.md``            (``check_roles`` / ``merge_roles``)
  * Unified ``check_all`` / ``restore_all`` covering all three surfaces.

Ref: https://code.claude.com/docs/en/hooks
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

# Default paths — overridable via env for tests.
_CLAUDE_JSON_PATH = Path.home() / ".claude.json"
_CLAUDE_MD_PATH = Path.home() / ".claude" / "CLAUDE.md"

CLAUDE_JSON_PATH: Path = Path(
    os.environ.get("SKILL_HUB_CLAUDE_JSON", str(_CLAUDE_JSON_PATH))
)
CLAUDE_MD_PATH: Path = Path(
    os.environ.get("SKILL_HUB_CLAUDE_MD", str(_CLAUDE_MD_PATH))
)

# Sentinel markers for the managed block in CLAUDE.md.
_ROLES_START = "<!-- skill-hub:base-roles:start -->"
_ROLES_END = "<!-- skill-hub:base-roles:end -->"

# Generic, self-contained "base roles" block — no paths, slugs, or PII.
_BASE_ROLES_CONTENT = """\
## Model-Tier Routing

- **Opus** — design, architecture, brainstorming, complex decisions. Use for judgement, not labour.
- **Sonnet** — implementation, refactoring, test writing, code review, commit-message authoring.
- **Haiku** — read-only research (symbol lookup, grep sweeps, log trawls), git mechanics (stage, commit, push, branch, PR plumbing).

Dispatch every Agent with an explicit `model:` parameter matching the tier above.
Inheriting Opus for mechanical work is a token leak.

## Skill-Hub MCP

Keep the skill-hub MCP server in the request loop (hooks + router active).
Call `search_skills` at conversation start and after topic changes.

## Karpathy Guidelines

- **Think Before Coding** — read the spec and existing patterns before writing.
- **Simplicity First** — smallest correct change; no speculative abstractions.
- **Surgical Changes** — match surrounding style; don't refactor unrelated code.
- **Goal-Driven Execution** — every line must be traceable to a stated requirement.
"""


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


def _repo_root() -> Path:
    """Absolute path to the repository root (parent of ``src/`` and ``hooks/``)."""
    return Path(__file__).resolve().parents[2]


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


# ---------------------------------------------------------------------------
# MCP-server registration in ~/.claude.json
# ---------------------------------------------------------------------------

def _default_mcp_entry() -> dict[str, Any]:
    """Canonical skill-hub MCP entry derived from the repo layout.

    Uses the venv-installed ``skill-hub`` binary if present; otherwise falls
    back to ``python -m skill_hub.server`` so the entry works even outside a
    venv (e.g. a dev editable install).
    """
    venv_bin = _repo_root() / ".venv" / "bin" / "skill-hub"
    if venv_bin.exists():
        return {"type": "stdio", "command": str(venv_bin)}
    return {
        "type": "stdio",
        "command": "python",
        "args": ["-m", "skill_hub.server"],
    }


def _resolve_claude_json_path(claude_json_path: Path | None) -> Path:
    if claude_json_path is not None:
        return Path(claude_json_path)
    env = os.environ.get("SKILL_HUB_CLAUDE_JSON")
    return Path(env) if env else _CLAUDE_JSON_PATH


def check_mcp(claude_json_path: Path | None = None) -> list[str]:
    """Return a list of issues with the skill-hub MCP registration.

    Returns an empty list when the entry is present and well-formed.
    """
    path = _resolve_claude_json_path(claude_json_path)
    if not path.exists():
        return ["skill-hub entry missing (claude.json not found)"]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return ["claude.json is not a JSON object"]
    except (OSError, ValueError) as exc:
        return [f"cannot read claude.json: {exc}"]

    servers = data.get("mcpServers") or {}
    if "skill-hub" not in servers:
        return ["skill-hub not in mcpServers"]
    entry = servers["skill-hub"]
    if not isinstance(entry, dict) or not entry.get("command"):
        return ["skill-hub mcpServers entry is malformed (no command)"]
    return []


def merge_mcp(
    claude_json_path: Path | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Ensure skill-hub is registered in ``claude_json_path``.

    Reads the existing file (if present) to use its skill-hub entry as the
    template, preserving any secrets already there.  If no entry exists,
    derives a default from the repo layout.  All other keys and mcpServers
    are left untouched.

    Returns ``(data, added)`` where ``added`` is a list of string labels
    describing what changed (empty when already correct).
    """
    path = _resolve_claude_json_path(claude_json_path)
    data: dict[str, Any] = {}
    parse_ok = True
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                data = raw
            else:
                parse_ok = False
        except (OSError, ValueError):
            parse_ok = False

    if not parse_ok:
        # Refuse to overwrite a file we couldn't parse — data-loss prevention.
        return data, []

    servers = data.setdefault("mcpServers", {})
    added: list[str] = []

    entry = servers.get("skill-hub")
    if not isinstance(entry, dict) or not entry.get("command"):
        servers["skill-hub"] = _default_mcp_entry()
        added.append("mcpServers:skill-hub")

    return data, added


def install_mcp(
    claude_json_path: Path | None = None,
    *,
    dry_run: bool = False,
    backup: bool = True,
) -> dict[str, Any]:
    """Idempotently register skill-hub in ``claude_json_path``.

    Returns a report dict parallel in structure to ``install()``.
    """
    path = _resolve_claude_json_path(claude_json_path)
    existed = path.exists()

    issues_before = check_mcp(path)
    data, added = merge_mcp(path)

    report: dict[str, Any] = {
        "claude_json_path": str(path),
        "existed": existed,
        "added": added,
        "issues_before": issues_before,
        "backup_path": None,
        "dry_run": dry_run,
    }

    if dry_run or not added:
        return report

    if backup and existed:
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            report["backup_path"] = str(bak)
        except OSError:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Managed "base roles" block in ~/.claude/CLAUDE.md
# ---------------------------------------------------------------------------

def _resolve_claude_md_path(claude_md_path: Path | None) -> Path:
    if claude_md_path is not None:
        return Path(claude_md_path)
    env = os.environ.get("SKILL_HUB_CLAUDE_MD")
    return Path(env) if env else _CLAUDE_MD_PATH


def check_roles(claude_md_path: Path | None = None) -> list[str]:
    """Return issues with the managed base-roles block in CLAUDE.md.

    Returns an empty list when the block is present and intact.
    """
    path = _resolve_claude_md_path(claude_md_path)
    if not path.exists():
        return ["base-roles block missing (CLAUDE.md not found)"]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [f"cannot read CLAUDE.md: {exc}"]
    if _ROLES_START not in text:
        return ["base-roles sentinel block not present in CLAUDE.md"]
    if _ROLES_END not in text:
        return ["base-roles block start found but end sentinel missing"]
    return []


def merge_roles(
    claude_md_path: Path | None = None,
) -> tuple[str, list[str]]:
    """Idempotently insert/refresh the base-roles sentinel block.

    Content outside the sentinels is preserved byte-for-byte.  The inner
    content is replaced with ``_BASE_ROLES_CONTENT`` on every call so it
    stays fresh.  If the sentinels are absent the block is appended.

    Returns ``(new_text, added)`` where ``added`` describes what changed.
    """
    path = _resolve_claude_md_path(claude_md_path)
    existing = ""
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError:
            existing = ""

    block = f"{_ROLES_START}\n{_BASE_ROLES_CONTENT}{_ROLES_END}\n"
    added: list[str] = []

    if _ROLES_START in existing and _ROLES_END in existing:
        # Replace only the inner content.
        start_idx = existing.index(_ROLES_START)
        end_idx = existing.index(_ROLES_END) + len(_ROLES_END)
        old_block = existing[start_idx:end_idx]
        new_block = f"{_ROLES_START}\n{_BASE_ROLES_CONTENT}{_ROLES_END}"
        if old_block == new_block:
            return existing, []  # already up to date
        new_text = existing[:start_idx] + new_block + existing[end_idx:]
        added.append("CLAUDE.md:base-roles:refreshed")
    elif _ROLES_START not in existing:
        # Append the block with exactly one blank line separator.
        if not existing:
            sep = ""
        elif existing.endswith("\n\n"):
            sep = ""
        elif existing.endswith("\n"):
            sep = "\n"
        else:
            sep = "\n\n"
        new_text = existing + sep + block
        added.append("CLAUDE.md:base-roles:inserted")
    else:
        # Start sentinel present but end is missing — append end.
        new_text = existing.rstrip() + "\n" + _ROLES_END + "\n"
        added.append("CLAUDE.md:base-roles:end-sentinel-repaired")

    return new_text, added


def install_roles(
    claude_md_path: Path | None = None,
    *,
    dry_run: bool = False,
    backup: bool = True,
) -> dict[str, Any]:
    """Idempotently install/refresh the base-roles block in CLAUDE.md.

    Returns a report dict parallel in structure to ``install()``.
    """
    path = _resolve_claude_md_path(claude_md_path)
    existed = path.exists()

    issues_before = check_roles(path)
    new_text, added = merge_roles(path)

    report: dict[str, Any] = {
        "claude_md_path": str(path),
        "existed": existed,
        "added": added,
        "issues_before": issues_before,
        "backup_path": None,
        "dry_run": dry_run,
    }

    if dry_run or not added:
        return report

    if backup and existed:
        bak = path.with_suffix(path.suffix + ".bak")
        try:
            bak.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            report["backup_path"] = str(bak)
        except OSError:
            pass

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(new_text, encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Unified check_all / restore_all
# ---------------------------------------------------------------------------

def _resolve_settings_path(settings_path: Path | None) -> Path:
    """Return the effective settings.json path, using the module attribute as default."""
    if settings_path is not None:
        return Path(settings_path)
    return SETTINGS_PATH


def check_all(
    settings_path: Path | None = None,
    claude_json_path: Path | None = None,
    claude_md_path: Path | None = None,
) -> dict[str, list[str]]:
    """Return a dict with missing/broken items across all three surfaces.

    Keys: ``hooks``, ``mcp``, ``roles``.  Each value is a list of issue
    strings (empty list = all present / healthy).

    All path arguments default to the module-level constants (``SETTINGS_PATH``,
    ``CLAUDE_JSON_PATH``, ``CLAUDE_MD_PATH``), which tests can override via
    ``monkeypatch.setattr(base_config, "SETTINGS_PATH", ...)`` without the
    function's default being captured at definition time.
    """
    sp = _resolve_settings_path(settings_path)
    settings: dict[str, Any] = {}
    if sp.exists():
        try:
            raw = json.loads(sp.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                settings = raw
        except (OSError, ValueError):
            pass

    return {
        "hooks": check(settings),
        "mcp": check_mcp(claude_json_path),
        "roles": check_roles(claude_md_path),
    }


def restore_all(
    settings_path: Path | None = None,
    claude_json_path: Path | None = None,
    claude_md_path: Path | None = None,
    *,
    dry_run: bool = False,
    backup: bool = True,
) -> dict[str, Any]:
    """Re-apply all three surfaces idempotently.

    Returns a dict with keys ``hooks``, ``mcp``, ``roles`` — each value is
    the corresponding install report.  Never raises.

    Path arguments default to the module-level constants so that tests can
    patch them via ``monkeypatch.setattr`` without a frozen default value.
    """
    sp = _resolve_settings_path(settings_path)
    try:
        hooks_report = install(sp, dry_run=dry_run, backup=backup)
    except Exception as exc:
        hooks_report = {"error": str(exc)}

    try:
        mcp_report = install_mcp(claude_json_path, dry_run=dry_run, backup=backup)
    except Exception as exc:
        mcp_report = {"error": str(exc)}

    try:
        roles_report = install_roles(claude_md_path, dry_run=dry_run, backup=backup)
    except Exception as exc:
        roles_report = {"error": str(exc)}

    return {"hooks": hooks_report, "mcp": mcp_report, "roles": roles_report}


# ---------------------------------------------------------------------------
# format_report — extended to cover all three sections
# ---------------------------------------------------------------------------

def format_report(report: dict[str, Any]) -> str:
    """Human-readable summary for the CLI / slash command.

    Handles both the legacy hooks-only shape and the new unified shape
    returned by ``restore_all()``.
    """
    # Unified report from restore_all
    if "hooks" in report and "mcp" in report and "roles" in report:
        parts: list[str] = []
        parts.append(_format_hooks_section(report["hooks"]))
        parts.append(_format_mcp_section(report["mcp"]))
        parts.append(_format_roles_section(report["roles"]))
        return "\n\n".join(parts)

    # Legacy hooks-only shape (from install())
    return _format_hooks_section(report)


def _format_hooks_section(report: dict[str, Any]) -> str:
    lines = ["=== mcp-skill-hub hook install ==="]
    if "error" in report:
        lines.append(f"ERROR: {report['error']}")
        return "\n".join(lines)
    lines.append(f"settings: {report.get('settings_path', '?')}"
                 + ("" if report.get("existed") else " (created)"))
    if report.get("missing_scripts"):
        lines.append("WARNING base hook scripts missing on disk: "
                     + ", ".join(report["missing_scripts"]))
    if report.get("dry_run"):
        if report.get("added"):
            lines.append(f"Would add {len(report['added'])} hook(s):")
            lines.extend(f"  + {a}" for a in report["added"])
        else:
            lines.append("All skill-hub hooks already present — nothing to do.")
        return "\n".join(lines)
    if report.get("added"):
        lines.append(f"Re-applied {len(report['added'])} missing hook(s):")
        lines.extend(f"  + {a}" for a in report["added"])
        if report.get("backup_path"):
            lines.append(f"Backup written: {report['backup_path']}")
        lines.append("WARNING Restart Claude Code for the hooks to take effect.")
    else:
        lines.append("All skill-hub hooks already present — nothing to do.")
    return "\n".join(lines)


def _format_mcp_section(report: dict[str, Any]) -> str:
    lines = ["=== MCP server registration (~/.claude.json) ==="]
    if "error" in report:
        lines.append(f"ERROR: {report['error']}")
        return "\n".join(lines)
    lines.append(f"path: {report.get('claude_json_path', '?')}"
                 + ("" if report.get("existed") else " (created)"))
    if report.get("dry_run"):
        if report.get("added"):
            lines.append("Would add: " + ", ".join(report["added"]))
        else:
            lines.append("skill-hub already registered — nothing to do.")
        return "\n".join(lines)
    if report.get("added"):
        lines.append("Registered: " + ", ".join(report["added"]))
        if report.get("backup_path"):
            lines.append(f"Backup written: {report['backup_path']}")
    else:
        lines.append("skill-hub already registered — nothing to do.")
    return "\n".join(lines)


def _format_roles_section(report: dict[str, Any]) -> str:
    lines = ["=== Base roles block (~/.claude/CLAUDE.md) ==="]
    if "error" in report:
        lines.append(f"ERROR: {report['error']}")
        return "\n".join(lines)
    lines.append(f"path: {report.get('claude_md_path', '?')}"
                 + ("" if report.get("existed") else " (created)"))
    if report.get("dry_run"):
        if report.get("added"):
            lines.append("Would update: " + ", ".join(report["added"]))
        else:
            lines.append("Base-roles block already present — nothing to do.")
        return "\n".join(lines)
    if report.get("added"):
        lines.append("Updated: " + ", ".join(report["added"]))
        if report.get("backup_path"):
            lines.append(f"Backup written: {report['backup_path']}")
    else:
        lines.append("Base-roles block already present — nothing to do.")
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
