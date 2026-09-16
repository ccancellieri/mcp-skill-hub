#!/usr/bin/env python3
"""Optional deterministic PreToolUse policy.

Native client permissions are the default. Set ``hook_approval_policy`` to
``"explicit"`` to enable the existing allow-list and deny-pattern policy.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import verdict_cache  # noqa: E402


def load_yaml(path: Path) -> dict:
    """Load a list-only allow-list file, returning an empty mapping on errors."""
    if not path.exists():
        return {}
    try:
        import yaml  # type: ignore
    except ImportError:
        data: dict[str, list[str]] = {}
        key: str | None = None
        try:
            for raw in path.read_text().splitlines():
                line = raw.rstrip()
                if not line or line.lstrip().startswith("#"):
                    continue
                if not line.startswith((" ", "\t")) and line.endswith(":"):
                    key = line[:-1].strip()
                    data[key] = []
                elif key is not None and line.lstrip().startswith("-"):
                    value = line.lstrip()[1:].strip().strip('"').strip("'")
                    if value:
                        data[key].append(value)
            return data
        except OSError:
            return {}
    try:
        with path.open() as stream:
            return yaml.safe_load(stream) or {}
    except (OSError, yaml.YAMLError):
        return {}


def load_allow_list(cwd: Path) -> dict:
    """Merge user then project allow-list entries."""
    merged = {"safe_bash_prefixes": [], "safe_tools": [], "deny_patterns": []}
    for path in (
        Path.home() / ".claude" / "skill-hub-allow.yml",
        cwd / ".claude" / "skill-hub-allow.yml",
    ):
        data = load_yaml(path)
        for key in merged:
            values = data.get(key, []) if isinstance(data, dict) else []
            merged[key].extend(value for value in values if isinstance(value, str))
    return merged


def extract_bash_command(tool_input: dict) -> str:
    return (tool_input.get("command") or "").strip()


def _quote_mask(command: str) -> list[bool]:
    """Return whether each character is inside single or double quotes."""
    mask = [False] * len(command)
    quote: str | None = None
    index = 0
    while index < len(command):
        char = command[index]
        if quote is not None:
            mask[index] = True
            if char == "\\" and quote == '"' and index + 1 < len(command):
                mask[index + 1] = True
                index += 2
                continue
            if char == quote:
                quote = None
        elif char in ("'", '"'):
            quote = char
        index += 1
    return mask


def _has_unsupported_shell_syntax(command: str) -> bool:
    """Reject shell forms that the deterministic segment parser cannot model."""
    quote: str | None = None
    index = 0
    while index < len(command):
        char = command[index]
        if quote == "'":
            if char == "'":
                quote = None
            index += 1
            continue
        if quote == '"':
            if char == "\\" and index + 1 < len(command):
                index += 2
                continue
            if char == '"':
                quote = None
            elif char in ("$", "`"):
                return True
            index += 1
            continue
        if char in ("'", '"'):
            quote = char
        elif char == "\\" and index + 1 < len(command):
            index += 2
            continue
        elif char in ("$", "`", "&", "<", ">", "\n", "\r"):
            return True
        index += 1
    return quote is not None


def split_compound_segments(command: str) -> list[str]:
    """Split shell commands at unquoted ``&&``, ``||``, ``;``, and ``|``."""
    if not command:
        return []
    mask = _quote_mask(command)
    segments: list[str] = []
    start = index = 0
    while index < len(command):
        if mask[index]:
            index += 1
            continue
        operator = next(
            (op for op in ("&&", "||", ";", "|") if command.startswith(op, index)),
            None,
        )
        if operator is None:
            index += 1
            continue
        segment = command[start:index].strip()
        if segment:
            segments.append(segment)
        index += len(operator)
        start = index
    tail = command[start:].strip()
    if tail:
        segments.append(tail)
    return segments or [command.strip()]


def _cd_target_ok(segment: str) -> bool:
    """Allow a standalone ``cd`` only when its destination remains local."""
    try:
        tokens = shlex.split(segment)
    except ValueError:
        return False
    if not tokens or tokens[0] != "cd" or len(tokens) > 2:
        return False
    if len(tokens) == 1 or tokens[1] in ("-", "~") or tokens[1].startswith("~/"):
        return True
    target = os.path.expanduser(os.path.expandvars(tokens[1]))
    if not target.startswith("/"):
        return ".." not in target.split("/")
    home = str(Path.home()).rstrip("/") + "/"
    return (target + "/").startswith(home)


def _matches_prefix(segment: str, prefixes: list[str]) -> str | None:
    for prefix in prefixes:
        if segment == prefix or segment.startswith(prefix + " ") or segment.startswith(prefix + "\n"):
            return prefix
    return None


def decide(tool_name: str, tool_input: dict, allow: dict) -> tuple[str, str]:
    """Apply only the explicit allow-list and deny-pattern policy."""
    if tool_name in allow.get("safe_tools", []):
        return "approve", f"tool '{tool_name}' in safe_tools"
    if tool_name != "Bash":
        return "", ""

    command = extract_bash_command(tool_input)
    if not command:
        return "", ""
    if _has_unsupported_shell_syntax(command):
        return "", ""
    for pattern in allow.get("deny_patterns", []):
        try:
            if re.search(pattern, command):
                return "block", f"matched deny_pattern: {pattern}"
        except re.error:
            continue

    prefixes = allow.get("safe_bash_prefixes", [])
    segments = split_compound_segments(command)
    matched: list[str] = []
    for segment in segments:
        if _cd_target_ok(segment):
            matched.append("cd")
            continue
        prefix = _matches_prefix(segment, prefixes)
        if prefix is None:
            return "", ""
        matched.append(prefix)
    if len(segments) == 1:
        return "approve", f"matched safe_bash_prefix: {matched[0]}"
    return "approve", f"all {len(segments)} segments in allow-list ({', '.join(sorted(set(matched)))})"


def main() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, EOFError):
        return 0

    if verdict_cache.load_config().get("hook_approval_policy", "native") != "explicit":
        return 0

    tool_name = data.get("tool_name", "")
    tool_input = data.get("tool_input", {}) or {}
    allow = load_allow_list(Path(data.get("cwd") or os.getcwd()))
    decision, reason = decide(tool_name, tool_input, allow)
    if not decision:
        return 0
    permission = "allow" if decision == "approve" else "deny"
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PreToolUse",
        "permissionDecision": permission,
        "permissionDecisionReason": reason,
    }}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
