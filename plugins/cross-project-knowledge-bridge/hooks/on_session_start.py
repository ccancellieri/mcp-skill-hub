#!/usr/bin/env python3
"""Hook: on_session_start — Load relevant cross-project context.

Invoked by skill_hub.plugin_hooks.dispatch() at the start of each session.

Reads the plugin config to discover related projects, then:
1. Finds decisions/patterns relevant to the current session's working directory
2. Returns context to inject into the session
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent


def _load_config() -> dict:
    mf = PLUGIN_ROOT / "plugin.json"
    if not mf.exists():
        return {}
    try:
        return json.loads(mf.read_text(encoding="utf-8")).get("config", {})
    except (json.JSONDecodeError, OSError):
        return {}


def _detect_current_project(cwd: str, projects: list[dict]) -> dict | None:
    cwd_path = Path(cwd).resolve()
    for p in projects:
        proj_path = Path(str(p.get("path", ""))).expanduser().resolve()
        try:
            if cwd_path.is_relative_to(proj_path):
                return p
        except (OSError, ValueError):
            continue
    return None


def _find_related_projects(current: dict, all_projects: list[dict]) -> list[dict]:
    if not current:
        return []
    current_tags = set(current.get("tags", []))
    related = []
    for p in all_projects:
        if p.get("name") == current.get("name"):
            continue
        p_tags = set(p.get("tags", []))
        if current_tags & p_tags:
            related.append(p)
    return related


def _extract_key_decisions(memory_dir: Path, limit: int = 5) -> list[dict]:
    decisions_file = memory_dir / "decisions.md"
    if not decisions_file.exists():
        return []
    try:
        text = decisions_file.read_text(encoding="utf-8")
    except OSError:
        return []

    decisions = []
    current = None
    for line in text.splitlines():
        if line.startswith("# "):
            if current and len(decisions) < limit:
                decisions.append(current)
            current = {"title": line[2:].strip(), "summary": ""}
        elif current and line.strip() and not line.startswith("#"):
            if not current["summary"]:
                current["summary"] = line.strip()
    if current and len(decisions) < limit:
        decisions.append(current)
    return decisions


def _extract_patterns(memory_dir: Path, limit: int = 3) -> list[dict]:
    patterns_file = memory_dir / "patterns.md"
    if not patterns_file.exists():
        return []
    try:
        text = patterns_file.read_text(encoding="utf-8")
    except OSError:
        return []

    patterns = []
    current = None
    for line in text.splitlines():
        if line.startswith("## "):
            if current and len(patterns) < limit:
                patterns.append(current)
            current = {"name": line[3:].strip(), "description": ""}
        elif current and line.strip() and not line.startswith("#"):
            if not current["description"]:
                current["description"] = line.strip()
    if current and len(patterns) < limit:
        patterns.append(current)
    return patterns


def main() -> None:
    payload = json.load(sys.stdin)
    cwd = payload.get("cwd", "")
    session_id = payload.get("session_id", "")

    config = _load_config()
    projects = config.get("projects", [])

    current = _detect_current_project(cwd, projects)
    related = _find_related_projects(current, projects)

    context_sections = []
    warnings = []

    if current:
        proj_path = Path(str(current.get("path", ""))).expanduser()
        memory_dir = proj_path / ".memory"
        if memory_dir.exists():
            decisions = _extract_key_decisions(memory_dir)
            if decisions:
                context_sections.append({
                    "title": f"Key Decisions ({current['name']})",
                    "items": decisions,
                })
            patterns = _extract_patterns(memory_dir)
            if patterns:
                context_sections.append({
                    "title": f"Patterns ({current['name']})",
                    "items": patterns,
                })

    for rel in related:
        proj_path = Path(str(rel.get("path", ""))).expanduser()
        memory_dir = proj_path / ".memory"
        if not memory_dir.exists():
            continue
        decisions = _extract_key_decisions(memory_dir, limit=2)
        if decisions:
            context_sections.append({
                "title": f"Related Decisions ({rel['name']})",
                "items": decisions,
                "shared_tags": list(set(current.get("tags", [])) & set(rel.get("tags", []))) if current else [],
            })

    result = {
        "plugin": "cross-project-knowledge-bridge",
        "session_id": session_id,
        "current_project": current.get("name") if current else None,
        "related_projects": [r.get("name") for r in related],
        "context_sections": context_sections,
    }

    if warnings:
        result["warnings"] = warnings

    print(json.dumps(result))


if __name__ == "__main__":
    main()
