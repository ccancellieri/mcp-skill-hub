#!/usr/bin/env python3
"""Extract decisions from .memory/decisions.md → wiki entities.

Standalone script that can be run to:
1. Parse decisions.md format (YAML frontmatter + markdown)
2. Create wiki entity pages with proper metadata
3. Link to related projects and patterns

Usage:
    python scripts/extract_decisions.py [project_name]

If project_name is given, only that project is processed.
Otherwise, all configured projects are processed.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import date
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


def _slugify(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "page"


def _stable_id(source: str) -> str:
    digest = hashlib.sha1(source.encode()).hexdigest()
    return f"dec-{digest[:12]}"


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text

    rest = text[3:]
    nl = rest.find("\n")
    if nl == -1:
        return {}, text
    if rest[:nl].strip():
        return {}, text
    rest = rest[nl + 1:]

    close = rest.find("\n---")
    if close == -1:
        return {}, text
    fm_text = rest[:close]
    after = rest[close + 4:]
    if after.startswith("\n"):
        after = after[1:]

    try:
        import yaml
        fm = yaml.safe_load(fm_text) or {}
    except Exception:
        return {}, text

    return fm if isinstance(fm, dict) else {}, after


def extract_decisions(content: str, source_path: str) -> list[dict]:
    fm, body = _parse_frontmatter(content)
    decisions = []
    current = None

    for line in body.splitlines():
        heading = re.match(r"^#+\s+(.+)$", line)
        if heading:
            if current:
                decisions.append(current)
            title = heading.group(1).strip()
            current = {
                "id": _stable_id(f"{source_path}:{title}"),
                "slug": _slugify(title),
                "title": title,
                "body": "",
                "source_path": source_path,
            }
        elif current:
            current["body"] += line + "\n"

    if current:
        decisions.append(current)

    for dec in decisions:
        dec["body"] = dec["body"].strip()

    return decisions


def render_wiki_entity(decision: dict, project: str, tags: list[str]) -> str:
    today = date.today().isoformat()
    tags_str = ", ".join(json.dumps(t) for t in tags)

    return f"""---
id: {decision['id']}
slug: {decision['slug']}
title: "{decision['title']}"
type: entity
projects: ["{project}"]
scope: public
created: "{today}"
updated: "{today}"
source_refs:
  - "{decision['source_path']}"
tags: [{tags_str}]
---

# {decision['title']}

{decision['body']}
"""


def main() -> None:
    config = _load_config()
    projects = config.get("projects", [])
    filter_project = sys.argv[1] if len(sys.argv) > 1 else None

    for proj in projects:
        name = proj.get("name", "unknown")
        if filter_project and name != filter_project:
            continue

        path = Path(str(proj.get("path", ""))).expanduser()
        tags = proj.get("tags", [])

        if not path.exists():
            print(f"Project {name}: path not found ({path})")
            continue

        memory_dir = path / ".memory"
        decisions_file = memory_dir / "decisions.md"

        if not decisions_file.exists():
            print(f"Project {name}: no decisions.md")
            continue

        try:
            content = decisions_file.read_text(encoding="utf-8")
        except OSError as e:
            print(f"Project {name}: error reading decisions.md: {e}")
            continue

        decisions = extract_decisions(content, str(decisions_file))
        print(f"Project {name}: extracted {len(decisions)} decisions")

        for dec in decisions:
            print(f"  - [{dec['slug']}] {dec['title']}")


if __name__ == "__main__":
    main()
