#!/usr/bin/env python3
"""Bidirectional wiki sync between projects.

Scans configured project .memory/ directories and syncs:
- decisions.md → wiki entity pages
- patterns.md → wiki concept pages
- Cross-references between related projects

Creates wiki pages under the shared wiki root with proper edges.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

PLUGIN_ROOT = Path(__file__).resolve().parent.parent

_WIKI_FRONTMATTER = """---
id: {id}
slug: {slug}
title: "{title}"
type: {type}
projects: [{projects}]
scope: public
created: "{created}"
updated: "{updated}"
source_refs:
  - "{source_ref}"
tags: [{tags}]
---

{body}
"""


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


def _stable_id(source_path: Path) -> str:
    digest = hashlib.sha1(str(source_path).encode()).hexdigest()
    return f"cpkb-{digest[:12]}"


def _parse_decisions_md(content: str) -> list[dict]:
    decisions = []
    current = None
    current_level = 0

    for line in content.splitlines():
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            level = len(heading.group(1))
            title = heading.group(2).strip()
            if level == 1:
                if current:
                    decisions.append(current)
                current = {"title": title, "body": "", "level": level}
            elif current and level > 1:
                current["body"] += line + "\n"
        elif current:
            current["body"] += line + "\n"

    if current:
        decisions.append(current)
    return decisions


def _parse_patterns_md(content: str) -> list[dict]:
    patterns = []
    current = None

    for line in content.splitlines():
        heading = re.match(r"^##\s+(.+)$", line)
        if heading:
            if current:
                patterns.append(current)
            current = {"name": heading.group(1).strip(), "body": ""}
        elif current:
            current["body"] += line + "\n"

    if current:
        patterns.append(current)
    return patterns


def _write_wiki_page(
    wiki_root: Path,
    slug: str,
    title: str,
    page_type: str,
    projects: list[str],
    tags: list[str],
    body: str,
    source_ref: str,
) -> str:
    page_id = _stable_id(Path(source_ref + "/" + slug))
    today = date.today().isoformat()

    projects_str = ", ".join(json.dumps(p) for p in projects)
    tags_str = ", ".join(json.dumps(t) for t in tags)

    content = _WIKI_FRONTMATTER.format(
        id=page_id,
        slug=slug,
        title=title.replace('"', '\\"'),
        type=page_type,
        projects=projects_str,
        created=today,
        updated=today,
        source_ref=source_ref,
        tags=tags_str,
        body=body.strip(),
    )

    page_dir = wiki_root / "pages" / page_type
    page_dir.mkdir(parents=True, exist_ok=True)
    page_path = page_dir / f"{slug}.md"

    existing_hash = ""
    if page_path.exists():
        try:
            existing = page_path.read_text(encoding="utf-8")
            existing_hash = hashlib.sha256(existing.encode()).hexdigest()[:16]
        except OSError:
            pass

    new_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    if new_hash == existing_hash:
        return ""

    page_path.write_text(content, encoding="utf-8")
    return slug


def sync_project(
    project: dict,
    wiki_root: Path,
) -> dict:
    proj_name = project.get("name", "unknown")
    proj_path = Path(str(project.get("path", ""))).expanduser()
    tags = project.get("tags", [])

    if not proj_path.exists():
        return {"project": proj_name, "status": "missing", "pages_created": 0}

    memory_dir = proj_path / ".memory"
    if not memory_dir.exists():
        return {"project": proj_name, "status": "no_memory", "pages_created": 0}

    pages_created = 0
    errors = []

    decisions_file = memory_dir / "decisions.md"
    if decisions_file.exists():
        try:
            content = decisions_file.read_text(encoding="utf-8")
            decisions = _parse_decisions_md(content)
            for dec in decisions:
                slug = _slugify(f"{proj_name}-decision-{dec['title']}")
                created = _write_wiki_page(
                    wiki_root=wiki_root,
                    slug=slug,
                    title=dec["title"],
                    page_type="entity",
                    projects=[proj_name],
                    tags=tags,
                    body=dec["body"],
                    source_ref=str(decisions_file),
                )
                if created:
                    pages_created += 1
        except OSError as e:
            errors.append(f"decisions.md: {e}")

    patterns_file = memory_dir / "patterns.md"
    if patterns_file.exists():
        try:
            content = patterns_file.read_text(encoding="utf-8")
            patterns = _parse_patterns_md(content)
            for pat in patterns:
                slug = _slugify(f"{proj_name}-pattern-{pat['name']}")
                created = _write_wiki_page(
                    wiki_root=wiki_root,
                    slug=slug,
                    title=pat["name"],
                    page_type="concept",
                    projects=[proj_name],
                    tags=tags,
                    body=pat["body"],
                    source_ref=str(patterns_file),
                )
                if created:
                    pages_created += 1
        except OSError as e:
            errors.append(f"patterns.md: {e}")

    return {
        "project": proj_name,
        "status": "synced",
        "pages_created": pages_created,
        "errors": errors,
    }


def create_cross_links(
    projects: list[dict],
    wiki_root: Path,
) -> int:
    links_created = 0
    tag_projects: dict[str, list[str]] = {}

    for proj in projects:
        for tag in proj.get("tags", []):
            tag_projects.setdefault(tag, []).append(proj["name"])

    for tag, proj_names in tag_projects.items():
        if len(proj_names) < 2:
            continue

        slug = _slugify(f"shared-{tag}")
        title = f"Shared: {tag}"
        body = f"# {title}\n\nProjects sharing the **{tag}** tag:\n\n"
        for proj_name in proj_names:
            body += f"- [[{proj_name}]]\n"

        created = _write_wiki_page(
            wiki_root=wiki_root,
            slug=slug,
            title=title,
            page_type="concept",
            projects=proj_names,
            tags=[tag],
            body=body,
            source_ref=f"tag:{tag}",
        )
        if created:
            links_created += 1

    return links_created


def main() -> None:
    config = _load_config()
    projects = config.get("projects", [])
    wiki_roots = config.get("wiki_roots", {})

    shared_wiki = Path(str(wiki_roots.get("shared", "~/.claude/mcp-skill-hub/wiki"))).expanduser()
    shared_wiki.mkdir(parents=True, exist_ok=True)

    results = []
    total_pages = 0

    for proj in projects:
        result = sync_project(proj, shared_wiki)
        results.append(result)
        total_pages += result.get("pages_created", 0)

    cross_links = create_cross_links(projects, shared_wiki)

    summary = {
        "status": "complete",
        "projects_synced": len([r for r in results if r["status"] == "synced"]),
        "pages_created": total_pages,
        "cross_links_created": cross_links,
        "wiki_root": str(shared_wiki),
        "details": results,
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
