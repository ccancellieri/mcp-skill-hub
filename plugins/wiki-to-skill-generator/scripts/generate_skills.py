#!/usr/bin/env python3
"""Generate project-specialized skills from the memory of work done.

Source: the ``memory:user-project`` vector namespace — the per-project
auto-memory (lessons, decisions, patterns) captured under
``~/.claude/projects/<project>/memory/*.md``. Each entry is attributed to its
originating project, so skills come out *specialized per project* rather than
generic.

Pipeline (uses the LLM escalation ladder — L1/L2/L3 = cheap/mid/smart tiers):

  Stage 0  Rearrange upfront   reindex memory, group by project, dedup in RAM
                               (never mutates the stored memory files)
  Stage 1  L1  cheap           triage + cluster a project's lessons into themes,
                               keep only skill-worthy ones
  Stage 2  L2  mid             draft a specialized skill per surviving cluster
  Stage 3  L3  smart           author a proper SKILL.md following the local
                               skill-authoring skill's conventions

Usage:
    python scripts/generate_skills.py [--dry-run] [--project SUBSTR] [--min-access N]

``--dry-run`` runs Stage 0 + L1 only (cheap) and prints the plan without
writing files or spending mid/smart-tier tokens.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

_PLUGINDIR = Path(__file__).resolve().parent.parent
import sys

sys.path.insert(0, str(_PLUGINDIR.parent.parent / "src"))

_log = logging.getLogger(__name__)

_MEMORY_NS = "memory:user-project"
# Index files are pointers, not lessons — never turn them into a skill.
_SKIP_FILENAMES = {"MEMORY.md", "index.md"}
_MAX_ENTRIES_PER_PROJECT = 40   # cap L1 input size
_MAX_SNIPPET = 240              # chars of each entry shown to L1
_MAX_CLUSTER_BODY = 6000       # chars of concatenated bodies shown to L2


def _load_plugin_config() -> dict[str, Any]:
    plugin_json = _PLUGINDIR / "plugin.json"
    try:
        return json.loads(plugin_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _parse_json(text: str) -> Any:
    """Extract the first JSON object/array from an LLM response (fence-tolerant)."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _request_json(request: Any, tiers: tuple[str, ...], prompt: str,
                  op: str, max_tokens: int) -> Any:
    """Call the ladder, escalating tiers until the reply parses as JSON.

    Weak cheap-tier models often ignore "return JSON" and emit prose. Rather
    than lose the result, escalate L1 -> L2 -> L3 until one tier produces
    parseable JSON. Returns the parsed value, or None if every tier failed.
    """
    strict = prompt + (
        "\n\nRespond with ONLY the JSON value. No preamble, no explanation, "
        "no markdown fences."
    )
    for tier in tiers:
        parsed = _parse_json(request(tier, strict, op=op,
                                     max_tokens=max_tokens, temperature=0.2))
        if parsed is not None:
            return parsed
        _log.info("%s: %s-tier reply unparseable, escalating", op, tier)
    return None


def _strip_frontmatter(content: str) -> str:
    if content.startswith("---"):
        end = content.find("\n---\n", 4)
        if end != -1:
            return content[end + 5:].strip()
    return content.strip()


# --------------------------------------------------------------------------- #
# Stage 0 — gather + rearrange (safe, RAM-only)
# --------------------------------------------------------------------------- #
def _access_by_path(store: Any) -> dict[str, int]:
    """Best-effort map of memory file path -> stored access_count (no embedding)."""
    try:
        rows = store._conn.execute(
            """
            SELECT json_extract(metadata, '$.path') AS path, MAX(access_count) AS ac
            FROM vectors WHERE namespace = ? AND metadata IS NOT NULL GROUP BY path
            """,
            (_MEMORY_NS,),
        ).fetchall()
        return {r["path"]: (r["ac"] or 0) for r in rows if r["path"]}
    except Exception:  # noqa: BLE001
        return {}


def _gather_memory_by_project(
    store: Any, min_access: int, project_filter: str | None, reindex: bool = False
) -> dict[str, list[dict]]:
    """Group memory-of-work-done files by project (working copy only).

    Reads memory files straight from disk via ``iter_user_memory_files`` — no
    embedding, so it stays fast — skips index/empty files, and drops
    exact-content duplicates within a project. Never mutates the stored memory
    files. ``reindex=True`` refreshes embeddings/access first (slow: loads the
    embedding model); off by default so the dashboard button stays responsive.
    """
    from skill_hub.memory_index import iter_user_memory_files

    if reindex:
        try:
            from skill_hub.memory_index import index_user_memory
            index_user_memory(store)
        except Exception as exc:  # noqa: BLE001
            _log.warning("memory reindex skipped: %s", exc)

    access = _access_by_path(store) if min_access > 0 else {}
    files = iter_user_memory_files()

    by_project: dict[str, list[dict]] = {}
    seen: dict[str, set[str]] = {}
    for f in files:
        if f.name in _SKIP_FILENAMES:
            continue
        # Same attribution as index_user_memory: the project dir name.
        project = f.parents[1].name if len(f.parents) >= 2 else "user"
        if project_filter and project_filter.lower() not in project.lower():
            continue
        ac = access.get(str(f), 0)
        if ac < min_access:
            continue
        try:
            body = _strip_frontmatter(f.read_text(encoding="utf-8"))
        except OSError:
            continue
        if not body:
            continue
        dupes = seen.setdefault(project, set())
        fingerprint = body[:200]
        if fingerprint in dupes:
            continue
        dupes.add(fingerprint)
        entries = by_project.setdefault(project, [])
        if len(entries) >= _MAX_ENTRIES_PER_PROJECT:
            continue
        entries.append({"title": f.stem, "body": body, "access_count": ac})
    return by_project


# --------------------------------------------------------------------------- #
# Stage 1 — L1 (cheap): triage + cluster
# --------------------------------------------------------------------------- #
def _triage_cluster(request: Any, project: str, entries: list[dict]) -> list[dict]:
    """Cluster a project's lessons into skill-worthy themes (cheap tier)."""
    listing = "\n".join(
        f"[{i}] {e['title']} :: {e['body'][:_MAX_SNIPPET].replace(chr(10), ' ')}"
        for i, e in enumerate(entries)
    )
    prompt = f"""Triage the accumulated memory of project "{project}" to decide which \
clusters of lessons deserve a dedicated, specialized skill.

ENTRIES (id :: title :: snippet):
{listing}

Group related entries into coherent themes. A theme is skill-worthy only when its \
entries encode concrete, reusable practice specific to THIS project — not generic advice.

Return ONLY JSON:
{{"clusters": [{{"theme": "short name", "entry_ids": [0, 2], "skill_worthy": true, "reason": "why"}}]}}"""
    parsed = _request_json(request, ("cheap", "mid"), prompt,
                           "skill_gen_triage", 1200)
    clusters = (parsed or {}).get("clusters", []) if isinstance(parsed, dict) else []
    out = []
    for c in clusters:
        if not c.get("skill_worthy"):
            continue
        ids = [i for i in c.get("entry_ids", []) if isinstance(i, int) and 0 <= i < len(entries)]
        if not ids:
            continue
        out.append({"theme": c.get("theme", "lessons"), "entry_ids": ids,
                    "reason": c.get("reason", "")})
    return out


# --------------------------------------------------------------------------- #
# Stage 2 — L2 (mid): draft a specialized skill
# --------------------------------------------------------------------------- #
def _draft_skill(request: Any, project: str, theme: str, entries: list[dict]) -> dict | None:
    """Draft a project-specialized skill from a cluster's lessons (mid tier)."""
    body = "\n\n---\n\n".join(f"## {e['title']}\n{e['body']}" for e in entries)
    body = body[:_MAX_CLUSTER_BODY]
    prompt = f"""Draft a specialized skill for project "{project}", theme "{theme}", from \
its own memory of work done:

{body}

The skill MUST be specialized to "{project}": keep the concrete lesson content — the \
names, paths, decisions, and gotchas. Do NOT generalize into vague advice.

Return ONLY JSON:
{{"skill_name": "kebab-case", "title": "human title", "description": "one line naming the \
project/domain", "triggers": ["phrases someone working on this project would say"], \
"body": "markdown body: Overview / When to use / Procedure / Gotchas — concrete and project-specific"}}"""
    draft = _request_json(request, ("mid", "smart"), prompt,
                          "skill_gen_draft", 1600)
    if not isinstance(draft, dict) or not draft.get("skill_name"):
        _log.warning("draft failed for %s / %s", project, theme)
        return None
    return draft


# --------------------------------------------------------------------------- #
# Stage 3 — L3 (smart): author a proper SKILL.md
# --------------------------------------------------------------------------- #
def _load_authoring_guidance() -> str:
    """Load the local skill-authoring skill so L3 follows its conventions."""
    factory = _PLUGINDIR / "skills" / "wiki-skill-factory" / "SKILL.md"
    try:
        return _strip_frontmatter(factory.read_text(encoding="utf-8"))[:3000]
    except OSError:
        return (
            "A proper skill has a one-line trigger-optimized description, clear "
            "triggers, a focused procedure, and concrete examples/gotchas."
        )


def _author_skill(request: Any, project: str, draft: dict, guidance: str) -> dict | None:
    """Polish a draft into a proper, specialized SKILL.md (smart tier)."""
    prompt = f"""{guidance}

Using the authoring guidance above, polish this DRAFT into a proper skill. Keep it \
SPECIALIZED to project "{project}" — concrete, not generic. Tighten the description so \
triggering is accurate, sharpen the triggers, and improve the body's structure.

DRAFT (JSON):
{json.dumps(draft, ensure_ascii=False)}

Return ONLY JSON:
{{"skill_name": "kebab-case", "description": "one-line trigger-optimized description", \
"triggers": ["..."], "body": "final markdown body"}}"""
    final = _request_json(request, ("smart",), prompt,
                          "skill_gen_author", 2600)
    if not isinstance(final, dict) or not final.get("body"):
        _log.warning("author failed for %s / %s — using draft", project, draft.get("skill_name"))
        final = {
            "skill_name": draft.get("skill_name"),
            "description": draft.get("description", ""),
            "triggers": draft.get("triggers", []),
            "body": draft.get("body", ""),
        }
    final.setdefault("skill_name", draft.get("skill_name"))
    return final


# --------------------------------------------------------------------------- #
# Write + register
# --------------------------------------------------------------------------- #
def _project_slug(project: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", project.lower()).strip("-")
    return slug or "user"


def _render_skill_md(skill: dict, project: str) -> str:
    triggers = "".join(f'\n  - "{t}"' for t in skill.get("triggers", []))
    ts = datetime.utcnow().isoformat() + "Z"
    return (
        "---\n"
        f"name: {skill['skill_name']}\n"
        f"description: {skill.get('description', '')}\n"
        f"triggers:{triggers}\n"
        "source: memory-generated\n"
        f"project: {project}\n"
        f"generated_at: {ts}\n"
        "---\n\n"
        f"{skill.get('body', '').strip()}\n"
    )


def _register_skill(store: Any, project: str, skill: dict, skill_path: Path,
                    access_count: int) -> None:
    try:
        rel = str(skill_path.relative_to(_PLUGINDIR))
    except ValueError:
        rel = str(skill_path)
    db = store.plugin_db("wiki")
    db.execute(
        """
        INSERT OR REPLACE INTO plugin_wiki_skills
            (wiki_slug, wiki_title, wiki_type, skill_path, skill_id, access_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            f"{project}/{skill['skill_name']}",
            skill.get("description", skill["skill_name"]),
            f"memory:{project}",
            rel,
            skill["skill_name"],
            access_count,
        ),
    )


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def main(dry_run: bool = False, min_access: int = 0,
         project: str | None = None, reindex: bool = False) -> str:
    """Generate per-project specialized skills from the memory of work done."""
    from skill_hub.llm.request import request
    from skill_hub.store import SkillStore

    config = _load_plugin_config().get("config", {})
    output_root = _PLUGINDIR / config.get("skill_output_dir", "skills/generated")

    store = SkillStore()
    by_project = _gather_memory_by_project(store, min_access, project, reindex)
    if not by_project:
        store.close()
        return "No project memory found to build skills from."

    guidance = _load_authoring_guidance()
    lines = [f"Projects with memory: {len(by_project)}"]
    generated = 0

    for proj, entries in by_project.items():
        clusters = _triage_cluster(request, proj, entries)   # L1
        if not clusters:
            lines.append(f"\n{proj}: no skill-worthy clusters")
            continue
        lines.append(f"\n{proj}: {len(clusters)} skill-worthy cluster(s)")

        for cluster in clusters:
            picked = [entries[i] for i in cluster["entry_ids"]]
            access = max((e["access_count"] for e in picked), default=0)
            if dry_run:
                lines.append(f"  - {cluster['theme']} (dry-run) — {cluster['reason']}")
                continue

            draft = _draft_skill(request, proj, cluster["theme"], picked)   # L2
            if not draft:
                lines.append(f"  - {cluster['theme']}: draft failed, skipped")
                continue
            skill = _author_skill(request, proj, draft, guidance)           # L3
            if not skill or not skill.get("skill_name"):
                lines.append(f"  - {cluster['theme']}: author failed, skipped")
                continue

            proj_dir = output_root / _project_slug(proj) / skill["skill_name"]
            proj_dir.mkdir(parents=True, exist_ok=True)
            skill_path = proj_dir / "SKILL.md"
            skill_path.write_text(_render_skill_md(skill, proj), encoding="utf-8")
            try:
                _register_skill(store, proj, skill, skill_path, access)
            except Exception as exc:  # noqa: BLE001 — a tracking hiccup must not lose the skill
                _log.warning("register failed for %s: %s", skill["skill_name"], exc)
            generated += 1
            lines.append(f"  - {cluster['theme']} -> {skill['skill_name']}")

    store.close()
    lines.append(f"\nSkills generated: {generated}" if not dry_run
                 else "\n(dry-run: L1 triage only, no files written)")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate project-specialized skills from the memory of work done")
    parser.add_argument("--dry-run", action="store_true",
                        help="L1 triage only: preview the plan, write nothing")
    parser.add_argument("--project", type=str, default=None,
                        help="Only build skills for projects matching this substring")
    parser.add_argument("--min-access", type=int, default=0,
                        help="Minimum memory access_count threshold")
    parser.add_argument("--reindex", action="store_true",
                        help="Refresh memory embeddings/access first (slow)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    print(main(dry_run=args.dry_run, min_access=args.min_access,
               project=args.project, reindex=args.reindex))
