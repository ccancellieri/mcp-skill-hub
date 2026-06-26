#!/usr/bin/env python3
"""Generate skills from wiki pages with high access counts.

This script:
1. Queries wiki pages with high access_count from the vectors table
2. Filters by type (excludes source/overview by default)
3. Uses LLM to extract actionable patterns from page content
4. Generates SKILL.md files using Jinja2 template
5. Registers mappings in plugin_wiki_skills table

Usage:
    python scripts/generate_skills.py [--dry-run] [--min-access N] [--types TYPE,...]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_PLUGINDIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PLUGINDIR.parent.parent / "src"))

_log = logging.getLogger(__name__)


def _load_plugin_config() -> dict[str, Any]:
    plugin_json = _PLUGINDIR / "plugin.json"
    try:
        return json.loads(plugin_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _get_wiki_pages_with_high_access(
    store: Any,
    min_access: int,
    exclude_types: list[str],
    wiki_root: Path,
) -> list[dict]:
    """Find wiki pages with high access_count that aren't source/overview."""
    conn = store._conn
    
    # Get wiki vectors with high access_count
    rows = conn.execute(
        """
        SELECT v.doc_id, v.metadata, v.access_count, v.level
        FROM vectors v
        WHERE v.namespace IN ('wiki', 'wiki-private')
          AND v.access_count >= ?
        ORDER BY v.access_count DESC
        """,
        (min_access,),
    ).fetchall()
    
    # Enrich with wiki_pages metadata
    pages = []
    for row in rows:
        slug = row["doc_id"]
        
        # Get page info from wiki_pages table
        page_row = conn.execute(
            "SELECT title, type, scope, projects, rel_path FROM wiki_pages WHERE slug = ?",
            (slug,),
        ).fetchone()
        
        if not page_row:
            continue
        
        if page_row["type"] in exclude_types:
            continue
        
        # Load full page body from disk
        page_path = wiki_root / page_row["rel_path"]
        try:
            content = page_path.read_text(encoding="utf-8")
            # Extract body (after frontmatter)
            if content.startswith("---"):
                fm_end = content.find("\n---\n", 4)
                if fm_end != -1:
                    body = content[fm_end + 5:].strip()
                else:
                    body = content
            else:
                body = content
        except OSError:
            continue
        
        pages.append({
            "slug": slug,
            "title": page_row["title"],
            "type": page_row["type"],
            "scope": page_row["scope"],
            "projects": json.loads(page_row["projects"]) if page_row["projects"] else [],
            "access_count": row["access_count"],
            "body": body,
        })
    
    return pages


def _extract_skill_from_page(
    llm: Any,
    page: dict,
    tier: str,
) -> dict[str, Any] | None:
    """Use LLM to extract actionable skill from wiki page content."""
    prompt = f"""Analyze this wiki page and extract an actionable skill.

WIKI PAGE: {page['title']} (type: {page['type']})
SLUG: {page['slug']}

CONTENT:
{page['body'][:4000]}

Extract a structured skill with these fields (JSON format):
{{
  "skill_name": "kebab-case-name-for-skill",
  "title": "Human readable title",
  "description": "One-line description of what this skill helps with",
  "triggers": ["phrase 1", "phrase 2", "phrase 3"],
  "overview": "Optional intro paragraph",
  "context": "Background context needed",
  "procedure": "Step-by-step instructions (markdown)",
  "examples": "Concrete examples (markdown)",
  "gotchas": "Common pitfalls or edge cases (markdown)"
}}

Guidelines:
- skill_name should be kebab-case, derived from the page topic
- triggers should be phrases a user might say that match this skill
- procedure should be actionable steps, not just theory
- If the page doesn't contain actionable patterns, return {{"skip": true, "reason": "..."}}
"""
    
    try:
        response = llm.complete(prompt, tier=tier, max_tokens=1500, temperature=0.3)
        result = json.loads(response)
        if result.get("skip"):
            _log.info("Skipping %s: %s", page["slug"], result.get("reason", "no actionable patterns"))
            return None
        return result
    except (json.JSONDecodeError, Exception) as exc:
        _log.warning("LLM extraction failed for %s: %s", page["slug"], exc)
        return None


def _render_skill(template_env: Any, skill_data: dict, page: dict) -> str:
    """Render skill markdown from template."""
    template = template_env.get_template("skill_template.md.jinja")
    return template.render(
        skill_name=skill_data.get("skill_name", page["slug"]),
        title=skill_data.get("title", page["title"]),
        description=skill_data.get("description", ""),
        triggers=skill_data.get("triggers", []),
        overview=skill_data.get("overview", ""),
        context=skill_data.get("context", ""),
        procedure=skill_data.get("procedure", ""),
        examples=skill_data.get("examples", ""),
        gotchas=skill_data.get("gotchas", ""),
        wiki_slug=page["slug"],
        generated_at=datetime.utcnow().isoformat() + "Z",
    )


def _register_skill(
    store: Any,
    page: dict,
    skill_path: Path,
    skill_data: dict,
) -> None:
    """Register skill mapping in plugin_wiki_skills table."""
    db = store.plugin_db("wiki_skills")
    db.execute(
        """
        INSERT OR REPLACE INTO plugin_wiki_skills
            (wiki_slug, wiki_title, wiki_type, skill_path, skill_id, access_count)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            page["slug"],
            page["title"],
            page["type"],
            str(skill_path.relative_to(_PLUGINDIR)),
            skill_data.get("skill_name"),
            page["access_count"],
        ),
    )


def main(dry_run: bool = False, min_access: int = 3, types: str | None = None) -> str:
    """Main generation logic."""
    from jinja2 import Environment, FileSystemLoader
    
    from skill_hub import config as _cfg
    from skill_hub.llm import get_provider
    from skill_hub.store import SkillStore
    
    config = _load_plugin_config()
    plugin_config = config.get("config", {})
    
    min_access = min_access or plugin_config.get("min_access_count", 3)
    exclude_types = plugin_config.get("exclude_types", ["source", "overview"])
    tier = plugin_config.get("llm_tier", "tier_mid")
    output_dir = _PLUGINDIR / plugin_config.get("skill_output_dir", "skills/generated/")
    
    if types:
        exclude_types = [t for t in ["entity", "concept", "source", "overview"] if t not in types.split(",")]
    
    wiki_root = Path(_cfg.get("wiki_root") or Path.home() / ".claude" / "mcp-skill-hub" / "wiki")
    
    store = SkillStore()
    llm = get_provider()
    template_env = Environment(loader=FileSystemLoader(_PLUGINDIR / "templates"))
    
    pages = _get_wiki_pages_with_high_access(store, min_access, exclude_types, wiki_root)
    
    if not pages:
        return f"No wiki pages found with access_count >= {min_access}"
    
    generated = []
    skipped = []
    
    for page in pages:
        skill_data = _extract_skill_from_page(llm, page, tier)
        if not skill_data:
            skipped.append(page["slug"])
            continue
        
        skill_name = skill_data.get("skill_name", page["slug"])
        skill_path = output_dir / f"{skill_name}.md"
        
        if dry_run:
            generated.append(f"{page['slug']} -> {skill_name} (dry-run)")
            continue
        
        output_dir.mkdir(parents=True, exist_ok=True)
        skill_md = _render_skill(template_env, skill_data, page)
        skill_path.write_text(skill_md, encoding="utf-8")
        
        _register_skill(store, page, skill_path, skill_data)
        
        generated.append(f"{page['slug']} -> {skill_name}")
    
    store.close()
    
    result = [f"Wiki pages scanned: {len(pages)}"]
    result.append(f"Skills generated: {len(generated)}")
    if generated:
        result.append("\nGenerated:")
        result.extend(f"  - {g}" for g in generated)
    if skipped:
        result.append(f"\nSkipped ({len(skipped)}): {', '.join(skipped)}")
    
    return "\n".join(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate skills from wiki pages")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--min-access", type=int, default=0, help="Minimum access count threshold")
    parser.add_argument("--types", type=str, help="Comma-separated page types to include")
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    
    output = main(dry_run=args.dry_run, min_access=args.min_access, types=args.types)
    print(output)
