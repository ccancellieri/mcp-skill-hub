"""REST endpoints: stats, generate, skills list."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse

router = APIRouter()

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT.parent.parent / "src"))


def _get_hub_db_path() -> Path:
    from skill_hub import config as _cfg
    return Path(_cfg.get("db_path") or Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db")


def _get_wiki_root() -> Path:
    from skill_hub import config as _cfg
    return Path(_cfg.get("wiki_root") or Path.home() / ".claude" / "mcp-skill-hub" / "wiki")


def _load_plugin_config() -> dict[str, Any]:
    plugin_json = _PLUGIN_ROOT / "plugin.json"
    try:
        return json.loads(plugin_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


@router.get("/api/stats")
def get_stats():
    """Return stats: wiki pages count, generated skills count, pending candidates."""
    db_path = _get_hub_db_path()
    wiki_root = _get_wiki_root()
    config = _load_plugin_config()
    plugin_config = config.get("config", {})
    min_access = plugin_config.get("min_access_count", 3)
    exclude_types = plugin_config.get("exclude_types", ["source", "overview"])

    wiki_pages_count = 0
    generated_skills_count = 0
    pending_candidates = 0

    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row

                # Count wiki pages
                wiki_pages_row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM wiki_pages"
                ).fetchone()
                wiki_pages_count = wiki_pages_row["cnt"] if wiki_pages_row else 0

                # Count generated skills
                skills_row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM plugin_wiki_skills"
                ).fetchone()
                generated_skills_count = skills_row["cnt"] if skills_row else 0

                # Count pending candidates (high access, not yet generated)
                placeholders = ",".join("?" * len(exclude_types))
                pending_row = conn.execute(
                    f"""
                    SELECT COUNT(*) as cnt
                    FROM vectors v
                    LEFT JOIN plugin_wiki_skills s ON v.doc_id = s.wiki_slug
                    LEFT JOIN wiki_pages p ON v.doc_id = p.slug
                    WHERE v.namespace IN ('wiki', 'wiki-private')
                      AND v.access_count >= ?
                      AND s.id IS NULL
                      AND p.type NOT IN ({placeholders})
                    """,
                    (min_access, *exclude_types),
                ).fetchone()
                pending_candidates = pending_row["cnt"] if pending_row else 0
        except sqlite3.Error:
            pass

    return JSONResponse({
        "wiki_pages_count": wiki_pages_count,
        "generated_skills_count": generated_skills_count,
        "pending_candidates": pending_candidates,
        "min_access_threshold": min_access,
    })


@router.get("/api/skills")
def list_skills(limit: int = Query(50, ge=1, le=500)):
    """List generated skills with their source wiki pages."""
    db_path = _get_hub_db_path()
    skills = []

    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT id, wiki_slug, wiki_title, wiki_type, skill_path,
                           skill_id, access_count, generated_at, last_used, use_count
                    FROM plugin_wiki_skills
                    ORDER BY generated_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
                skills = [dict(r) for r in rows]
        except sqlite3.Error:
            pass

    return JSONResponse({"skills": skills})


@router.get("/api/candidates")
def list_candidates(limit: int = Query(20, ge=1, le=100)):
    """List wiki pages with high access_count that haven't been converted to skills yet."""
    db_path = _get_hub_db_path()
    config = _load_plugin_config()
    plugin_config = config.get("config", {})
    min_access = plugin_config.get("min_access_count", 3)
    exclude_types = plugin_config.get("exclude_types", ["source", "overview"])
    candidates = []

    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as conn:
                conn.row_factory = sqlite3.Row
                placeholders = ",".join("?" * len(exclude_types))
                rows = conn.execute(
                    f"""
                    SELECT v.doc_id as slug, p.title, p.type, v.access_count
                    FROM vectors v
                    LEFT JOIN plugin_wiki_skills s ON v.doc_id = s.wiki_slug
                    LEFT JOIN wiki_pages p ON v.doc_id = p.slug
                    WHERE v.namespace IN ('wiki', 'wiki-private')
                      AND v.access_count >= ?
                      AND s.id IS NULL
                      AND p.type NOT IN ({placeholders})
                    ORDER BY v.access_count DESC
                    LIMIT ?
                    """,
                    (min_access, *exclude_types, limit),
                ).fetchall()
                candidates = [dict(r) for r in rows]
        except sqlite3.Error:
            pass

    return JSONResponse({"candidates": candidates})


@router.post("/api/generate")
def generate_skills(dry_run: bool = Query(False)):
    """Trigger skill generation, return results."""
    try:
        from scripts.generate_skills import main as generate_main
        result = generate_main(dry_run=dry_run)
        return JSONResponse({"status": "completed", "output": result})
    except Exception as exc:
        raise HTTPException(500, f"Generation failed: {exc}") from exc
