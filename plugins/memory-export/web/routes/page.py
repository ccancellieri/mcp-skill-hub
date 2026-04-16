"""``GET /`` — overview page with project picker + per-target mode controls."""
from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Request

from memexp import scope, snapshot

router = APIRouter()


@router.get("/")
def index(request: Request):
    templates = request.app.state.templates

    projects = [p.to_dict() for p in scope.list_projects()]

    tables: list[str] = []
    db_path = snapshot.DEFAULT_DB_PATH
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            tables = scope.list_exportable_tables(conn)

    # LLM tiers come straight from config so the picker matches whatever the
    # user has configured (e.g. tier_cheap / tier_default / tier_smart).
    try:
        from skill_hub import config as _cfg

        providers = _cfg.get("llm_providers") or {}
        llm_tiers = sorted(providers.keys()) if isinstance(providers, dict) else []
    except Exception:  # noqa: BLE001
        llm_tiers = ["tier_cheap", "tier_default", "tier_smart"]

    return templates.TemplateResponse(
        request,
        "page.html",
        {
            "projects": projects,
            "tables": tables,
            "llm_tiers": llm_tiers,
            "modes": ["skip", "override", "llm"],
        },
    )
