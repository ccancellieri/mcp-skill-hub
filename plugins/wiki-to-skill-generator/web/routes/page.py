"""``GET /`` — overview page with stats and generated skills table."""
from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/")
def index(request: Request):
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "page.html",
        {"active_tab": "wiki-skill-generator"},
    )
