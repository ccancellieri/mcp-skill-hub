"""FastAPI sub-app for the /contradiction-detector mount.

Provides:
- GET / — Review pending contradictions
- POST /resolve — Mark a contradiction as resolved
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from jinja2 import FileSystemLoader
from starlette.templating import Jinja2Templates

HERE = Path(__file__).parent
PLUGIN_ROOT = HERE.parent
TEMPLATES_DIR = HERE / "templates"

if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

_DEFAULT_SHARED = (
    "/Users/ccancellieri/work/code/mcp-skill-hub/src/skill_hub/webapp/templates"
)


def _build_templates() -> Jinja2Templates:
    import os
    shared = os.environ.get("SKILL_HUB_SHARED_TEMPLATES", _DEFAULT_SHARED)
    search_paths = [str(TEMPLATES_DIR)]
    if shared and Path(shared).exists():
        search_paths.append(shared)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.loader = FileSystemLoader(search_paths)
    return templates


def get_app() -> FastAPI:
    app = FastAPI(
        title="Contradiction Detector",
        description="Review and resolve contradictory wiki claims",
        docs_url="/docs",
    )
    templates = _build_templates()
    app.state.templates = templates
    app.state.plugin_root = PLUGIN_ROOT

    from .routes import api as api_routes
    from .routes import page as page_routes

    app.include_router(page_routes.router)
    app.include_router(api_routes.router)

    return app
