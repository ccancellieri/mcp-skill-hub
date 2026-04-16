"""FastAPI sub-app for the /memory-export mount.

``get_app()`` returns a FastAPI instance that mcp-skill-hub mounts under the
path declared in ``plugin.json -> web_mount.mount`` (``/memory-export``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from jinja2 import FileSystemLoader
from starlette.templating import Jinja2Templates

HERE = Path(__file__).parent
PLUGIN_ROOT = HERE.parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

# Make ``memexp`` (sibling to ``web/``) importable via absolute path. We use a
# unique top-level name to avoid colliding with other plugins on sys.path.
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

_DEFAULT_SHARED = (
    "/Users/ccancellieri/work/code/mcp-skill-hub/src/skill_hub/webapp/templates"
)


def _build_templates() -> Jinja2Templates:
    """Templates with the plugin dir first, then the shared skill-hub dir.

    Two-element ``FileSystemLoader`` keeps the cache key hashable (vs.
    nesting a ``ChoiceLoader``, which broke template lookups in starlette).
    """
    shared = os.environ.get("SKILL_HUB_SHARED_TEMPLATES", _DEFAULT_SHARED)
    search_paths = [str(TEMPLATES_DIR)]
    if shared and Path(shared).exists():
        search_paths.append(shared)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    templates.env.loader = FileSystemLoader(search_paths)
    return templates


def get_app() -> FastAPI:
    """Return the mounted FastAPI sub-app."""
    app = FastAPI(
        title="Memory Export",
        description="Export and import the full mcp-skill-hub memory",
        docs_url="/docs",
    )
    templates = _build_templates()
    app.state.templates = templates
    app.state.plugin_root = PLUGIN_ROOT

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    from .routes import api as api_routes
    from .routes import page as page_routes

    app.include_router(page_routes.router)
    app.include_router(api_routes.router)

    return app
