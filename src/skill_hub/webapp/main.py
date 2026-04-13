"""FastAPI app factory for skill-hub webapp."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .routes import dashboard as dashboard_routes

PKG_DIR = Path(__file__).resolve().parent
STATIC_DIR = PKG_DIR / "static"
TEMPLATES_DIR = PKG_DIR / "templates"


def create_app(store: Any) -> FastAPI:
    """Build the FastAPI app bound to the given SkillStore."""
    app = FastAPI(title="skill-hub control suite", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.store = store
    app.state.templates = templates

    app.mount(
        "/static", StaticFiles(directory=str(STATIC_DIR)), name="static"
    )

    @app.get("/healthz")
    def healthz() -> dict:
        return {"ok": True}

    app.include_router(dashboard_routes.router)
    return app
