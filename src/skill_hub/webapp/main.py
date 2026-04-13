"""FastAPI app factory for skill-hub webapp."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .routes import dashboard as dashboard_routes
from .routes import intents as intents_routes
from .routes import logs as logs_routes
from .routes import questions as questions_routes
from .routes import settings as settings_routes
from .routes import skills as skills_routes
from .routes import tasks as tasks_routes
from .routes import teachings as teachings_routes
from .routes import vector as vector_routes
from .routes import verdicts as verdicts_routes

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
    app.include_router(settings_routes.router)
    app.include_router(verdicts_routes.router)
    app.include_router(tasks_routes.router)
    app.include_router(skills_routes.router)
    app.include_router(teachings_routes.router)
    app.include_router(logs_routes.router)
    app.include_router(vector_routes.router)
    app.include_router(intents_routes.router)
    app.include_router(questions_routes.router)
    return app
