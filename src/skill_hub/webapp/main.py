"""FastAPI app factory for skill-hub webapp."""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.util
import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import ChoiceLoader, FileSystemLoader

from .routes import dashboard as dashboard_routes
from .routes import intents as intents_routes
from .routes import logs as logs_routes
from .routes import questions as questions_routes
from .routes import settings as settings_routes
from .routes import skills as skills_routes
from .routes import task_logs as task_logs_routes
from .routes import tasks as tasks_routes
from .routes import teachings as teachings_routes
from .routes import vector as vector_routes
from .routes import verdicts as verdicts_routes

PKG_DIR = Path(__file__).resolve().parent
STATIC_DIR = PKG_DIR / "static"
TEMPLATES_DIR = PKG_DIR / "templates"
SHARED_MACROS_DIR = TEMPLATES_DIR / "_macros"

_log = logging.getLogger(__name__)

# Core (built-in) nav entries. Plugin web-mounts append to this via
# ``app.state.plugin_nav`` — see base.html.
_CORE_NAV: list[dict[str, Any]] = [
    {"key": "dashboard", "label": "Dashboard", "href": "/"},
    {"key": "settings", "label": "Settings", "href": "/settings"},
    {"key": "logs", "label": "Logs", "href": "/logs"},
    {"key": "verdicts", "label": "Verdicts", "href": "/verdicts"},
    {"key": "tasks", "label": "Tasks", "href": "/tasks"},
    {"key": "skills", "label": "Skills", "href": "/skills"},
    {"key": "teachings", "label": "Teachings", "href": "/teachings"},
    {"key": "vector", "label": "Vector", "href": "/vector"},
    {"key": "intents", "label": "Intents", "href": "/intents"},
    {"key": "questions", "label": "Questions", "href": "/questions"},
]


def _load_plugin_subapp(plugin_path: Path) -> FastAPI | None:
    """Import {plugin_path}/web/app.py and return ``get_app()``.

    Returns ``None`` (and logs) on any failure — plugins must never crash the
    host webapp.
    """
    web_dir = plugin_path / "web"
    app_py = web_dir / "app.py"
    if not app_py.exists():
        return None
    pkg_name = f"_skillhub_plugin_{plugin_path.name}_web"
    mod_name = f"{pkg_name}.app"
    try:
        # Register web/ as a package so relative imports (e.g. `from .routes import x`)
        # inside the plugin's web/app.py resolve correctly.
        pkg_init = web_dir / "__init__.py"
        if pkg_name not in sys.modules:
            if pkg_init.exists():
                pkg_spec = importlib.util.spec_from_file_location(
                    pkg_name, pkg_init,
                    submodule_search_locations=[str(web_dir)],
                )
            else:
                pkg_spec = importlib.machinery.ModuleSpec(
                    pkg_name, loader=None, is_package=True,
                )
                pkg_spec.submodule_search_locations = [str(web_dir)]
            if pkg_spec is None:
                return None
            pkg = importlib.util.module_from_spec(pkg_spec)
            sys.modules[pkg_name] = pkg
            if pkg_spec.loader is not None:
                pkg_spec.loader.exec_module(pkg)
        spec = importlib.util.spec_from_file_location(mod_name, app_py)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = pkg_name
        sys.modules[mod_name] = mod
        spec.loader.exec_module(mod)
        get_app = getattr(mod, "get_app", None)
        if get_app is None:
            _log.warning("plugin %s: web/app.py missing get_app()", plugin_path.name)
            return None
        sub_app = get_app()
        if not isinstance(sub_app, FastAPI):
            _log.warning("plugin %s: get_app() did not return FastAPI", plugin_path.name)
            return None
        # Plugin extension-point: A11 — inject shared macros dir into sub-app's
        # Jinja2 env so plugins can `{% from "_macros/kpi.html" import kpi_card %}`.
        _inject_shared_macros(sub_app)
        return sub_app
    except Exception as exc:  # noqa: BLE001
        _log.warning("plugin %s: failed to load web/app.py: %s", plugin_path.name, exc)
        return None


def _inject_shared_macros(sub_app: FastAPI) -> None:
    """Add the shared macros dir to the sub-app's Jinja loader search path.

    Plugin sub-apps that expose ``app.state.templates`` (a ``Jinja2Templates``)
    get their underlying environment's loader wrapped in a ``ChoiceLoader``
    that falls back to the skill-hub shared macros.
    """
    tpls = getattr(sub_app.state, "templates", None)
    if tpls is None:
        return
    try:
        env = tpls.env
        existing = env.loader
        shared = FileSystemLoader(str(TEMPLATES_DIR))  # exposes _macros/*.html
        if isinstance(existing, ChoiceLoader):
            env.loader = ChoiceLoader(list(existing.loaders) + [shared])
        elif existing is not None:
            env.loader = ChoiceLoader([existing, shared])
        else:
            env.loader = shared
    except Exception as exc:  # noqa: BLE001
        _log.debug("could not wire shared macros into plugin sub-app: %s", exc)


def create_app(store: Any) -> FastAPI:
    """Build the FastAPI app bound to the given SkillStore."""
    app = FastAPI(title="skill-hub control suite", docs_url=None, redoc_url=None)
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.store = store
    app.state.templates = templates
    # Plugin extension-point: A11 — expose the shared macros dir to plugin
    # sub-apps so `{% from "_macros/kpi.html" import kpi_card %}` works.
    app.state.shared_templates_dir = str(SHARED_MACROS_DIR)

    # Plugin extension-point: A1 — base.html iterates core_nav + plugin_nav.
    app.state.core_nav = list(_CORE_NAV)
    app.state.plugin_nav = []

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
    app.include_router(task_logs_routes.router)
    app.include_router(skills_routes.router)
    app.include_router(teachings_routes.router)
    app.include_router(logs_routes.router)
    app.include_router(vector_routes.router)
    app.include_router(intents_routes.router)
    app.include_router(questions_routes.router)

    # Plugin extension-point: A1 — mount plugin web sub-apps.
    # See docs/plugin-extension-points.md for plugin.json "web_mount" + the
    # optional "extra_web_mounts" config override.
    try:
        from ..plugin_registry import load_web_mounts
        for cfg in load_web_mounts():
            plugin_path = Path(cfg["plugin_path"])
            sub_app = _load_plugin_subapp(plugin_path)
            if sub_app is None:
                continue
            try:
                app.mount(cfg["mount"], sub_app, name=cfg["plugin_name"])
            except Exception as exc:  # noqa: BLE001
                _log.warning("mount %s failed: %s", cfg["mount"], exc)
                continue
            if cfg.get("nav"):
                app.state.plugin_nav.append({
                    "key": cfg["plugin_name"],
                    "label": cfg["title"],
                    "href": cfg["mount"],
                    "icon": cfg.get("icon"),
                })
    except Exception as exc:  # noqa: BLE001
        _log.warning("plugin mount discovery failed: %s", exc)

    return app
