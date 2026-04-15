"""Dashboard route — KPIs, metrics, and home page."""
from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ... import dashboard as _dashboard
from ... import dashboard_api  # noqa: F401 (plan reference; future use)
from ..services import intents_queue, questions_queue
from .router_page import _read_entries, _compute_stats


def _rss_mb() -> float | None:
    """Best-effort resident-set size of this process in MiB (stdlib only)."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS returns bytes; Linux returns kilobytes.
        if os.uname().sysname == "Darwin":
            return round(rss / (1024 * 1024), 1)
        return round(rss / 1024, 1)
    except Exception:  # noqa: BLE001
        return None

router = APIRouter()


def _collect_metrics(store: Any) -> dict[str, Any]:
    db = _dashboard._db_metrics(store)
    logm = _dashboard._parse_log()
    vcache = _dashboard._verdict_metrics()
    tokens_saved = int(db.get("tokens_saved") or 0)
    llm_seconds = sum(logm["llm_ms"]) / 1000.0
    llm_cost_eq = int(llm_seconds * _dashboard.TOKENS_PER_LLM_SECOND)
    net = tokens_saved - llm_cost_eq
    tasks_open = db["tasks"].get("open", 0) if isinstance(db.get("tasks"), dict) else 0
    tasks_closed = db["tasks"].get("closed", 0) if isinstance(db.get("tasks"), dict) else 0
    helpful = db.get("feedback_helpful", 0)
    unhelpful = db.get("feedback_unhelpful", 0)
    total_fb = helpful + unhelpful
    helpful_pct = (helpful / total_fb * 100.0) if total_fb else 0.0
    approve = logm["auto_approve"].get("allow", 0)
    deny = logm["auto_approve"].get("deny", 0)
    pass_through = logm["auto_approve"].get("pass", 0)
    return {
        "tokens_saved": tokens_saved,
        "llm_cost_eq": llm_cost_eq,
        "net": net,
        "tasks_open": tasks_open,
        "tasks_closed": tasks_closed,
        "skills": db.get("skills", 0),
        "teachings": db.get("teachings", 0),
        "helpful": helpful,
        "unhelpful": unhelpful,
        "helpful_pct": round(helpful_pct, 1),
        "approve": approve,
        "deny": deny,
        "pass_through": pass_through,
        "auto_proceed_fires": logm.get("auto_proceed_fires", 0),
        "resume_consumed": logm.get("resume_consumed", 0),
        "intercept_errors": logm.get("intercept_errors", 0),
        "llm_samples": len(logm["llm_ms"]),
        "verdict_total": vcache.get("total", 0),
        "verdict_hits": vcache.get("hits_total", 0),
        "log_missing": logm.get("log_missing", False),
    }


@router.get("/api/metrics")
def api_metrics(request: Request) -> JSONResponse:
    store = request.app.state.store
    return JSONResponse(_collect_metrics(store))


def _recent_open_tasks(store: Any, limit: int = 5) -> list[dict]:
    try:
        rows = store.list_tasks(status="open")
    except Exception:  # noqa: BLE001
        return []
    out = []
    for r in rows[:limit]:
        d = dict(r)
        out.append({
            "id": d.get("id"),
            "title": (d.get("title") or "")[:80],
            "tags": d.get("tags") or "",
        })
    return out


def _intercept_by_type(store: Any, limit: int = 5) -> list[dict]:
    try:
        rows = store.get_interception_stats()
    except Exception:  # noqa: BLE001
        return []
    out = []
    for r in rows[:limit]:
        out.append({
            "type": r["command_type"],
            "n": r["intercept_count"],
            "tokens": r["total_tokens_saved"] or 0,
        })
    return out


def collect_plugin_sections() -> list[dict]:
    """Plugin extension-point: A2 — gather dashboard sections from plugins.

    Each enabled plugin may ship ``web/dashboard_sections.py`` with a
    ``get_sections() -> list[dict]`` function. Each section is a dict with
    ``id``, ``title``, ``order`` plus either:
      - ``html``: pre-rendered HTML string (marked safe in template), or
      - ``template`` + ``context``: jinja template path (relative to plugin's
        ``web/templates/`` dir) and a context dict.

    When ``template`` is set, this function renders it using a Jinja2
    environment that includes both the plugin's own templates directory and
    the shared macros directory (so ``{% from "_macros/kpi.html" import … %}``
    works).  The rendered HTML is stored in ``html`` and ``template``/
    ``context`` are removed before returning.

    Returns sections sorted by ``order`` (default 100). Failures per-plugin
    are logged and skipped; never crash the dashboard.
    """
    import importlib.util
    import logging
    import sys
    from pathlib import Path

    from jinja2 import ChoiceLoader, Environment, FileSystemLoader

    log = logging.getLogger(__name__)
    # Shared macros dir — lives alongside this file's templates parent
    shared_templates = Path(__file__).parent.parent / "templates"

    sections: list[dict] = []
    try:
        from ...plugin_registry import iter_enabled_plugins
    except Exception:  # noqa: BLE001
        return sections

    for p in iter_enabled_plugins():
        ds_py = Path(p["path"]) / "web" / "dashboard_sections.py"
        if not ds_py.exists():
            continue
        mod_name = f"_skillhub_plugin_{p['path'].name}_dashboard_sections"
        plugin_templates = Path(p["path"]) / "web" / "templates"
        try:
            spec = importlib.util.spec_from_file_location(mod_name, ds_py)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = mod
            spec.loader.exec_module(mod)
            get_sections = getattr(mod, "get_sections", None)
            if get_sections is None:
                continue
            for sec in get_sections() or []:
                s = dict(sec)
                s.setdefault("order", 100)
                s.setdefault("plugin", p["name"])
                # Render template-based sections using plugin's own Jinja2 env
                # so shared macros (_macros/kpi.html etc.) resolve correctly.
                if "template" in s and "html" not in s:
                    tpl_name = s.pop("template")
                    ctx = s.pop("context", {})
                    try:
                        loaders = [FileSystemLoader(str(plugin_templates))]
                        if shared_templates.exists():
                            loaders.append(FileSystemLoader(str(shared_templates)))
                        env = Environment(
                            loader=ChoiceLoader(loaders),
                            autoescape=True,
                        )
                        tpl = env.get_template(tpl_name)
                        s["html"] = tpl.render(**ctx)
                    except Exception as render_exc:  # noqa: BLE001
                        log.warning(
                            "plugin %s template %s render failed: %s",
                            p["name"], tpl_name, render_exc,
                        )
                        s["html"] = ""
                sections.append(s)
        except Exception as exc:  # noqa: BLE001
            log.warning("plugin %s dashboard_sections failed: %s", p["name"], exc)
    sections.sort(key=lambda s: s.get("order", 100))
    return sections


@router.get("/", response_class=HTMLResponse)
def index(request: Request) -> Any:
    store = request.app.state.store
    metrics = _collect_metrics(store)
    metrics["rss_mb"] = _rss_mb()
    try:
        metrics["intents_pending"] = intents_queue.pending_count()
    except Exception:  # noqa: BLE001
        metrics["intents_pending"] = 0
    try:
        metrics["questions_open"] = len(questions_queue.list_open())
    except Exception:  # noqa: BLE001
        metrics["questions_open"] = 0
    recent_tasks = _recent_open_tasks(store)
    intercept_types = _intercept_by_type(store)
    # Router mini-stats for dashboard
    try:
        router_entries = _read_entries(200)
        router_stats = _compute_stats(router_entries)
    except Exception:  # noqa: BLE001
        router_stats = {}
    # Plugin extension-point: A2 — see docs/plugin-extension-points.md
    plugin_sections = collect_plugin_sections()
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "m": metrics,
            "recent_tasks": recent_tasks,
            "intercept_types": intercept_types,
            "router_stats": router_stats,
            "plugin_sections": plugin_sections,
            "active_tab": "dashboard",
        },
    )
