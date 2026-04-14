"""Control Panel — plugins tab (enable/disable + profile activation)."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ... import config as _cfg
from ... import plugin_registry as _pr

router = APIRouter()


def _group_by_source(plugins: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in plugins:
        grouped[p["source"]].append(p)
    for src in grouped:
        grouped[src].sort(key=lambda p: p["name"])
    return dict(sorted(grouped.items()))


def _detect_active_profile() -> str | None:
    """Return the profile name whose enabled set matches the current settings, if any."""
    cfg = _cfg.load_config()
    profiles = (cfg.get("profiles") or {})
    enabled_now = {
        k.split("@", 1)[0]
        for k, v in _pr._enabled_map().items()
        if v
    }
    for name, prof in profiles.items():
        plugs = prof.get("plugins")
        if plugs == "__all__":
            continue
        if isinstance(plugs, list) and set(plugs) == enabled_now:
            return name
    return None


@router.get("/control/plugins", response_class=HTMLResponse)
def plugins_panel(request: Request) -> Any:
    plugins = list(_pr.iter_all_plugins())
    cfg = _cfg.load_config()
    profiles = (cfg.get("profiles") or {})
    return request.app.state.templates.TemplateResponse(
        request,
        "control_plugins.html",
        {
            "grouped": _group_by_source(plugins),
            "profiles": profiles,
            "active_profile": _detect_active_profile(),
            "total": len(plugins),
            "enabled_total": sum(1 for p in plugins if p["enabled"]),
        },
    )


def _render_plugin_card(request: Request, plugin_id: str) -> HTMLResponse:
    for p in _pr.iter_all_plugins():
        if p["full_key"] == plugin_id or p["name"] == plugin_id:
            return request.app.state.templates.TemplateResponse(
                request, "_plugin_card.html", {"p": p},
            )
    return HTMLResponse(f"<div class='plugin-card error'>unknown plugin: {plugin_id}</div>", status_code=404)


@router.post("/control/plugins/{plugin_id}/toggle", response_class=HTMLResponse)
def toggle(request: Request, plugin_id: str) -> Any:
    # Find current state to flip.
    current = False
    for p in _pr.iter_all_plugins():
        if p["full_key"] == plugin_id or p["name"] == plugin_id:
            current = p["enabled"]
            plugin_id = p["full_key"] or plugin_id
            break
    _pr.toggle(plugin_id, not current)
    return _render_plugin_card(request, plugin_id)


@router.post("/control/plugins/profile/{name}", response_class=HTMLResponse)
def apply_profile(request: Request, name: str) -> Any:
    _pr.apply_profile(name)
    return plugins_panel(request)


@router.post("/control/plugins/reindex", response_class=HTMLResponse)
def reindex(request: Request) -> Any:
    try:
        from ...server import index_plugins as _idx

        _idx()
    except Exception:  # noqa: BLE001
        pass
    return plugins_panel(request)
