"""Control Panel — service lifecycle UI + API."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, PlainTextResponse

from ... import config as _cfg
from ...services.base import install_log_path
from ...services.registry import get_pressure, get_registry

router = APIRouter()


def _render_card(request: Request, svc_name: str) -> HTMLResponse:
    reg = get_registry()
    svc = reg.get(svc_name)
    if svc is None:
        return HTMLResponse(f"<div class='service-card error'>unknown service: {svc_name}</div>", status_code=404)
    svc_cfg = (_cfg.load_config().get("services") or {}).get(svc_name) or {}
    available, reason = svc.is_available()
    has_installer = svc.installable()

    top_suggestion = ""
    for s in reg.suggestions:
        if svc.label in s:
            top_suggestion = s
            break

    return request.app.state.templates.TemplateResponse(
        request,
        "_service_card.html",
        {
            "svc": svc,
            "status": svc.status(),
            "available": available,
            "reason": reason,
            "svc_cfg": svc_cfg,
            "has_installer": has_installer,
            "top_suggestion": top_suggestion,
        },
    )


def _save_enabled(svc_name: str, enabled: bool) -> None:
    cfg = _cfg.load_config()
    services = cfg.setdefault("services", {})
    entry = services.setdefault(svc_name, {})
    entry["enabled"] = enabled
    _cfg.save_config(cfg)


def _save_field(svc_name: str, field: str, value: Any) -> None:
    cfg = _cfg.load_config()
    services = cfg.setdefault("services", {})
    entry = services.setdefault(svc_name, {})
    entry[field] = value
    _cfg.save_config(cfg)


@router.get("/control", response_class=HTMLResponse)
def control_page(request: Request) -> Any:
    reg = get_registry()
    services = reg.all()
    return request.app.state.templates.TemplateResponse(
        request,
        "control.html",
        {
            "active_tab": "control",
            "services": services,
            "disabled_services": reg.disabled_services,
        },
    )


@router.get("/control/{svc}/card", response_class=HTMLResponse)
def control_card(request: Request, svc: str) -> Any:
    return _render_card(request, svc)


@router.post("/control/{svc}/start", response_class=HTMLResponse)
def control_start(request: Request, svc: str) -> Any:
    reg = get_registry()
    service = reg.get(svc)
    if service is None:
        return HTMLResponse("unknown service", status_code=404)
    service.start()
    _save_enabled(svc, True)
    return _render_card(request, svc)


@router.post("/control/{svc}/stop", response_class=HTMLResponse)
def control_stop(request: Request, svc: str) -> Any:
    reg = get_registry()
    service = reg.get(svc)
    if service is None:
        return HTMLResponse("unknown service", status_code=404)
    service.stop()
    _save_enabled(svc, False)
    return _render_card(request, svc)


@router.post("/control/{svc}/toggle", response_class=HTMLResponse)
def control_toggle(request: Request, svc: str) -> Any:
    reg = get_registry()
    service = reg.get(svc)
    if service is None:
        return HTMLResponse("unknown service", status_code=404)
    if service.status() == "running":
        service.stop()
        _save_enabled(svc, False)
    else:
        service.start()
        _save_enabled(svc, True)
    return _render_card(request, svc)


@router.post("/control/{svc}/config", response_class=HTMLResponse)
async def control_config(request: Request, svc: str) -> Any:
    reg = get_registry()
    if reg.get(svc) is None:
        return HTMLResponse("unknown service", status_code=404)
    form = await request.form()
    for field in ("model", "container"):
        if field in form:
            _save_field(svc, field, str(form[field]).strip())
    if "auto_disable_under_pressure" in form:
        _save_field(svc, "auto_disable_under_pressure", form["auto_disable_under_pressure"] == "on")
    if "auto_start" in form:
        _save_field(svc, "auto_start", form["auto_start"] == "on")
    return _render_card(request, svc)


_install_threads: dict[str, threading.Thread] = {}


@router.post("/control/{svc}/install", response_class=HTMLResponse)
def control_install(request: Request, svc: str) -> Any:
    reg = get_registry()
    service = reg.get(svc)
    if service is None:
        return HTMLResponse("unknown service", status_code=404)

    existing = _install_threads.get(svc)
    if existing is None or not existing.is_alive():
        def _run() -> None:
            try:
                service.install()
            except Exception:
                pass

        t = threading.Thread(target=_run, name=f"install-{svc}", daemon=True)
        t.start()
        _install_threads[svc] = t

    return _render_card(request, svc)


@router.get("/control/{svc}/install-log", response_class=PlainTextResponse)
def control_install_log(svc: str) -> Any:
    path = install_log_path(svc)
    if not path.exists():
        return PlainTextResponse("(no log yet)", status_code=200)
    try:
        content = path.read_text()
    except OSError as e:
        return PlainTextResponse(f"(read error: {e})", status_code=200)
    # Keep response short — only the last 80 lines.
    lines = content.splitlines()[-80:]
    return PlainTextResponse("\n".join(lines))


@router.get("/control/monitor", response_class=HTMLResponse)
def control_monitor(request: Request) -> Any:
    reg = get_registry()
    sample = reg.last_sample
    if sample is None:
        sample = get_pressure().sample()
    return request.app.state.templates.TemplateResponse(
        request,
        "_monitor_bar.html",
        {
            "sample": sample,
            "suggestions": reg.suggestions,
            "disabled_services": reg.disabled_services,
        },
    )
