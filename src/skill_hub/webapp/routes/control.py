"""Control Panel — service lifecycle UI + API."""
from __future__ import annotations

import re
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

from ... import config as _cfg
from ...services.base import install_log_path
from ...services.registry import get_pressure, get_registry

router = APIRouter()
# Separate router for /control/llm/* so it registers BEFORE the
# /control/{svc}/* wildcards below (main.py includes llm_router first).
llm_router = APIRouter()

# Where model-pull logs land (separate from service install logs).
_MODEL_LOG_DIR = Path.home() / ".claude" / "mcp-skill-hub" / "logs" / "model-pulls"
_VALID_MODEL_RE = re.compile(r"^[A-Za-z0-9._:\-/]+$")


def _model_log_path(model: str) -> Path:
    _MODEL_LOG_DIR.mkdir(parents=True, exist_ok=True)
    safe = model.replace("/", "_").replace(":", "-")
    return _MODEL_LOG_DIR / f"{safe}.log"


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


# Services that cannot function without the Ollama daemon.
_OLLAMA_DEPENDENTS = frozenset({"ollama_router", "ollama_embed"})


def _save_enabled(svc_name: str, enabled: bool) -> None:
    cfg = _cfg.load_config()
    services = cfg.setdefault("services", {})
    services.setdefault(svc_name, {})["enabled"] = enabled
    # Cascade: disabling the daemon makes its dependents useless too.
    # Re-enabling the daemon does NOT auto-enable dependents — user controls that.
    if svc_name == "ollama_daemon" and not enabled:
        for dep in _OLLAMA_DEPENDENTS:
            services.setdefault(dep, {})["enabled"] = False
    # Cascade: re-enabling a dependent service implies the daemon must be on.
    if svc_name in _OLLAMA_DEPENDENTS and enabled:
        services.setdefault("ollama_daemon", {})["enabled"] = True
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
    if svc == "ollama_daemon":
        for dep in _OLLAMA_DEPENDENTS:
            dep_svc = reg.get(dep)
            if dep_svc is not None:
                dep_svc.stop()
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
        if svc == "ollama_daemon":
            for dep in _OLLAMA_DEPENDENTS:
                dep_svc = reg.get(dep)
                if dep_svc is not None:
                    dep_svc.stop()
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


# ──────────────────────────────────────────────────────────────────────
# Model picker + pull (S2 follow-up)
# ──────────────────────────────────────────────────────────────────────


_TIER_KEYS = ("tier_cheap", "tier_mid", "tier_smart", "embed")

_pull_threads: dict[str, threading.Thread] = {}
_pull_state: dict[str, dict[str, Any]] = {}  # model → {"status": ..., "started_at": ...}


def _list_installed_models() -> list[dict[str, Any]]:
    """Return installed Ollama models (name + size_gb). Empty if unreachable."""
    base = _cfg.get("ollama_base") or "http://localhost:11434"
    try:
        resp = httpx.get(f"{base}/api/tags", timeout=5.0)
        resp.raise_for_status()
    except Exception:  # noqa: BLE001
        return []
    models: list[dict[str, Any]] = []
    for m in (resp.json().get("models") or []):
        models.append({
            "name": m.get("name", ""),
            "size_gb": round((m.get("size") or 0) / 1_073_741_824, 2),
            "modified_at": m.get("modified_at", ""),
        })
    models.sort(key=lambda m: m["name"])
    return models


def _current_tier_models() -> dict[str, str]:
    providers = _cfg.get("llm_providers") or {}
    if not isinstance(providers, dict):
        return {}
    return {k: str(providers.get(k, "")) for k in _TIER_KEYS if k in providers}


def _strip_ollama_prefix(model_id: str) -> str:
    return model_id.split("/", 1)[1] if model_id.startswith("ollama/") else model_id


@llm_router.get("/control/llm", response_class=JSONResponse)
def control_llm_json() -> Any:
    """JSON view: installed Ollama models, current tier assignments, pull state."""
    installed = _list_installed_models()
    tiers = _current_tier_models()
    # For each tier flag whether its Ollama model is actually installed.
    installed_names = {m["name"] for m in installed}
    tier_report = []
    for k, v in tiers.items():
        name = _strip_ollama_prefix(v)
        tier_report.append({
            "tier": k,
            "model_id": v,
            "installed": (name in installed_names) if v.startswith("ollama/") else None,
        })
    pulls = {
        name: {"status": st.get("status"),
               "started_at": st.get("started_at"),
               "duration_s": round(time.time() - st.get("started_at", time.time()), 1)}
        for name, st in _pull_state.items()
    }
    return JSONResponse({
        "installed": installed,
        "tiers": tier_report,
        "pulls": pulls,
    })


@llm_router.get("/control/llm/card", response_class=HTMLResponse)
def control_llm_card(request: Request) -> Any:
    """HTMX fragment rendering the models card."""
    return request.app.state.templates.TemplateResponse(
        request,
        "_models_card.html",
        {
            "installed": _list_installed_models(),
            "tiers": _current_tier_models(),
            "tier_keys": _TIER_KEYS,
            "pulls": dict(_pull_state),
        },
    )


@llm_router.post("/control/llm/pull", response_class=HTMLResponse)
async def control_llm_pull(request: Request) -> Any:
    form = await request.form()
    model = str(form.get("model") or "").strip()
    if not model or not _VALID_MODEL_RE.match(model):
        return HTMLResponse(
            f"<div class='error'>invalid model name: {model!r}</div>",
            status_code=400,
        )

    existing = _pull_threads.get(model)
    if existing is not None and existing.is_alive():
        return control_llm_card(request)  # already running

    log_path = _model_log_path(model)
    _pull_state[model] = {"status": "running", "started_at": time.time()}

    def _run() -> None:
        try:
            with log_path.open("w") as f:
                f.write(f"pulling {model}...\n")
                result = subprocess.run(
                    ["ollama", "pull", model],
                    stdout=f, stderr=subprocess.STDOUT,
                    timeout=3600,
                )
                f.write(f"\nexit code: {result.returncode}\n")
            _pull_state[model]["status"] = (
                "done" if result.returncode == 0 else f"error ({result.returncode})"
            )
        except FileNotFoundError:
            _pull_state[model]["status"] = "error (ollama CLI missing)"
        except subprocess.TimeoutExpired:
            _pull_state[model]["status"] = "error (timeout)"
        except Exception as exc:  # noqa: BLE001
            _pull_state[model]["status"] = f"error ({exc})"

    t = threading.Thread(target=_run, name=f"pull-{model}", daemon=True)
    t.start()
    _pull_threads[model] = t
    return control_llm_card(request)


@llm_router.get("/control/llm/pull-log/{model:path}",
            response_class=PlainTextResponse)
def control_llm_pull_log(model: str) -> Any:
    path = _model_log_path(model)
    if not path.exists():
        return PlainTextResponse("(no log yet)", status_code=200)
    try:
        content = path.read_text()
    except OSError as e:
        return PlainTextResponse(f"(read error: {e})", status_code=200)
    lines = content.splitlines()[-80:]
    return PlainTextResponse("\n".join(lines))


@llm_router.post("/control/llm/tier", response_class=HTMLResponse)
async def control_llm_tier(request: Request) -> Any:
    """Update ``llm_providers.<tier>`` from a form submission."""
    form = await request.form()
    tier = str(form.get("tier") or "").strip()
    model_id = str(form.get("model_id") or "").strip()
    if tier not in _TIER_KEYS:
        return HTMLResponse(f"<div class='error'>unknown tier: {tier}</div>",
                            status_code=400)
    if not model_id or not _VALID_MODEL_RE.match(model_id):
        return HTMLResponse(f"<div class='error'>invalid model_id: {model_id!r}</div>",
                            status_code=400)

    cfg = _cfg.load_config()
    providers = cfg.setdefault("llm_providers", {})
    providers[tier] = model_id
    _cfg.save_config(cfg)
    return control_llm_card(request)


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
