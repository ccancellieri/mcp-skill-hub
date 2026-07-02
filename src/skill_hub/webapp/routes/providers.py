"""Providers settings route — auxiliary LLM registry editor."""
from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ... import config as _config
from ...llm import credentials as _creds
from ...llm import importers as _importers
from ...llm import registry as _registry

router = APIRouter()


def _credential_label(api_key: dict) -> str:
    """Return a display label for the credential source — never the secret value."""
    source = api_key.get("source") if isinstance(api_key, dict) else None
    ref = (api_key.get("ref") or "") if isinstance(api_key, dict) else ""
    if source == "opencode":
        return f"opencode:{ref}" if ref else "opencode"
    if source == "env":
        return f"env:{ref}" if ref else "env"
    if source == "inline":
        return "inline"
    return "none"


def _build_provider_view(raw_list: list) -> list[dict]:
    """Build a secret-free view list from raw registry records."""
    views = []
    for rec in raw_list:
        if not isinstance(rec, dict):
            continue
        api_key = rec.get("api_key") or {}
        views.append({
            "name": rec.get("name", ""),
            "kind": rec.get("kind", ""),
            "personal": bool(rec.get("personal")) or rec.get("level") == "personal",
            "api_base": rec.get("api_base", ""),
            "cred_label": _credential_label(api_key),
            "enabled": bool(rec.get("enabled", True)),
            "order": rec.get("order", 100),
            "models": [
                {
                    "id": m.get("id", ""),
                    "complexity": m.get("complexity", "light"),
                    "monthly_cap_tokens": m.get("monthly_cap_tokens"),
                    "tags": m.get("tags") or [],
                }
                for m in (rec.get("models") or [])
                if isinstance(m, dict)
            ],
        })
    # The ladder IS the order — render the table as the actual chain.
    views.sort(key=lambda v: v["order"])
    return views


def _attach_usage(views: list[dict]) -> None:
    """Annotate each provider view with live metering (calls / errors / tokens).

    Reads the ``llm_call`` metering aggregated by the store and folds the
    per-model rows onto the provider that owns each model id. This is what makes
    gateway usage visible instead of looking like 'credits = 0'.
    """
    try:
        from ...store import get_store
        stats = get_store().get_llm_stats()
    except Exception:  # noqa: BLE001 - usage panel is best-effort
        stats = {}
    by_model = stats.get("by_model") or {}
    for view in views:
        ids = {m["id"] for m in view.get("models", [])}
        calls = errors = tokens = 0
        for mid, row in by_model.items():
            # The litellm dispatch id may carry an ``openai/`` route prefix for
            # gateway models — match on suffix so both forms fold together.
            if mid in ids or any(mid.endswith(i) or i.endswith(mid) for i in ids):
                calls += int(row.get("count") or 0)
                errors += int(row.get("errors") or 0)
                tokens += int(row.get("total_tokens") or 0)
        view["usage"] = {"calls": calls, "errors": errors,
                         "ok": calls - errors, "tokens": tokens}


@router.get("/providers", response_class=HTMLResponse)
def providers_page(request: Request) -> Any:
    raw_list = _config.get("llm_provider_registry") or []
    if not isinstance(raw_list, list):
        raw_list = []
    provider_views = _build_provider_view(raw_list)
    _attach_usage(provider_views)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "providers.html",
        {
            "active_tab": "providers",
            "providers": provider_views,
        },
    )


@router.post("/providers")
async def providers_save(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON body"})

    registry_list = body.get("registry")
    if not isinstance(registry_list, list):
        return JSONResponse({"ok": False, "error": "Body must have a 'registry' list"})

    # Preserve credentials across a save from the secret-free page view: the
    # rendered table omits api_key, so a record posted without one (or with an
    # empty one) inherits the stored provider's credential by name. This keeps
    # the page secret-free yet prevents a Save from silently wiping creds.
    stored = _config.get("llm_provider_registry") or []
    stored_keys = {
        r.get("name"): r.get("api_key")
        for r in (stored if isinstance(stored, list) else [])
        if isinstance(r, dict) and r.get("api_key")
    }

    merged: list = []
    for rec in registry_list:
        if isinstance(rec, dict):
            rec = {k: v for k, v in rec.items() if k not in ("cred_label", "usage")}
            if not rec.get("api_key") and rec.get("name") in stored_keys:
                rec["api_key"] = stored_keys[rec["name"]]
        merged.append(rec)

    # Validate every record — reject the whole write if any parses to None.
    # Error text is sanitized: never echo the raw record (it may carry an
    # inline credential).
    for i, rec in enumerate(merged):
        if _registry._parse_provider(rec) is None:
            label = rec.get("name") if isinstance(rec, dict) else None
            ref = repr(label) if label else f"entry {i}"
            return JSONResponse({"ok": False, "error": f"Invalid provider record: {ref}"})

    _config.set("llm_provider_registry", merged)
    return JSONResponse({"ok": True})


@router.post("/providers/test")
async def providers_test(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "status": 0, "models_count": 0})

    name = body.get("name", "")
    raw_list = _config.get("llm_provider_registry") or []
    provider = None
    for rec in (raw_list if isinstance(raw_list, list) else []):
        parsed = _registry._parse_provider(rec)
        if parsed and parsed.name == name:
            provider = parsed
            break

    if provider is None:
        return JSONResponse({"ok": False, "status": 0, "models_count": 0})

    api_base, api_key = _creds.resolve_credentials(provider)
    if not api_base:
        return JSONResponse({"ok": False, "status": 0, "models_count": 0})

    url = api_base.rstrip("/") + "/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url, headers=headers)
        try:
            data = resp.json()
            count = len(data.get("data") or data.get("models") or [])
        except Exception:
            count = 0
        return JSONResponse({"ok": resp.status_code < 400, "status": resp.status_code, "models_count": count})
    except Exception:
        return JSONResponse({"ok": False, "status": 0, "models_count": 0})


async def _import_incoming(body: dict) -> tuple[list, str | None]:
    """Resolve (incoming_providers, error) from an import request body.

    For ``opencode`` with no pasted payload, reads the on-disk opencode config
    (the one-click Sync path). Other formats require a pasted ``payload`` object.
    """
    fmt = body.get("format")
    if fmt not in _importers.SUPPORTED_FORMATS:
        return [], "Unsupported format"
    payload = body.get("payload")
    if fmt == "opencode" and not payload:
        payload = _importers.read_opencode_config()
    if not isinstance(payload, dict) or not payload:
        return [], "No config payload (paste one, or check ~/.config/opencode)"
    try:
        incoming = _importers.normalize(fmt, payload)
    except Exception:  # noqa: BLE001 — never echo the payload (may carry a secret)
        return [], "Could not parse the config for this format"
    if not incoming:
        return [], "No providers found in the config"
    return incoming, None


@router.post("/providers/import/preview")
async def providers_import_preview(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON body"})
    incoming, err = await _import_incoming(body)
    if err:
        return JSONResponse({"ok": False, "error": err})
    current = _config.get("llm_provider_registry") or []
    return JSONResponse({"ok": True, "diff": _importers.diff_registry(current, incoming)})


@router.post("/providers/import/apply")
async def providers_import_apply(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid JSON body"})
    incoming, err = await _import_incoming(body)
    if err:
        return JSONResponse({"ok": False, "error": err})
    current = _config.get("llm_provider_registry") or []
    merged, diff = _importers.merge_registry(current, incoming)
    # Best-effort: label newly-added models (complexity + tags) with a light LLM
    # so they don't all land as "light". On by default; skipped if the client
    # opts out or the classifier is unreachable (new models stay "light").
    classified: dict = {}
    if body.get("classify", True):
        # classify_models does a blocking LLM round-trip — offload it so the
        # event loop is not stalled for the duration of the gateway call.
        classified = await asyncio.to_thread(
            _importers.apply_classification, merged, diff,
            _importers.classify_models)
    # Validate the whole write; reject if any record is malformed.
    for i, rec in enumerate(merged):
        if _registry._parse_provider(rec) is None:
            label = rec.get("name") if isinstance(rec, dict) else None
            return JSONResponse({"ok": False,
                                 "error": f"Invalid provider record: {label or i}"})
    _config.set("llm_provider_registry", merged)
    return JSONResponse({"ok": True, "diff": diff, "count": len(merged),
                         "classified": len(classified)})


@router.post("/providers/opencode/inject-mcp")
async def providers_opencode_inject_mcp() -> JSONResponse:
    """(Re)write the skill-hub MCP server block into the opencode config so
    opencode regains skill-hub's tools after its config is regenerated."""
    try:
        result = _importers.inject_skill_hub_mcp()
    except Exception:  # noqa: BLE001 — never leak a filesystem path detail
        return JSONResponse({"ok": False, "error": "Could not write opencode config"})
    return JSONResponse({"ok": True, **result})
