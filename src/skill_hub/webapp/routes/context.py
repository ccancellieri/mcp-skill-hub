"""Deterministic context workspace routes."""
from __future__ import annotations

import asyncio
import html
import os
import sqlite3
from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


def prepare_composition(*args: Any, **kwargs: Any) -> dict:
    from ...context_composer import prepare_composition as prepare
    return prepare(*args, **kwargs)


def compose_context(*args: Any, **kwargs: Any) -> dict:
    from ...context_composer import compose_context as compose
    return compose(*args, **kwargs)


def optimize_prompt(prompt: str) -> dict:
    from ...context_composer import optimize_prompt as optimize
    return optimize(prompt)


def _error(message: str, status: int = 422) -> HTMLResponse:
    return HTMLResponse(f'<p class="context-error" role="alert">{html.escape(message)}</p>', status_code=status)


def _known_project_roots(store: Any, limit: int = 20) -> list[str]:
    """Return bounded explicit project roots already known to configuration or tasks."""
    from ... import config

    values: list[Any] = []
    roots = config.get("context_project_roots")
    if isinstance(roots, list):
        values.extend(roots)
    aliases = config.get("context_project_aliases")
    if isinstance(aliases, Mapping):
        values.extend(aliases.keys())
    try:
        rows = store._conn.execute(
            "SELECT cwd FROM tasks WHERE cwd IS NOT NULL AND cwd != '' "
            "ORDER BY updated_at DESC LIMIT 50"
        ).fetchall()
        values.extend(row["cwd"] for row in rows)
    except (AttributeError, TypeError, sqlite3.Error):
        pass

    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not os.path.isabs(os.path.expanduser(value)):
            continue
        canonical = os.path.normcase(os.path.normpath(os.path.expanduser(value)))
        if canonical not in result:
            result.append(canonical)
        if len(result) >= limit:
            break
    return result


@router.post("/context/prepare", response_class=HTMLResponse)
async def context_prepare(request: Request) -> Any:
    data = await request.form()
    try:
        budget = int(str(data.get("token_budget") or "1500"))
        submitted_roots = [str(value).strip() for value in data.getlist("selected_projects")]
        submitted_roots.extend(
            path.strip() for path in str(data.get("projects") or "").splitlines()
        )
        project_roots = list(dict.fromkeys(path for path in submitted_roots if path))
        result = await asyncio.to_thread(
            prepare_composition, str(data.get("prompt") or ""),
            project_roots=project_roots,
            token_budget=budget, mode="manual",
            session_id=str(data.get("session_id") or ""),
            task_id=int(str(data["task_id"])) if data.get("task_id") else None,
            store=request.app.state.store,
        )
    except ValueError as exc:
        return _error(str(exc))
    except Exception:  # noqa: BLE001 - show outages without exposing internal data
        return _error("Context is unavailable. Your original prompt has not changed.", 503)
    return request.app.state.templates.TemplateResponse(request, "_context_candidates.html", {
        "draft": result,
    })


@router.post("/context/compose", response_class=HTMLResponse)
async def context_compose(request: Request) -> Any:
    data = await request.form()
    try:
        selected_ids = [str(value) for value in data.getlist("selected_ids")]
        result = await asyncio.to_thread(
            compose_context, str(data.get("draft_id") or ""),
            selected_ids=selected_ids,
            rejected_ids=[],
            excerpts={key.removeprefix("excerpt:"): str(value) for key, value in data.items()
                      if key.startswith("excerpt:") and key.removeprefix("excerpt:") in selected_ids
                      and str(value).strip()},
            confirmed=False, store=request.app.state.store,
        )
    except ValueError as exc:
        return _error(str(exc))
    except Exception:  # noqa: BLE001 - preserve the draft during transient outages
        return _error("Composition is unavailable. Prepare the candidates again.", 503)
    return request.app.state.templates.TemplateResponse(request, "_context_composed.html", {"composition": result})


@router.get("/context/source/{draft_id}/{candidate_id}", response_class=HTMLResponse)
async def context_source(request: Request, draft_id: str, candidate_id: str) -> Any:
    from ...context_composer import get_composition_candidate
    try:
        candidate = await asyncio.to_thread(get_composition_candidate, draft_id, candidate_id, store=request.app.state.store)
    except ValueError as exc:
        return _error(str(exc))
    return HTMLResponse('<pre>' + html.escape(str(candidate.get("text") or "")) + '</pre>')


@router.post("/context/optimize", response_class=HTMLResponse)
async def context_optimize(request: Request) -> Any:
    data = await request.form()
    try:
        result = optimize_prompt(str(data.get("prompt") or ""))
    except ValueError as exc:
        return _error(str(exc))
    return request.app.state.templates.TemplateResponse(request, "_context_optimized.html", {"optimization": result})


def build_context(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Load the deterministic context builder when a request needs it."""
    from ...context_service import build_context as _build_context

    return _build_context(*args, **kwargs)


def _page_context(**extra: Any) -> dict[str, Any]:
    return {
        "active_tab": "context",
        "result": None,
        "form": {"prompt": "", "cwd": "", "session_id": "", "task_id": ""},
        **extra,
    }


@router.get("/context", response_class=HTMLResponse)
def context_page(request: Request) -> Any:
    """Render the context builder without making capability claims."""
    return request.app.state.templates.TemplateResponse(
        request, "context.html", _page_context(
            known_projects=_known_project_roots(getattr(request.app.state, "store", None))
        )
    )


@router.post("/context/build", response_class=HTMLResponse)
async def context_build(request: Request) -> Any:
    """Build context off the event loop and return the result panel."""
    form_data = await request.form()
    form = {
        "prompt": str(form_data.get("prompt") or ""),
        "cwd": str(form_data.get("cwd") or "").strip(),
        "session_id": str(form_data.get("session_id") or "").strip(),
        "task_id": str(form_data.get("task_id") or "").strip(),
    }
    if not form["prompt"].strip():
        return HTMLResponse("<p class=\"context-error\">Prompt is required.</p>", status_code=422)
    task_id: int | None = None
    if form["task_id"]:
        try:
            task_id = int(form["task_id"])
        except ValueError:
            return HTMLResponse("<p class=\"context-error\">Task ID must be a whole number.</p>", status_code=422)

    try:
        result = await asyncio.to_thread(
            build_context,
            form["prompt"],
            cwd=form["cwd"],
            session_id=form["session_id"],
            task_id=task_id,
            store=getattr(request.app.state, "store", None),
        )
    except Exception:  # noqa: BLE001 - context inspection tolerates temporary outages
        result = {
            "original_prompt": form["prompt"],
            "context": "",
            "items": [],
            "warnings": ["Context is unavailable right now. Try again shortly."],
            "elapsed_ms": 0,
            "mode": "deterministic",
            "selected_count": 0,
            "omitted_count": 0,
        }
    return request.app.state.templates.TemplateResponse(
        request, "_context_result.html", {"result": result, "form": form}
    )
