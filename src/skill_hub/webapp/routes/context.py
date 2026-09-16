"""Deterministic context workspace routes."""
from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()


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
        request, "context.html", _page_context()
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
    except Exception:  # Context inspection must not turn a temporary outage into a 500.
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
