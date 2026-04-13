"""Chrome-intents queue route."""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from ..services import intents_queue

router = APIRouter()


class IntentCreate(BaseModel):
    url: str = Field(..., min_length=1)
    action: str = "navigate"
    note: str = ""


@router.get("/intents", response_class=HTMLResponse)
def intents_page(request: Request) -> Any:
    rows = intents_queue.list_intents()
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request, "intents.html",
        {"active_tab": "intents", "rows": rows,
         "pending": intents_queue.pending_count()},
    )


@router.get("/intents/list", response_class=JSONResponse)
def intents_list() -> Any:
    return JSONResponse({"rows": intents_queue.list_intents(),
                         "pending": intents_queue.pending_count()})


@router.post("/intents", response_class=JSONResponse)
def intents_add(body: IntentCreate) -> Any:
    entry = intents_queue.enqueue(body.url, action=body.action, note=body.note)
    return JSONResponse(entry, status_code=201)


@router.post("/intents/{intent_id}/done", response_class=JSONResponse)
def intents_done(intent_id: str) -> Any:
    ok = intents_queue.mark_done(intent_id, status="done")
    return JSONResponse({"ok": ok}, status_code=200 if ok else 404)


@router.post("/intents/{intent_id}/cancel", response_class=JSONResponse)
def intents_cancel(intent_id: str) -> Any:
    ok = intents_queue.mark_done(intent_id, status="cancelled")
    return JSONResponse({"ok": ok}, status_code=200 if ok else 404)
