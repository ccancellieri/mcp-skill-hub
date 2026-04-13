"""Questions queue route + SSE stream.

Hooks call POST /questions/ask (or use the python module directly) to enqueue
a question and short-poll for an answer. The UI subscribes to /questions/stream
(SSE) to surface toasts and offers allow/deny buttons.
"""
from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ..services import questions_queue

router = APIRouter()


class QuestionAsk(BaseModel):
    prompt: str = Field(..., min_length=1)
    command: str = ""
    tool_name: str = ""


class QuestionAnswer(BaseModel):
    decision: str = Field(..., pattern="^(allow|deny)$")
    reason: str = ""


@router.get("/questions", response_class=HTMLResponse)
def questions_page(request: Request) -> Any:
    open_rows = questions_queue.list_open()
    recent = questions_queue.list_recent(limit=20)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request, "questions.html",
        {"active_tab": "questions", "open_rows": open_rows, "recent": recent},
    )


@router.get("/questions/list", response_class=JSONResponse)
def questions_list() -> Any:
    return JSONResponse({"open": questions_queue.list_open(),
                         "recent": questions_queue.list_recent(20)})


@router.post("/questions/ask", response_class=JSONResponse)
def questions_ask(body: QuestionAsk) -> Any:
    entry = questions_queue.enqueue_question(
        body.prompt, command=body.command, tool_name=body.tool_name,
    )
    return JSONResponse(entry, status_code=201)


@router.post("/questions/{qid}/answer", response_class=JSONResponse)
def questions_answer(qid: str, body: QuestionAnswer) -> Any:
    ok = questions_queue.answer(qid, body.decision, reason=body.reason)
    return JSONResponse({"ok": ok}, status_code=200 if ok else 404)


@router.get("/questions/stream")
async def questions_stream() -> StreamingResponse:
    return StreamingResponse(
        questions_queue.stream_events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
