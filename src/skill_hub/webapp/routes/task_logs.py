"""Per-task log scoping — filter hook/activity logs by ``task=<id>`` tag.

Hook log lines written while a task is active are tagged with a ``task=<id>``
token by the hook ``log()`` helpers (see ``hooks/verdict_cache.py``
``task_tag()``). This router exposes per-task views of those lines.
"""
from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse

from ..services import log_tail

router = APIRouter()

# Word-bounded task=<id> matcher avoids collisions (task=10 vs task=102).
_TASK_TAG_RE = re.compile(r"\btask=(\d+)\b")


def _predicate_for(task_id: int):
    needle = f"task={task_id}"
    def _match(line: str) -> bool:
        # Cheap substring check first; fall back to regex word-boundary match.
        if needle not in line:
            return False
        m = _TASK_TAG_RE.search(line)
        return bool(m and m.group(1) == str(task_id))
    return _match


def _filtered_tail(task_id: int, max_lines: int = 200) -> list[str]:
    pred = _predicate_for(task_id)
    out: list[str] = []
    for path in (log_tail.HOOK_LOG, log_tail.ACTIVITY_LOG):
        for ln in log_tail.grep_file_sync(path, pred, max_results=max_lines * 3):
            out.append(ln.rstrip("\n"))
    return out[-max_lines:]


@router.get("/tasks/{task_id}/logs", response_class=HTMLResponse)
def task_logs(task_id: int, request: Request) -> Any:
    templates = request.app.state.templates
    lines = _filtered_tail(task_id, max_lines=200)
    return templates.TemplateResponse(
        request,
        "_task_logs.html",
        {"task_id": task_id, "lines": lines},
    )


@router.get("/tasks/{task_id}/logs/raw", response_class=PlainTextResponse)
def task_logs_raw(task_id: int) -> Any:
    """Plain-text log lines for the Alpine.js task detail view."""
    lines = _filtered_tail(task_id, max_lines=200)
    return PlainTextResponse("\n".join(lines))


@router.websocket("/tasks/{task_id}/logs/ws")
async def task_logs_ws(ws: WebSocket, task_id: int) -> None:
    await ws.accept()
    paths = [p for p in (log_tail.HOOK_LOG, log_tail.ACTIVITY_LOG) if p.exists()]
    if not paths:
        await ws.send_text(json.dumps({
            "ts": "", "level": "warn", "source": "system",
            "text": "no log files found",
        }))
        await ws.close()
        return
    pred = _predicate_for(task_id)
    try:
        async for entry in log_tail.tail_files(paths, seed_lines=200, predicate=pred):
            try:
                await ws.send_text(json.dumps(entry))
            except (WebSocketDisconnect, RuntimeError):
                break
    except (WebSocketDisconnect, asyncio.CancelledError):
        pass
    finally:
        try:
            await ws.close()
        except RuntimeError:
            pass
