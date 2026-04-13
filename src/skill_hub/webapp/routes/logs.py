"""Logs route — live WebSocket tail + download."""
from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Any

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, StreamingResponse

from ..services import log_tail

router = APIRouter()

# Server-side ring buffer cap, shared across WS clients (best-effort).
_RING: deque[dict] = deque(maxlen=2000)


@router.get("/logs", response_class=HTMLResponse)
def logs_page(request: Request) -> Any:
    templates = request.app.state.templates
    sources = [p.stem for p in (log_tail.HOOK_LOG, log_tail.ACTIVITY_LOG) if p.exists()]
    return templates.TemplateResponse(
        request,
        "logs.html",
        {"sources": sources, "active_tab": "logs"},
    )


@router.get("/logs/download")
def logs_download(source: str = "hook-debug", lines: int = 5000) -> StreamingResponse:
    if source == "activity":
        path = log_tail.ACTIVITY_LOG
    else:
        path = log_tail.HOOK_LOG
    lines = max(1, min(lines, 20000))
    tail = log_tail.tail_file_sync(path, lines)

    def gen():
        for ln in tail:
            yield ln if ln.endswith("\n") else ln + "\n"

    return StreamingResponse(
        gen(),
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{path.name}.tail"'},
    )


@router.websocket("/ws/logs")
async def ws_logs(ws: WebSocket) -> None:
    await ws.accept()
    paths = [p for p in (log_tail.HOOK_LOG, log_tail.ACTIVITY_LOG) if p.exists()]
    if not paths:
        await ws.send_text(json.dumps({
            "ts": "", "level": "warn", "source": "system",
            "text": "no log files found under ~/.claude/mcp-skill-hub/logs",
        }))
        await ws.close()
        return
    try:
        async for entry in log_tail.tail_files(paths, seed_lines=200):
            _RING.append(entry)
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
