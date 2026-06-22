"""Control Panel — Chrome tab (status-only, no subprocess)."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter()


def _chrome_plugin_enabled() -> bool:
    """Return True when the chrome-devtools-mcp plugin is enabled."""
    try:
        from ... import plugin_registry as _pr
        for p in _pr.iter_all_plugins():
            if "chrome" in p["name"].lower() or "chrome" in str(p.get("full_key", "")).lower():
                return bool(p.get("enabled", False))
    except Exception:
        pass
    return False


def _recent_intents(limit: int = 5) -> list[dict]:
    """Return recent chrome intents; degrades gracefully to empty list."""
    try:
        from ..services.intents_queue import list_intents
        return list_intents(include_done=True)[:limit]
    except Exception:
        return []


def _total_intent_count() -> int:
    try:
        from ..services.intents_queue import list_intents
        return len(list_intents(include_done=True))
    except Exception:
        return 0


@router.get("/control/chrome", response_class=HTMLResponse)
def chrome_panel(request: Request) -> Any:
    try:
        enabled = _chrome_plugin_enabled()
    except Exception:
        enabled = False

    try:
        intents = _recent_intents()
    except Exception:
        intents = []

    try:
        total_intents = _total_intent_count()
    except Exception:
        total_intents = 0

    return request.app.state.templates.TemplateResponse(
        request,
        "control_chrome.html",
        {
            "chrome_enabled": enabled,
            "intents": intents,
            "total_intents": total_intents,
        },
    )
