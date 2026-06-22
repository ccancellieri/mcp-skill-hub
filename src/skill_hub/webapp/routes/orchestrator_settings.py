"""Tooling Orchestrator settings route.

POST /settings/orchestrator  — validate + persist orchestrator_mode and
                               orchestrator_auto_init_roots; returns an inline
                               status span. The panel itself is rendered by the
                               settings page (see routes/settings.py).
"""
from __future__ import annotations

from html import escape
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ... import config as _config
from ...orchestrator.engine import VALID_MODES, resolve_mode

router = APIRouter()


@router.post("/settings/orchestrator", response_class=HTMLResponse)
async def orchestrator_save(request: Request) -> HTMLResponse:
    """Validate and persist orchestrator_mode + orchestrator_auto_init_roots."""
    form = await request.form()

    # --- mode ---
    raw_mode: str = str(form.get("orchestrator_mode", "")).strip().lower()
    if raw_mode not in VALID_MODES:
        # Escape the echoed value: it is reflected into an HTMLResponse that the
        # client swaps into the DOM, so an unescaped value would be reflected XSS.
        safe_mode = escape(raw_mode)
        return HTMLResponse(
            f'<span class="status err">Invalid mode: "{safe_mode}". '
            f"Must be one of: {', '.join(VALID_MODES)}</span>"
        )

    # --- roots ---
    # One folder per line. Expand ``~`` so a path entered with a home shortcut
    # is stored absolute rather than being resolved against the daemon's cwd
    # later by the engine. Blank lines are dropped.
    raw_roots: str = str(form.get("orchestrator_auto_init_roots", "")).strip()
    roots: list[str] = []
    for line in raw_roots.splitlines():
        line = line.strip()
        if line:
            roots.append(str(Path(line).expanduser()))

    try:
        _config.set("orchestrator_mode", raw_mode)
        _config.set("orchestrator_auto_init_roots", roots)
        effective = resolve_mode(_config.get)
        msg = f"Saved ✓ — effective mode: {effective}"
        cls = "ok"
    except OSError as exc:
        msg = f"Error saving: {escape(str(exc))}"
        cls = "err"

    return HTMLResponse(f'<span class="status {cls}">{msg}</span>')
