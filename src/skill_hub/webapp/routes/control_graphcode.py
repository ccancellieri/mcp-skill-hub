"""Control Panel — Code Graph tab (observability + one-click actions)."""
from __future__ import annotations

import json
import logging
from html import escape
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse

from ... import config as _cfg

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_mode() -> str:
    try:
        from ...orchestrator.engine import resolve_mode
        return resolve_mode()
    except Exception:
        return "unknown"


def _configured_roots() -> list[str]:
    try:
        roots = _cfg.get("orchestrator_auto_init_roots") or []
        if isinstance(roots, list):
            return [str(r) for r in roots if r]
    except Exception:
        pass
    return []


def _recent_roots_from_store(store: Any) -> list[str]:
    """Extract deduplicated project paths from orchestrator_decision events."""
    if store is None:
        return []
    try:
        events = store.get_events(kind="orchestrator_decision", limit=500)
        seen: dict[str, float] = {}
        for evt in events:
            try:
                payload = evt.get("payload") or {}
                if isinstance(payload, str):
                    payload = json.loads(payload)
                for decision in payload.get("decisions", []):
                    target = decision.get("target")
                    ts = evt.get("ts", 0.0)
                    if target and (target not in seen or seen[target] < ts):
                        seen[target] = ts
            except Exception:
                continue
        # Return sorted by most-recent first.
        return [t for t, _ in sorted(seen.items(), key=lambda x: -x[1])]
    except Exception:
        return []


def _build_root_list(store: Any) -> list[str]:
    """Merge configured roots + recently-seen roots, deduped, no duplicates."""
    seen: set[str] = set()
    result: list[str] = []
    for root in _configured_roots() + _recent_roots_from_store(store):
        if root not in seen:
            seen.add(root)
            result.append(root)
    return result


def _probe_root(root_str: str) -> dict[str, Any]:
    """Run probe_codegraph for a single root; return structured info."""
    try:
        from ...orchestrator.engine import probe_codegraph
        readiness = probe_codegraph(Path(root_str))

        if readiness.present and readiness.fresh:
            badge = "fresh"
            badge_cls = "tag status-running"
        elif readiness.present and not readiness.fresh:
            badge = "stale"
            badge_cls = "tag status-transitioning"
        elif not readiness.present and readiness.worktree_mismatch:
            badge = "worktree index mismatch"
            badge_cls = "tag status-unavailable"
        elif not readiness.present and "0 nodes" in readiness.detail:
            badge = "empty"
            badge_cls = "tag status-stopped"
        else:
            badge = "not indexed"
            badge_cls = "tag status-stopped"

        # Parse node count from detail string (e.g. ", 1234 nodes").
        node_count: str | None = None
        import re
        m = re.search(r"(\d+)\s+nodes?", readiness.detail)
        if m:
            node_count = m.group(1)

        age_str: str | None = None
        if readiness.stale_age is not None:
            secs = int(readiness.stale_age)
            if secs < 60:
                age_str = f"{secs}s"
            elif secs < 3600:
                age_str = f"{secs // 60}m"
            else:
                age_str = f"{secs // 3600}h {(secs % 3600) // 60}m"

        return {
            "root": root_str,
            "badge": badge,
            "badge_cls": badge_cls,
            "node_count": node_count,
            "age_str": age_str,
            "detail": readiness.detail,
            "present": readiness.present,
            "fresh": readiness.fresh,
            "error": None,
        }
    except Exception as exc:
        logger.debug("_probe_root failed for %s: %s", root_str, exc)
        return {
            "root": root_str,
            "badge": "probe error",
            "badge_cls": "tag status-unavailable",
            "node_count": None,
            "age_str": None,
            "detail": str(exc),
            "present": False,
            "fresh": False,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("/control/graphcode", response_class=HTMLResponse)
def graphcode_panel(request: Request) -> Any:
    store = getattr(request.app.state, "store", None)
    try:
        mode = _resolve_mode()
    except Exception:
        mode = "unknown"

    try:
        roots = _build_root_list(store)
    except Exception:
        roots = []

    rows = [_probe_root(r) for r in roots]

    return request.app.state.templates.TemplateResponse(
        request,
        "control_graphcode.html",
        {"mode": mode, "rows": rows},
    )


@router.post("/control/graphcode/sync", response_class=HTMLResponse)
async def graphcode_sync(request: Request, path: str = Form(...)) -> HTMLResponse:
    """Sync (refresh) the code-graph index for the given path."""
    safe_path = escape(path)
    root = Path(path).expanduser()
    if not root.is_dir():
        return HTMLResponse(
            f'<span class="status err">Not a directory: {safe_path}</span>'
        )
    try:
        from ...orchestrator.engine import ensure_tooling_core
        result = ensure_tooling_core(str(root), refresh=True, init=False)
        probe = _probe_root(str(root))
        return request.app.state.templates.TemplateResponse(
            request,
            "_graphcode_row.html",
            {"row": probe, "action_msg": f"sync: {result.get('action', 'done')}"},
        )
    except Exception as exc:
        return HTMLResponse(
            f'<span class="status err">Sync failed: {escape(str(exc))}</span>'
        )


@router.post("/control/graphcode/reindex", response_class=HTMLResponse)
async def graphcode_reindex(request: Request, path: str = Form(...)) -> HTMLResponse:
    """Re-index (init) the code-graph for the given path. May block ~120s."""
    safe_path = escape(path)
    root = Path(path).expanduser()
    if not root.is_dir():
        return HTMLResponse(
            f'<span class="status err">Not a directory: {safe_path}</span>'
        )
    try:
        from ...orchestrator.engine import ensure_tooling_core
        result = ensure_tooling_core(str(root), init=True, refresh=False)
        probe = _probe_root(str(root))
        return request.app.state.templates.TemplateResponse(
            request,
            "_graphcode_row.html",
            {"row": probe, "action_msg": f"reindex: {result.get('action', 'done')}"},
        )
    except Exception as exc:
        return HTMLResponse(
            f'<span class="status err">Reindex failed: {escape(str(exc))}</span>'
        )


@router.post("/control/graphcode/probe", response_class=HTMLResponse)
async def graphcode_probe(request: Request, path: str = Form(default="")) -> HTMLResponse:
    """Probe a user-supplied path and render a single-row result."""
    safe_path = escape(path.strip())
    if not path.strip():
        return HTMLResponse('<span class="status err">Path is required.</span>')
    root = Path(path.strip()).expanduser()
    if not root.is_dir():
        return HTMLResponse(
            f'<span class="status err">Not an existing directory: {safe_path}</span>'
        )
    probe = _probe_root(str(root))
    return request.app.state.templates.TemplateResponse(
        request,
        "control_graphcode_probe_result.html",
        {"row": probe},
    )
