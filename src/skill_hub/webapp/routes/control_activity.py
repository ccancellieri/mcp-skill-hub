"""Control Panel — Activity Sync tab (issue links, drift warnings, event stream)."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _fmt_ts(ts: float | None) -> str:
    """Convert a Unix timestamp to a compact UTC string, or '—'."""
    if not ts:
        return "—"
    try:
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return "—"


def _freshness(last_synced_at: str | None) -> tuple[str, str]:
    """Return (age_label, css_class) for a last_synced_at ISO string.

    Returns:
        age_label: human-readable age string or 'never'.
        css_class: one of 'sync-fresh', 'sync-stale', 'sync-unseen'.
    """
    if not last_synced_at:
        return "never", "sync-unseen"
    try:
        dt = datetime.fromisoformat(last_synced_at.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - dt).total_seconds()
        if age < 3600:
            label = f"{int(age // 60)}m ago"
        elif age < 86400:
            label = f"{int(age // 3600)}h ago"
        else:
            label = f"{int(age // 86400)}d ago"
        css = "sync-fresh" if age < 3600 else "sync-stale"
        return label, css
    except Exception:
        return last_synced_at[:16], "sync-stale"


def _load_panel_data(store: Any) -> dict:
    """Query the store for all activity-sync data, degrading gracefully on error."""
    links: list[dict] = []
    drift: list[dict] = []
    events: list[dict] = []
    error: str | None = None

    if store is None:
        return {
            "links": links,
            "drift": drift,
            "events": events,
            "error": "store unavailable",
        }

    try:
        raw_links = store.list_all_issue_links()

        # Enrich each link with its task title and compute drift.
        # All string values are passed as plain Python strings — Jinja2 autoescape
        # handles HTML encoding in the template. No Python-side escape() calls here.
        for link in raw_links:
            task_id = link.get("task_id")
            task = None
            try:
                task = store.get_task(task_id)
            except Exception:
                pass

            task_status = (task["status"] if task else "unknown") if task else "unknown"
            issue_state = (link.get("state") or "").lower()
            last_synced = link.get("last_synced_at")
            age_label, age_css = _freshness(last_synced)

            enriched = {
                "id": link.get("id"),
                "task_id": task_id,
                "task_title": task["title"] if task else "(task not found)",
                "task_status": task_status,
                "issue_number": link.get("issue_number"),
                "repo": link.get("repo") or "",
                "url": link.get("url") or "",
                "issue_state": issue_state,
                "last_synced_at": last_synced or "",
                "sync_age_label": age_label,
                "sync_age_css": age_css,
                "writeback_done": bool(link.get("writeback_done")),
            }

            # Detect drift: task and issue disagree on open/closed state.
            is_drift = (
                (task_status == "open" and issue_state == "closed")
                or (task_status == "closed" and issue_state == "open")
            )
            enriched["is_drift"] = is_drift
            if is_drift:
                enriched["drift_direction"] = (
                    "issue→task" if issue_state == "closed" else "task→issue"
                )
                drift.append(enriched)

            links.append(enriched)

    except Exception as exc:  # noqa: BLE001
        logger.debug("activity panel: issue links query failed: %s", exc)
        error = str(exc)

    try:
        # Most-recent 50 events, newest first (we reverse the ascending query result).
        raw_events = store.get_events(limit=50)
        raw_events.reverse()
        for evt in raw_events:
            payload_raw = evt.get("payload") or "{}"
            try:
                payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
            except Exception:
                payload = {}
            events.append({
                "id": evt.get("id"),
                "ts_label": _fmt_ts(evt.get("ts")),
                "kind": evt.get("kind") or "",
                "tool_name": evt.get("tool_name") or "",
                "session_id": (evt.get("session_id") or "")[:12],
                "payload_preview": str(payload)[:120],
            })
    except Exception as exc:  # noqa: BLE001
        logger.debug("activity panel: events query failed: %s", exc)
        if error is None:
            error = str(exc)

    return {
        "links": links,
        "drift": drift,
        "events": events,
        "error": error,
    }


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.get("/control/activity", response_class=HTMLResponse)
def activity_panel(request: Request) -> Any:
    store = getattr(request.app.state, "store", None)
    data = _load_panel_data(store)
    return request.app.state.templates.TemplateResponse(
        request,
        "control_activity.html",
        data,
    )
