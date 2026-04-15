"""Skills route — usage stats, filter, details drawer, pin."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

router = APIRouter()

PIN_FILE = Path.home() / ".claude" / "mcp-skill-hub" / "state" / "pinned_skills.json"


def _load_pinned() -> set[str]:
    try:
        data = json.loads(PIN_FILE.read_text())
        if isinstance(data, list):
            return {str(x) for x in data}
    except (OSError, json.JSONDecodeError):
        pass
    return set()


def _save_pinned(pinned: set[str]) -> None:
    PIN_FILE.parent.mkdir(parents=True, exist_ok=True)
    PIN_FILE.write_text(json.dumps(sorted(pinned)))


def _get_skill_task_counts(store) -> dict[str, int]:
    """Count tasks per skill from router.jsonl (best effort)."""
    import json as _json
    from pathlib import Path
    from collections import Counter

    router_log = Path.home() / ".claude" / "mcp-skill-hub" / "router.jsonl"
    skill_task_count: Counter = Counter()

    if router_log.exists():
        try:
            task_sessions_seen = set()
            with router_log.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = _json.loads(line)
                        session_id = entry.get("session_id") or ""
                        skills_obj = entry.get("skills") or {}
                        preloaded = skills_obj.get("preloaded") or entry.get("preload_skills") or []
                        # Count unique (session_id, skill) pairs
                        for sk in preloaded:
                            if sk:
                                key = (session_id, sk)
                                if key not in task_sessions_seen:
                                    task_sessions_seen.add(key)
                                    skill_task_count[sk] += 1
                    except (_json.JSONDecodeError, KeyError):
                        continue
        except OSError:
            pass

    return dict(skill_task_count)


def _enrich(stats: list[dict], pinned: set[str], task_counts: dict[str, int]) -> list[dict]:
    out = []
    for s in stats:
        d = dict(s)
        d["pinned"] = d["id"] in pinned
        d["task_count"] = task_counts.get(d["id"], 0)
        out.append(d)
    # pinned first, preserve stat ordering within each group
    out.sort(key=lambda d: (not d["pinned"],))
    return out


@router.get("/skills", response_class=HTMLResponse)
def skills_page(request: Request) -> Any:
    store = request.app.state.store
    stats = store.get_skill_usage_stats()
    pinned = _load_pinned()
    task_counts = _get_skill_task_counts(store)
    rows = _enrich(stats, pinned, task_counts)
    # distinct targets / plugins for filter dropdowns
    targets: list[str] = sorted({(r.get("target") or "") for r in rows if r.get("target")})
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "skills.html",
        {
            "rows": rows,
            "targets": targets,
            "active_tab": "skills",
        },
    )


@router.post("/skills/{skill_id}/pin", response_class=HTMLResponse)
def skill_pin(skill_id: str, request: Request) -> Any:
    pinned = _load_pinned()
    if skill_id in pinned:
        pinned.discard(skill_id)
    else:
        pinned.add(skill_id)
    _save_pinned(pinned)
    store = request.app.state.store
    stats = store.get_skill_usage_stats()
    row = next((dict(r) for r in stats if r["id"] == skill_id), None)
    if not row:
        return HTMLResponse("", status_code=404)
    row["pinned"] = skill_id in pinned
    templates = request.app.state.templates
    return templates.TemplateResponse(request, "_skill_row.html", {"r": row})


@router.get("/skills/{skill_id}/detail", response_class=HTMLResponse)
def skill_detail(skill_id: str, request: Request) -> Any:
    store = request.app.state.store
    skill = store.get_skill(skill_id)
    if not skill:
        return HTMLResponse("<div class='muted'>Not found</div>", status_code=404)
    # Recent feedback rows for this skill
    try:
        fb_rows = store._conn.execute(
            "SELECT query, helpful, created_at FROM feedback "
            "WHERE skill_id = ? ORDER BY created_at DESC LIMIT 15",
            (skill_id,),
        ).fetchall()
        feedback = [dict(r) for r in fb_rows]
    except Exception:
        feedback = []
    # Embedding vector norm
    norm: float | None = None
    try:
        vrow = store._conn.execute(
            "SELECT norm, vector FROM embeddings WHERE skill_id = ?",
            (skill_id,),
        ).fetchone()
        if vrow:
            if vrow["norm"] is not None:
                norm = float(vrow["norm"])
            elif vrow["vector"]:
                vec = json.loads(vrow["vector"])
                norm = math.sqrt(sum(x * x for x in vec))
    except Exception:
        norm = None
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "_skill_detail.html",
        {"s": skill, "feedback": feedback, "norm": norm},
    )


@router.get("/skills/{skill_id}/tasks")
def skill_tasks(skill_id: str, request: Request) -> Any:
    """Return recent tasks that used this skill (from router.jsonl analysis)."""
    import json as _json
    from pathlib import Path
    from collections import Counter
    from datetime import datetime, timedelta

    store = request.app.state.store
    skill = store.get_skill(skill_id)
    if not skill:
        return {"error": "skill not found", "tasks": []}, 404

    router_log = Path.home() / ".claude" / "mcp-skill-hub" / "router.jsonl"
    tasks_with_skill: dict[str, dict] = {}  # session_id → {count, earliest, latest}

    if router_log.exists():
        try:
            with router_log.open() as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = _json.loads(line)
                        skills_obj = entry.get("skills") or {}
                        preloaded = skills_obj.get("preloaded") or entry.get("preload_skills") or []
                        if skill_id in preloaded:
                            session_id = entry.get("session_id") or ""
                            ts = entry.get("ts") or datetime.now().isoformat()
                            if session_id not in tasks_with_skill:
                                tasks_with_skill[session_id] = {"count": 0, "earliest": ts, "latest": ts}
                            tasks_with_skill[session_id]["count"] += 1
                            tasks_with_skill[session_id]["latest"] = max(
                                tasks_with_skill[session_id]["latest"], ts
                            )
                    except (_json.JSONDecodeError, KeyError):
                        continue
        except OSError:
            pass

    # Find corresponding task rows (best effort via session_id and time window)
    result_tasks = []
    try:
        if tasks_with_skill:
            for session_id, meta in sorted(
                tasks_with_skill.items(),
                key=lambda x: x[1]["latest"],
                reverse=True
            )[:10]:
                # Find tasks overlapping with this skill usage window
                rows = store._conn.execute(
                    "SELECT id, title, created_at, status FROM tasks "
                    "WHERE session_id = ? LIMIT 5",
                    (session_id,),
                ).fetchall()
                for row in rows:
                    result_tasks.append({
                        "id": row["id"],
                        "title": row["title"],
                        "created_at": row["created_at"],
                        "status": row["status"],
                        "skill_uses": meta["count"],
                    })
    except Exception:
        pass

    return {"tasks": result_tasks, "total": len(result_tasks)}
