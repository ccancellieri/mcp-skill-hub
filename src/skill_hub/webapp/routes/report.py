"""Report route — global gains + per-task analytics dashboard."""
from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()

_DB        = Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"
_ROUTER    = Path.home() / ".claude" / "mcp-skill-hub" / "router.jsonl"
_VERDICTS  = Path.home() / ".claude" / "mcp-skill-hub" / "command_verdicts.db"
_LOG_DIR   = Path.home() / ".claude" / "mcp-skill-hub" / "logs"
_HOOK_LOG  = _LOG_DIR / "hook-debug.log"
_ACT_LOG   = _LOG_DIR / "activity.log"

_TASK_TAG_RE = re.compile(r"\btask=(\d+)\b")


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_router(max_lines: int = 5000) -> list[dict]:
    if not _ROUTER.exists():
        return []
    entries: list[dict] = []
    try:
        for line in _ROUTER.read_text().splitlines()[-max_lines:]:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return entries


def _normalize_entry(e: dict) -> tuple[dict, int, int]:
    v = e.get("verdict", {})
    tier = v.get("tier_used") if v.get("tier_used") is not None else v.get("tier", 1)
    lat_ms = e.get("latency_ms") or e.get("latency", {}).get("total_ms", 0) or 0
    return v, int(tier), int(lat_ms)


def _parse_dt(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.strptime(s[:19], "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None


def _fmt_hours(h: float) -> str:
    if h < 1:
        return f"{int(h * 60)}m"
    if h < 24:
        return f"{h:.1f}h"
    return f"{h / 24:.1f}d"


def _task_log_counts() -> dict[int, int]:
    counts: dict[int, int] = defaultdict(int)
    log_paths = [_HOOK_LOG, _ACT_LOG]
    log_paths += sorted(_LOG_DIR.glob("activity.log.*")) if _LOG_DIR.exists() else []
    for path in log_paths:
        if not path.exists():
            continue
        try:
            for ln in path.read_text(errors="replace").splitlines():
                m = _TASK_TAG_RE.search(ln)
                if m:
                    counts[int(m.group(1))] += 1
        except OSError:
            pass
    return dict(counts)


def _router_by_session(entries: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        sid = e.get("session_id", "")
        if sid:
            result[sid].append(e)
    return dict(result)


# Blended Anthropic pricing ($/1M tokens), weighted 30% input / 70% output.
# Haiku: $0.25 in + $1.25 out = $0.95/M blended
# Sonnet: $3 in + $15 out = $11.4/M blended
# Opus: $15 in + $75 out = $57/M blended
_USD_PER_M = {"haiku": 0.95, "sonnet": 11.4, "opus": 57.0}


def _estimate_usd(tokens_by_model: dict[str, int]) -> float:
    total = 0.0
    for model, tok in tokens_by_model.items():
        rate = _USD_PER_M.get(model, _USD_PER_M["sonnet"])
        total += (tok / 1_000_000) * rate
    return round(total, 2)


def _daily_trends(entries: list[dict]) -> list[dict]:
    """Aggregate router entries by YYYY-MM-DD — prompts, tokens saved, enriched."""
    buckets: dict[str, dict] = defaultdict(
        lambda: {"prompts": 0, "tokens": 0, "enriched": 0, "plan_mode": 0, "lat_sum": 0, "lat_n": 0}
    )
    for e in entries:
        day = (e.get("ts") or "")[:10]
        if not day:
            continue
        b = buckets[day]
        b["prompts"] += 1
        b["tokens"] += (e.get("savings") or {}).get("tokens_estimated", 0) or 0
        if (e.get("enrichment") or {}).get("applied"):
            b["enriched"] += 1
        v, _, lat = _normalize_entry(e)
        if v.get("plan_mode"):
            b["plan_mode"] += 1
        if lat > 0:
            b["lat_sum"] += lat
            b["lat_n"] += 1
    out = []
    for day in sorted(buckets.keys()):
        b = buckets[day]
        enrich_rate = round(100 * b["enriched"] / b["prompts"], 1) if b["prompts"] else 0
        avg_lat = round(b["lat_sum"] / b["lat_n"]) if b["lat_n"] else 0
        out.append({
            "day": day,
            "prompts": b["prompts"],
            "tokens": b["tokens"],
            "enriched": b["enriched"],
            "enrich_rate": enrich_rate,
            "plan_mode": b["plan_mode"],
            "avg_lat": avg_lat,
        })
    return out


# ── data aggregation ──────────────────────────────────────────────────────────

def _build_report() -> dict:
    # ── Router ────────────────────────────────────────────────────────────────
    router_entries = _read_router()
    model_counter: Counter = Counter()
    tokens_saved = 0
    tokens_by_model: Counter = Counter()
    enriched_count = 0
    plan_mode_count = 0
    hard_switch_count = 0
    latencies: list[int] = []
    savings_by_type: Counter = Counter()

    for e in router_entries:
        v, tier, lat_ms = _normalize_entry(e)
        model = v.get("model", "?")
        model_counter[model] += 1
        tok = (e.get("savings") or {}).get("tokens_estimated", 0) or 0
        tokens_saved += tok
        tokens_by_model[model] += tok
        if (e.get("enrichment") or {}).get("applied"):
            enriched_count += 1
            savings_by_type["enrichment"] += tok
        if v.get("plan_mode"):
            plan_mode_count += 1
            savings_by_type["plan_mode"] += tok
        if v.get("enforcement") == "hard_switch":
            hard_switch_count += 1
        if lat_ms > 0:
            latencies.append(lat_ms)

    router_total = len(router_entries)
    avg_lat = round(sum(latencies) / len(latencies)) if latencies else 0
    tokens_per_prompt = round(tokens_saved / router_total) if router_total else 0
    tokens_saved_usd = _estimate_usd(dict(tokens_by_model))

    router_by_session = _router_by_session(router_entries)
    daily = _daily_trends(router_entries)

    # ── Tasks ─────────────────────────────────────────────────────────────────
    if not _DB.exists():
        return {"error": "skill_hub.db not found"}

    conn = sqlite3.connect(str(_DB))
    conn.row_factory = sqlite3.Row

    tasks_raw = conn.execute(
        "SELECT id, title, summary, status, tags, session_id, "
        "created_at, updated_at, closed_at FROM tasks ORDER BY id DESC"
    ).fetchall()

    tasks: list[dict] = []
    open_count = closed_count = 0
    durations: list[float] = []
    tag_counter: Counter = Counter()

    log_counts = _task_log_counts()

    for row in tasks_raw:
        t = dict(row)
        dt_create = _parse_dt(t.get("created_at"))
        dt_close  = _parse_dt(t.get("closed_at"))
        duration_h: float | None = None
        if dt_create and dt_close:
            duration_h = (dt_close - dt_create).total_seconds() / 3600
            durations.append(duration_h)

        # per-task router data (if session matches)
        sid = t.get("session_id", "")
        session_entries = router_by_session.get(sid, [])
        task_models: Counter = Counter()
        for e in session_entries:
            v, _, _ = _normalize_entry(e)
            task_models[v.get("model", "?")] += 1
        task_prompts = len(session_entries)

        t["duration_h"] = duration_h
        t["duration_fmt"] = _fmt_hours(duration_h) if duration_h is not None else "—"
        t["log_count"] = log_counts.get(t["id"], 0)
        t["task_prompts"] = task_prompts
        t["task_models"] = dict(task_models.most_common(3))
        t["dominant_model"] = task_models.most_common(1)[0][0] if task_models else ""
        t["summary_short"] = (t.get("summary") or "")[:80]

        if t["status"] == "open":
            open_count += 1
        else:
            closed_count += 1

        for tag in (t.get("tags") or "").split(","):
            tag = tag.strip()
            if tag:
                tag_counter[tag] += 1

        tasks.append(t)

    # ── Sessions ──────────────────────────────────────────────────────────────
    sc_rows = conn.execute(
        "SELECT session_id, message_count, updated_at FROM session_context ORDER BY updated_at DESC"
    ).fetchall()
    total_messages = sum(r["message_count"] for r in sc_rows if r["message_count"])
    session_count = len(sc_rows)
    avg_msgs = round(total_messages / session_count) if session_count else 0

    # ── Skills ────────────────────────────────────────────────────────────────
    skills_total = conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0]
    try:
        inj_total = conn.execute("SELECT COUNT(*) FROM skill_injections").fetchone()[0]
    except Exception:
        inj_total = 0

    # ── Verdicts ──────────────────────────────────────────────────────────────
    verdict_stats: dict = {}
    if _VERDICTS.exists():
        try:
            vc = sqlite3.connect(str(_VERDICTS))
            vc.row_factory = sqlite3.Row
            vr = vc.execute(
                "SELECT COUNT(*) as n, SUM(hit_count) as hits, SUM(pinned) as pinned "
                "FROM command_verdicts"
            ).fetchone()
            verdict_stats = dict(vr)
        except Exception:
            pass

    conn.close()

    # ── Derived rates ─────────────────────────────────────────────────────────
    enrichment_rate  = round(100 * enriched_count / router_total, 1) if router_total else 0
    plan_mode_rate   = round(100 * plan_mode_count / router_total, 1) if router_total else 0
    closed_rate      = round(100 * closed_count / max(open_count + closed_count, 1), 1)
    avg_duration     = round(sum(durations) / len(durations), 1) if durations else 0
    model_pcts       = {m: round(100 * c / router_total, 1) for m, c in model_counter.most_common()}

    # ── Top router sessions ───────────────────────────────────────────────────
    top_sessions = []
    for sid, ents in sorted(router_by_session.items(), key=lambda x: -len(x[1]))[:8]:
        mc: Counter = Counter()
        pm = 0
        for e in ents:
            v2, _, _ = _normalize_entry(e)
            mc[v2.get("model", "?")] += 1
            if v2.get("plan_mode"):
                pm += 1
        top_sessions.append({
            "session_id": sid[:12],
            "prompts": len(ents),
            "dominant_model": mc.most_common(1)[0][0] if mc else "?",
            "models": dict(mc.most_common(3)),
            "plan_mode": pm,
        })

    return {
        # global
        "router_total":       router_total,
        "tokens_saved":       tokens_saved,
        "tokens_saved_usd":   tokens_saved_usd,
        "tokens_per_prompt":  tokens_per_prompt,
        "daily_trends":       daily,
        "avg_latency_ms":     avg_lat,
        "enriched_count":     enriched_count,
        "enrichment_rate":    enrichment_rate,
        "plan_mode_count":    plan_mode_count,
        "plan_mode_rate":     plan_mode_rate,
        "hard_switch_count":  hard_switch_count,
        "model_counts":       dict(model_counter.most_common()),
        "model_pcts":         model_pcts,
        "savings_by_type":    dict(savings_by_type),
        # tasks
        "task_total":         len(tasks),
        "open_count":         open_count,
        "closed_count":       closed_count,
        "closed_rate":        closed_rate,
        "avg_duration_h":     avg_duration,
        "avg_duration_fmt":   _fmt_hours(avg_duration) if avg_duration else "—",
        "tag_counts":         dict(tag_counter.most_common(15)),
        "tasks":              tasks,
        # sessions
        "session_count":      session_count,
        "total_messages":     total_messages,
        "avg_msgs_per_session": avg_msgs,
        # skills + verdicts
        "skills_total":       skills_total,
        "inj_total":          inj_total,
        "verdict_stats":      verdict_stats,
        # top sessions
        "top_router_sessions": top_sessions,
    }


# ── routes ────────────────────────────────────────────────────────────────────

@router.get("/report", response_class=HTMLResponse)
def report_page(request: Request) -> Any:
    data = _build_report()
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request, "report.html",
        {"active_tab": "report", **data},
    )


@router.get("/report/data")
def report_data(request: Request) -> JSONResponse:
    data = _build_report()
    # strip heavy per-task summaries for API response
    for t in data.get("tasks", []):
        t.pop("summary", None)
        t.pop("context", None)
        t.pop("vector", None)
    return JSONResponse(data)
