"""Router analytics page — visualises prompt-router audit log."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

router = APIRouter()

LOG_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "router.jsonl"
_MAX_ENTRIES = 2000


def _read_entries(n: int = _MAX_ENTRIES) -> list[dict]:
    if not LOG_PATH.exists():
        return []
    entries: list[dict] = []
    try:
        lines = LOG_PATH.read_text().splitlines()
        for line in lines[-n:]:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except OSError:
        pass
    return entries


def _compute_stats(entries: list[dict]) -> dict:
    if not entries:
        return {"total": 0}

    model_counts: Counter = Counter()
    tier_counts: Counter = Counter()
    domain_counts: Counter = Counter()
    scope_counts: Counter = Counter()
    enriched = 0
    plan_mode = 0
    compact_suggested = 0
    total_latency = 0
    tier1_lat = 0
    tier2_lat = 0
    tier3_lat = 0
    tier1_n = tier2_n = tier3_n = 0
    low_conf: list[dict] = []
    enforcements: Counter = Counter()
    tokens_saved = 0

    for e in entries:
        v = e.get("verdict", {})
        model_counts[v.get("model", "?")] += 1
        tier = v.get("tier", 0)
        tier_counts[tier] += 1
        for d in v.get("domain", []):
            domain_counts[d] += 1
        scope_counts[v.get("scope", "?")] += 1
        if v.get("plan_mode"):
            plan_mode += 1
        enf = v.get("enforcement", "none")
        if enf and enf != "none":
            enforcements[enf] += 1

        enr = e.get("enrichment", {})
        if enr.get("applied"):
            enriched += 1

        if e.get("compact", {}).get("suggested"):
            compact_suggested += 1

        lat = e.get("latency", {})
        t_total = lat.get("total_ms", 0) or 0
        total_latency += t_total
        if tier == 1:
            tier1_lat += lat.get("tier1_ms", 0) or 0
            tier1_n += 1
        elif tier == 2:
            tier2_lat += lat.get("tier2_ms", 0) or 0
            tier2_n += 1
        elif tier == 3:
            tier3_lat += lat.get("tier3_ms", 0) or 0
            tier3_n += 1

        savings = e.get("savings", {})
        tokens_saved += savings.get("tokens_estimated", 0) or 0

        conf = v.get("confidence", 1.0)
        if conf < 0.65:
            low_conf.append({
                "ts": (e.get("ts") or "")[:16],
                "tier": tier,
                "tier_label": v.get("tier_label", "?"),
                "model": v.get("model", "?"),
                "confidence": conf,
                "complexity": v.get("complexity", 0),
                "reasoning": v.get("reasoning", ""),
                "prompt": (e.get("prompt") or "")[:120],
                "enforcement": enf,
            })

    n = len(entries)
    return {
        "total": n,
        "model_counts": dict(model_counts.most_common()),
        "tier_counts": {str(k): v for k, v in sorted(tier_counts.items())},
        "domain_counts": dict(domain_counts.most_common(10)),
        "scope_counts": dict(scope_counts.most_common()),
        "enriched": enriched,
        "plan_mode": plan_mode,
        "compact_suggested": compact_suggested,
        "enforcements": dict(enforcements),
        "avg_latency_ms": round(total_latency / n) if n else 0,
        "avg_lat_t1": round(tier1_lat / tier1_n) if tier1_n else 0,
        "avg_lat_t2": round(tier2_lat / tier2_n) if tier2_n else 0,
        "avg_lat_t3": round(tier3_lat / tier3_n) if tier3_n else 0,
        "tokens_saved": tokens_saved,
        "low_conf_count": len(low_conf),
        "low_conf_samples": low_conf[:20],
    }


def _recent_rows(entries: list[dict], n: int = 50) -> list[dict]:
    out = []
    for e in reversed(entries[-n:]):
        v = e.get("verdict", {})
        lat = e.get("latency", {})
        enr = e.get("enrichment", {})
        out.append({
            "ts": (e.get("ts") or "")[:16],
            "session_id": (e.get("session_id") or "")[:8],
            "tier": v.get("tier", "?"),
            "tier_label": v.get("tier_label", "?"),
            "model": v.get("model", "?"),
            "confidence": round(v.get("confidence", 0), 2),
            "complexity": round(v.get("complexity", 0), 2),
            "scope": v.get("scope", "?"),
            "plan_mode": v.get("plan_mode", False),
            "enriched": enr.get("applied", False),
            "enforcement": v.get("enforcement", "none"),
            "total_ms": lat.get("total_ms", 0),
            "prompt": (e.get("prompt") or "")[:80],
        })
    return out


@router.get("/router", response_class=HTMLResponse)
def router_page(request: Request, n: int = 500) -> Any:
    entries = _read_entries(n)
    stats = _compute_stats(entries)
    recent = _recent_rows(entries)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request, "router.html",
        {
            "active_tab": "router",
            "stats": stats,
            "recent": recent,
            "n_loaded": len(entries),
            "log_exists": LOG_PATH.exists(),
        },
    )


@router.get("/router/stats", response_class=JSONResponse)
def router_stats_api(request: Request, n: int = 500) -> Any:
    entries = _read_entries(n)
    return _compute_stats(entries)
