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


def _normalize_entry(e: dict) -> tuple[dict, int, int]:
    """Return (verdict, tier_int, latency_ms) handling both old and new log formats."""
    v = e.get("verdict", {})
    # tier: new format uses "tier", old format uses "tier_used"
    tier = v.get("tier_used") if v.get("tier_used") is not None else v.get("tier", 1)
    # total latency: new format uses nested latency.total_ms, old uses top-level latency_ms
    lat_obj = e.get("latency") or {}
    lat_ms = lat_obj.get("total_ms") or e.get("latency_ms") or 0
    return v, int(tier), int(lat_ms)


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
    hard_switches = 0
    forced_samples: list[dict] = []
    enforcements: Counter = Counter()
    tokens_saved = 0

    # Per-tier latency from actual tier breakdown fields (not bucketed by winner)
    tier1_ms_list: list[int] = []
    tier2_ms_list: list[int] = []
    tier3_ms_list: list[int] = []

    # T2/T3 attempt tracking: how many times did each tier run
    t2_attempts = 0   # tier2_ms > 0
    t2_successes = 0  # verdict.tier_used == 2
    t3_attempts = 0

    # Confidence distribution — buckets aligned with router thresholds
    # T2 gate: < 0.85, T3 gate: < 0.7, hard-switch: >= 0.9
    conf_buckets: dict[str, int] = {"<0.5": 0, "0.5-0.7": 0, "0.7-0.85": 0, "0.85-0.9": 0, "≥0.9": 0}

    # Classifier scores
    complexities: list[float] = []

    # Skills preloaded
    skills_preloaded_prompts = 0
    skills_preloaded_total = 0

    # Enrichment chars added
    enrich_chars_list: list[int] = []

    for e in entries:
        v, tier, lat_ms = _normalize_entry(e)

        model_counts[v.get("model", "?")] += 1
        tier_counts[tier] += 1

        for d in v.get("domain") or []:
            domain_counts[d] += 1

        scope = v.get("scope") or "?"
        scope_counts[scope] += 1

        if v.get("plan_mode"):
            plan_mode += 1

        enf = v.get("enforcement") or "none"
        if enf and enf != "none":
            enforcements[enf] += 1
        if enf == "hard_switch":
            hard_switches += 1

        if (e.get("enrichment") or {}).get("applied"):
            enriched += 1
        if (e.get("compact") or {}).get("suggested"):
            compact_suggested += 1

        total_latency += lat_ms

        # Per-tier latency from actual breakdown (fixes old approach of
        # grouping total latency by winning tier)
        lat_obj = e.get("latency") or {}
        t1 = int(lat_obj.get("tier1_ms") or 0)
        t2 = int(lat_obj.get("tier2_ms") or 0)
        t3 = int(lat_obj.get("tier3_ms") or 0)
        if t1 > 0:
            tier1_ms_list.append(t1)
        if t2 > 0:
            tier2_ms_list.append(t2)
            t2_attempts += 1
        if t3 > 0:
            tier3_ms_list.append(t3)
            t3_attempts += 1

        # T2/T3 verdict wins (when tier_used was promoted)
        if tier == 2:
            t2_successes += 1

        tokens_saved += (e.get("savings") or {}).get("tokens_estimated", 0) or 0

        # Confidence buckets (aligned with router gate thresholds)
        conf = v.get("confidence", 1.0)
        if conf < 0.5:
            conf_buckets["<0.5"] += 1
        elif conf < 0.7:
            conf_buckets["0.5-0.7"] += 1
        elif conf < 0.85:
            conf_buckets["0.7-0.85"] += 1
        elif conf < 0.9:
            conf_buckets["0.85-0.9"] += 1
        else:
            conf_buckets["≥0.9"] += 1

        # Complexity score
        cplx = v.get("complexity")
        if cplx is not None:
            complexities.append(float(cplx))

        # Skills preloaded
        skills_obj = e.get("skills") or {}
        preloaded = skills_obj.get("preloaded") or e.get("preload_skills") or []
        if preloaded:
            skills_preloaded_prompts += 1
            skills_preloaded_total += len(preloaded)

        # Enrichment chars
        enrich = e.get("enrichment") or {}
        if enrich.get("applied") and enrich.get("chars_added"):
            enrich_chars_list.append(int(enrich["chars_added"]))

        # Forced samples: hard_switch enforcement OR low confidence (< 0.5)
        if enf == "hard_switch" or conf < 0.5:
            prompt_text = (e.get("prompt") or e.get("prompt_preview") or "")[:120]
            forced_samples.append({
                "ts": (e.get("ts") or "")[:16],
                "tier": tier,
                "tier_label": v.get("tier_label") or f"T{tier}",
                "model": v.get("model", "?"),
                "prev_model": v.get("prev_model", ""),
                "confidence": conf,
                "complexity": v.get("complexity", 0),
                "reasoning": v.get("reasoning", ""),
                "prompt": prompt_text,
                "enforcement": enf,
            })

    n = len(entries)

    def _avg(lst: list[int]) -> int:
        return round(sum(lst) / len(lst)) if lst else 0

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
        "hard_switches": hard_switches,
        "avg_latency_ms": round(total_latency / n) if n else 0,
        # Fixed: actual per-tier execution times (not total latency bucketed by winner)
        "avg_lat_t1": _avg(tier1_ms_list),
        "avg_lat_t2": _avg(tier2_ms_list),
        "avg_lat_t3": _avg(tier3_ms_list),
        "t2_attempts": t2_attempts,
        "t2_successes": t2_successes,
        "t3_attempts": t3_attempts,
        "tokens_saved": tokens_saved,
        "forced_count": len(forced_samples),
        "forced_samples": forced_samples[:20],
        # Confidence distribution buckets
        "conf_buckets": conf_buckets,
        # Classifier stats
        "avg_complexity": round(sum(complexities) / len(complexities), 2) if complexities else 0,
        # Skills preloading
        "skills_preloaded_prompts": skills_preloaded_prompts,
        "skills_preloaded_total": skills_preloaded_total,
        # Enrichment
        "avg_enrich_chars": round(sum(enrich_chars_list) / len(enrich_chars_list)) if enrich_chars_list else 0,
        # Date range
        "first_ts": (entries[0].get("ts") or "")[:10],
        "last_ts": (entries[-1].get("ts") or "")[:10],
        # keep old key for template compatibility
        "low_conf_count": len(forced_samples),
        "low_conf_samples": forced_samples[:20],
    }


def _recent_rows(entries: list[dict], n: int = 50) -> list[dict]:
    out = []
    for e in reversed(entries[-n:]):
        v, tier, lat_ms = _normalize_entry(e)
        prompt_text = (e.get("prompt") or e.get("prompt_preview") or "")[:80]
        out.append({
            "ts": (e.get("ts") or "")[:16],
            "session_id": (e.get("session_id") or "")[:8],
            "tier": tier,
            "tier_label": v.get("tier_label") or f"T{tier}",
            "model": v.get("model", "?"),
            "confidence": round(v.get("confidence", 0), 2),
            "complexity": round(v.get("complexity", 0), 2),
            "scope": v.get("scope") or "?",
            "plan_mode": v.get("plan_mode", False),
            "enriched": (e.get("enrichment") or {}).get("applied", False),
            "enforcement": v.get("enforcement") or "none",
            "total_ms": lat_ms,
            "prompt": prompt_text,
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
            "n_req": n,
            "log_exists": LOG_PATH.exists(),
        },
    )


@router.get("/router/stats", response_class=JSONResponse)
def router_stats_api(request: Request, n: int = 500) -> Any:
    entries = _read_entries(n)
    return _compute_stats(entries)
