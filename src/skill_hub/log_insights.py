"""Index activity-log events into vector memory and surface actionable insights (issue #90).

Design
------
LLM-free, deterministic.  Reads the structured event log via ``store.get_events``
and upserts compact text documents into the ``logs`` vector namespace so log
events become semantically searchable.

Additional analysers (Phase 2):
- ``cluster_failures``: group recurring tool failures by (tool, normalised_error).
- ``skill_selection_stats``: per-skill injection counts + feedback helpful-rates.

A bad event is skipped, never fatal.  All store / memory_index imports are
deferred inside functions so the module stays cheap to import.
"""
from __future__ import annotations

import json
import logging
import re

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Error normalisation helpers (for cluster_failures)
# ---------------------------------------------------------------------------

# Patterns replaced with stable placeholders in error messages before grouping.
_NORM_PATTERNS: list[tuple[re.Pattern, str]] = [
    # 0x... hex addresses / memory addresses
    (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<hex>"),
    # UUIDs
    (re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    ), "<uuid>"),
    # File paths (Unix + Windows)
    (re.compile(r"(?:/[\w.\-/]+(?:\.\w+)?|[A-Za-z]:\\[\w\\\-. ]+)"), "<path>"),
    # Standalone integers (row ids, line numbers, etc.)
    (re.compile(r"\b\d{2,}\b"), "<N>"),
]


def _normalise_error(error: str) -> str:
    """Return a stable, de-noised version of an error string for grouping."""
    text = (error or "").strip().lower()
    for pat, repl in _NORM_PATTERNS:
        text = pat.sub(repl, text)
    # Collapse runs of whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Cap at 200 chars — long traces differ only in stack frames
    return text[:200]

# Maximum characters for a single event document before chunking kicks in.
_MAX_EVENT_DOC_CHARS = 512


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _summarise_payload(payload_raw: str) -> str:
    """Return a short, human-readable summary of a raw JSON payload string."""
    try:
        data = json.loads(payload_raw)
        if not isinstance(data, dict):
            return str(payload_raw)[:120]
        # Keep only the first few keys; skip internal/noise keys.
        _skip = {"ok", "_raw"}
        parts = []
        for k, v in data.items():
            if k in _skip:
                continue
            v_str = str(v)
            parts.append(f"{k}={v_str[:60]}" if len(v_str) > 60 else f"{k}={v_str}")
            if len(parts) >= 4:
                break
        return " ".join(parts) if parts else ""
    except Exception:  # noqa: BLE001
        return str(payload_raw)[:120]


def _build_event_doc(event: dict) -> tuple[str, str, dict]:
    """Convert a single event row into (doc_id, text, metadata).

    doc_id format: ``event:<session_id>:<ts>:<kind>``
    text format:   ``[<kind>] <tool_name> :: <payload_summary>``
    """
    kind = event.get("kind") or ""
    tool_name = event.get("tool_name") or ""
    session_id = event.get("session_id") or ""
    ts = event.get("ts") or 0.0
    payload_raw = event.get("payload") or "{}"

    payload_summary = _summarise_payload(str(payload_raw))
    tool_part = f" {tool_name}" if tool_name else ""
    body = f"[{kind}]{tool_part} :: {payload_summary}".strip(" ::")

    doc_id = f"event:{session_id}:{ts}:{kind}"
    metadata = {
        "kind": kind,
        "tool": tool_name,
        "session_id": session_id,
        "ts": ts,
        "source": "event_log",
    }
    return doc_id, body, metadata


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def index_recent_logs(
    hours: int = 24,
    limit: int = 1000,
    dry_run: bool = False,
) -> dict:
    """Embed recent events from the structured event log and upsert into ``logs``.

    Parameters
    ----------
    hours:
        Look back this many hours (default 24).
    limit:
        Maximum number of events to scan (default 1000).
    dry_run:
        When True, build the documents but make no DB writes.

    Returns
    -------
    dict with keys: scanned, indexed, by_kind, dry_run, namespace.
    """
    import time as _time

    from . import memory_index as _mi
    from . import store as _store_mod

    since = _time.time() - max(1, int(hours)) * 3600

    store = _store_mod.get_store()
    events = store.get_events(since=since, limit=limit)

    report: dict = {
        "scanned": len(events),
        "indexed": 0,
        "by_kind": {},
        "dry_run": dry_run,
        "namespace": "logs",
    }

    chunk_size, chunk_overlap = _mi._index_chunk_config(store, "logs")

    for event in events:
        try:
            doc_id, text, metadata = _build_event_doc(event)
            if not text:
                continue

            # Chunk only if the document exceeds the size threshold.
            if len(text) > _MAX_EVENT_DOC_CHARS:
                chunks = _mi._split_text(text, chunk_size, chunk_overlap)
            else:
                chunks = [text]

            kind = metadata.get("kind") or "unknown"

            for idx, chunk in enumerate(chunks):
                chunk_doc_id = f"{doc_id}:{idx}" if len(chunks) > 1 else doc_id
                if not dry_run:
                    try:
                        store.upsert_vector(
                            namespace="logs",
                            doc_id=chunk_doc_id,
                            text=chunk,
                            source="event_log",
                            metadata=metadata,
                        )
                        report["indexed"] += 1
                    except Exception as exc:  # noqa: BLE001
                        _log.debug("upsert_vector failed for %s: %s", chunk_doc_id, exc)
                else:
                    report["indexed"] += 1

            report["by_kind"][kind] = report["by_kind"].get(kind, 0) + 1

        except Exception as exc:  # noqa: BLE001
            _log.debug("log_insights: event processing error: %s", exc)
            continue

    return report


# ---------------------------------------------------------------------------
# Recurring-failure clustering (Phase 2 — Part 1)
# ---------------------------------------------------------------------------

def cluster_failures(
    hours: int = 24,
    limit: int = 2000,
    min_count: int = 2,
) -> dict:
    """Cluster recurring tool failures from the event log.

    Scans ``tool_result`` events whose payload has ``ok == False``, normalises
    the ``error`` field, and groups by ``(tool_name, normalised_error)``.

    Returns
    -------
    dict with keys:
        scanned  — total ``tool_result`` events examined
        clusters — list of cluster dicts sorted by count desc, each:
                   {tool, pattern, count, example, first_ts, last_ts, suggested_action}
        hours    — the look-back window used
    """
    import time as _time

    from . import store as _store_mod

    since = _time.time() - max(1, int(hours)) * 3600
    store = _store_mod.get_store()
    events = store.get_events(since=since, kind="tool_result", limit=limit)

    # Group by (tool_name, normalised_error)
    groups: dict[tuple[str, str], dict] = {}
    for event in events:
        payload_raw = event.get("payload") or "{}"
        try:
            payload = json.loads(payload_raw)
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("ok") is not False:
            continue
        tool = event.get("tool_name") or "(unknown)"
        error = str(payload.get("error") or "")
        normalised = _normalise_error(error)
        key = (tool, normalised)
        ts = event.get("ts", 0.0)
        if key not in groups:
            groups[key] = {
                "tool": tool,
                "pattern": normalised,
                "count": 0,
                "example": error,
                "first_ts": ts,
                "last_ts": ts,
            }
        g = groups[key]
        g["count"] += 1
        if ts < g["first_ts"]:
            g["first_ts"] = ts
        if ts > g["last_ts"]:
            g["last_ts"] = ts

    # Filter by min_count and build output
    import datetime as _dt

    def _fmt_ts(ts: float) -> str:
        try:
            return _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc).isoformat(
                timespec="seconds"
            ).replace("+00:00", "Z")
        except Exception:
            return str(ts)

    clusters = []
    for g in groups.values():
        if g["count"] < min_count:
            continue
        clusters.append({
            "tool": g["tool"],
            "pattern": g["pattern"],
            "count": g["count"],
            "example": g["example"],
            "first_ts": _fmt_ts(g["first_ts"]),
            "last_ts": _fmt_ts(g["last_ts"]),
            "suggested_action": (
                f"recurring `{g['tool']}` failure: {g['pattern'][:80]} — investigate"
            ),
        })

    clusters.sort(key=lambda c: c["count"], reverse=True)

    return {
        "scanned": len(events),
        "clusters": clusters,
        "hours": hours,
        "min_count": min_count,
    }


# ---------------------------------------------------------------------------
# Skill-selection metrics (Phase 2 — Part 3, feasible subset)
# ---------------------------------------------------------------------------

def skill_selection_stats(limit: int = 500) -> dict:
    """Per-skill injection counts and feedback helpful-rates.

    Uses ``skill_injections`` (injection frequency) and ``feedback``
    (helpful signal) tables. Results are joined on ``skill_id``.

    Attribution gap
    ---------------
    There is no direct link between an injection row and a subsequent
    ``record_feedback`` call — feedback is recorded by the user explicitly
    after a search, not tied to a single injection event.  This function
    therefore reports injection counts and feedback separately; true
    injected-vs-used attribution requires a new ``injection_used`` event kind
    (tracked as a follow-up).

    Returns
    -------
    dict with keys:
        skills             — list of per-skill dicts sorted by injections desc
        total_injections   — total injection rows examined
        total_feedback     — total feedback rows
    """
    from . import store as _store_mod

    store = _store_mod.get_store()
    conn = store._conn

    # Injection counts per skill
    try:
        inj_rows = conn.execute(
            "SELECT skill_id, COUNT(*) AS cnt "
            "FROM skill_injections "
            "GROUP BY skill_id "
            "ORDER BY cnt DESC "
            "LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        _log.debug("skill_selection_stats: injection query failed: %s", exc)
        inj_rows = []

    total_injections = sum(r["cnt"] for r in inj_rows)

    # Feedback per skill: helpful-rate = sum(helpful) / count(*)
    try:
        fb_rows = conn.execute(
            "SELECT skill_id, "
            "COUNT(*) AS n, "
            "SUM(helpful) AS n_helpful "
            "FROM feedback "
            "GROUP BY skill_id"
        ).fetchall()
    except Exception as exc:  # noqa: BLE001
        _log.debug("skill_selection_stats: feedback query failed: %s", exc)
        fb_rows = []

    total_feedback = sum(r["n"] for r in fb_rows)
    feedback_map: dict[str, dict] = {
        r["skill_id"]: {"n": r["n"], "n_helpful": r["n_helpful"]}
        for r in fb_rows
    }

    # Build per-skill result rows
    skills = []
    injection_skill_ids = {r["skill_id"] for r in inj_rows}

    for r in inj_rows:
        sid = r["skill_id"]
        fb = feedback_map.get(sid)
        helpful_rate: float | None = None
        fb_n = 0
        if fb and fb["n"] > 0:
            fb_n = fb["n"]
            helpful_rate = fb["n_helpful"] / fb["n"]

        # Status heuristic: injected many times but never given positive feedback
        if fb_n > 0 and helpful_rate is not None and helpful_rate == 0.0:
            status = "review: never-helpful"
        elif fb_n == 0 and r["cnt"] >= 5:
            status = "no feedback yet"
        else:
            status = ""

        skills.append({
            "skill_id": sid,
            "injections": r["cnt"],
            "feedback_n": fb_n,
            "helpful_rate": helpful_rate,
            "status": status,
        })

    # Also include skills that have feedback but zero recent injections
    for sid, fb in feedback_map.items():
        if sid not in injection_skill_ids:
            helpful_rate = fb["n_helpful"] / fb["n"] if fb["n"] > 0 else None
            skills.append({
                "skill_id": sid,
                "injections": 0,
                "feedback_n": fb["n"],
                "helpful_rate": helpful_rate,
                "status": "feedback only (no recent injections)",
            })

    skills.sort(key=lambda x: x["injections"], reverse=True)

    return {
        "skills": skills,
        "total_injections": total_injections,
        "total_feedback": total_feedback,
    }
