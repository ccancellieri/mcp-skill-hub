"""Index activity-log events into vector memory (issue #90 — Phase 1).

Design
------
LLM-free, embedding-only.  Reads the structured event log via
``store.get_events`` and upserts compact text documents into the ``logs``
vector namespace so log events become semantically searchable.

A bad event is skipped, never fatal.  All store / memory_index imports are
deferred inside functions so the module stays cheap to import.
"""
from __future__ import annotations

import json
import logging

_log = logging.getLogger(__name__)

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
