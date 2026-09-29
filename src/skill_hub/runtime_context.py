"""Bounded, fail-soft runtime metadata observations.

Runtime data is diagnostic evidence supplied by a client integration.  It is
kept separate from prompt assembly so unavailable or malformed telemetry can
never prevent deterministic context retrieval.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .config import CONFIG_PATH

RUNTIME_DB_PATH = CONFIG_PATH.parent / "runtime-context.db"
_MAX_TEXT = 512
_SOURCES = {
    "native_event", "native_api", "mcp_client_info", "caller_reported",
    "adapter_reported", "configured", "unknown",
}
_SOURCE_RANK = {
    "unknown": 0, "configured": 1, "caller_reported": 2,
    "adapter_reported": 3, "mcp_client_info": 4, "native_api": 5,
    "native_event": 6,
}
_FIELDS = (
    "client_id", "client_version", "session_id", "turn_id", "model_id",
    "model_provider", "model_display_name", "effort_value", "effort_scheme",
)


def _text(value: Any) -> str:
    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        return ""
    return str(value).strip()[:_MAX_TEXT]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _observed_at(value: Any) -> str:
    text = _text(value)
    if not text:
        return _utc_now()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    except ValueError:
        return _utc_now()


def normalize_runtime(
    value: Any, *, default_source: str = "caller_reported",
    honor_provenance: bool = True,
) -> dict[str, Any]:
    """Normalize the optional wire object without guessing absent fields."""
    if not isinstance(value, dict):
        value = {}
    client = value.get("client") if isinstance(value.get("client"), dict) else {}
    session = value.get("session") if isinstance(value.get("session"), dict) else {}
    model = value.get("model") if isinstance(value.get("model"), dict) else {}
    effort = value.get("effort") if isinstance(value.get("effort"), dict) else {}
    result: dict[str, Any] = {
        "client_id": _text(client.get("id", value.get("client_id"))),
        "client_version": _text(client.get("version", value.get("client_version"))),
        "session_id": _text(session.get("id", value.get("session_id"))),
        "turn_id": _text(session.get("turn_id", value.get("turn_id"))),
        "model_id": _text(model.get("id", value.get("model_id"))),
        "model_provider": _text(model.get("provider", value.get("model_provider"))),
        "model_display_name": _text(
            model.get("display_name", value.get("model_display_name"))
        ),
        "effort_value": _text(effort.get("value", value.get("effort_value"))),
        "effort_scheme": _text(effort.get("scheme", value.get("effort_scheme"))),
        "observed_at": _observed_at(value.get("observed_at")),
    }
    source = default_source if default_source in _SOURCES else "unknown"
    supplied = value.get("provenance") if isinstance(value.get("provenance"), dict) else {}
    provenance: dict[str, str] = {}
    for field in _FIELDS:
        if not result[field]:
            continue
        field_source = _text(supplied.get(field)) if honor_provenance else ""
        provenance[field] = field_source if field_source in _SOURCES else source
    supplied_times = (
        value.get("observed_at_by_field")
        if honor_provenance and isinstance(value.get("observed_at_by_field"), dict)
        else {}
    )
    result["observed_at_by_field"] = {
        field: _observed_at(supplied_times.get(field, result["observed_at"]))
        for field in provenance
    }
    result["provenance"] = provenance
    sources = set(provenance.values())
    result["source"] = next(iter(sources)) if len(sources) == 1 else (
        "mixed" if sources else "unknown"
    )
    return result


def _connect(*, create: bool = True) -> sqlite3.Connection:
    if create:
        RUNTIME_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        target = str(RUNTIME_DB_PATH)
    else:
        target = f"file:{RUNTIME_DB_PATH}?mode=ro"
    connection = sqlite3.connect(target, timeout=0.05, uri=not create)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=50")
    if create:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS runtime_observations (
                client_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                observed_at TEXT NOT NULL,
                PRIMARY KEY (client_id, session_id)
            )
        """)
    return connection


def _merge(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    """Merge newer evidence field-by-field without downgrading provenance."""
    try:
        if datetime.fromisoformat(incoming["observed_at"]) < datetime.fromisoformat(
            existing["observed_at"]
        ):
            return existing
    except (KeyError, TypeError, ValueError):
        pass
    merged = dict(existing)
    merged_provenance = dict(existing.get("provenance", {}))
    merged_times = dict(existing.get("observed_at_by_field", {}))
    incoming_provenance = incoming.get("provenance", {})
    incoming_times = incoming.get("observed_at_by_field", {})
    incoming_model = incoming.get("model_id", "")
    model_source = incoming_provenance.get("model_id", "unknown")
    old_model_source = merged_provenance.get("model_id", "unknown")
    accepted_model_change = bool(
        incoming_model
        and incoming_model != existing.get("model_id", "")
        and _SOURCE_RANK.get(model_source, 0) >= _SOURCE_RANK.get(old_model_source, 0)
    )
    rejected_model_change = bool(
        incoming_model
        and incoming_model != existing.get("model_id", "")
        and not accepted_model_change
    )
    if accepted_model_change:
        for field in (
            "model_provider", "model_display_name", "effort_value", "effort_scheme",
        ):
            merged[field] = ""
            merged_provenance.pop(field, None)
            merged_times.pop(field, None)
    accepted_any = False
    for field in _FIELDS:
        if rejected_model_change and field in {
            "model_id", "model_provider", "model_display_name",
            "effort_value", "effort_scheme",
        }:
            continue
        value = incoming.get(field, "")
        if not value:
            continue
        new_source = incoming_provenance.get(field, "unknown")
        old_source = merged_provenance.get(field, "unknown")
        if _SOURCE_RANK.get(new_source, 0) < _SOURCE_RANK.get(old_source, 0):
            continue
        merged[field] = value
        merged_provenance[field] = new_source
        merged_times[field] = incoming_times.get(field, incoming["observed_at"])
        accepted_any = True
    if not accepted_any:
        return existing
    native_model_snapshot_without_effort = bool(
        incoming_model and not incoming.get("effort_value")
        and _SOURCE_RANK.get(model_source, 0) >= _SOURCE_RANK["adapter_reported"]
        and _SOURCE_RANK.get(model_source, 0) >= _SOURCE_RANK.get(old_model_source, 0)
        and _SOURCE_RANK.get(model_source, 0) >= _SOURCE_RANK.get(
            merged_provenance.get("effort_value", "unknown"), 0,
        )
    )
    if (accepted_model_change or native_model_snapshot_without_effort) and not incoming.get("effort_value"):
        merged["effort_value"] = ""
        merged["effort_scheme"] = ""
        merged_provenance.pop("effort_value", None)
        merged_provenance.pop("effort_scheme", None)
        merged_times.pop("effort_value", None)
        merged_times.pop("effort_scheme", None)
    merged["observed_at"] = incoming["observed_at"]
    merged["provenance"] = merged_provenance
    merged["observed_at_by_field"] = merged_times
    sources = set(merged_provenance.values())
    merged["source"] = next(iter(sources)) if len(sources) == 1 else (
        "mixed" if sources else "unknown"
    )
    return merged


def observe_runtime(
    value: Any, *, default_source: str = "caller_reported",
    honor_provenance: bool | None = None,
) -> dict[str, Any] | None:
    """Persist one client/session snapshot; return ``None`` on any failure."""
    try:
        normalized = normalize_runtime(
            value, default_source=default_source,
            honor_provenance=False if honor_provenance is None else honor_provenance,
        )
        if not normalized["client_id"] or not normalized["session_id"]:
            return None
        connection = _connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT payload FROM runtime_observations WHERE client_id=? AND session_id=?",
                (normalized["client_id"], normalized["session_id"]),
            ).fetchone()
            if current:
                parsed = json.loads(current["payload"])
                if isinstance(parsed, dict):
                    normalized = _merge(parsed, normalized)
            payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
            connection.execute(
                """INSERT INTO runtime_observations
                   (client_id, session_id, payload, observed_at) VALUES (?, ?, ?, ?)
                   ON CONFLICT(client_id, session_id) DO UPDATE SET
                       payload=excluded.payload, observed_at=excluded.observed_at""",
                (normalized["client_id"], normalized["session_id"], payload,
                 normalized["observed_at"]),
            )
            connection.commit()
        finally:
            connection.close()
        return normalized
    except Exception:
        return None


def list_runtime_sessions(
    *, limit: int = 100, max_age_seconds: float | None = None,
) -> list[dict[str, Any]]:
    """Return newest normalized runtime snapshots, or an empty list on failure."""
    try:
        if not RUNTIME_DB_PATH.is_file():
            return []
        bounded_limit = max(1, min(int(limit), 500))
        parameters: list[Any] = []
        where = ""
        if max_age_seconds is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(
                seconds=max(0.0, float(max_age_seconds))
            )
            where = "WHERE observed_at >= ?"
            parameters.append(cutoff.isoformat())
        parameters.append(bounded_limit)
        connection = _connect(create=False)
        try:
            rows = connection.execute(
                f"SELECT payload FROM runtime_observations {where} "
                "ORDER BY observed_at DESC LIMIT ?",
                parameters,
            ).fetchall()
        finally:
            connection.close()
        result = []
        for row in rows:
            parsed = json.loads(row["payload"])
            if isinstance(parsed, dict):
                result.append(normalize_runtime(
                    parsed, default_source="unknown", honor_provenance=True,
                ))
        return result
    except Exception:
        return []
