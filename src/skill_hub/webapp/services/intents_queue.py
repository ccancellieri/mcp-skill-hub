"""JSONL-backed queue for chrome-devtools intents.

File: ~/.claude/mcp-skill-hub/state/chrome_intents.jsonl
Each line is a JSON object: {id, created_at, url, action, note, status,
done_at?}.

status ∈ {"pending", "done", "cancelled"}.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any

QUEUE_PATH = (
    Path.home() / ".claude" / "mcp-skill-hub" / "state" / "chrome_intents.jsonl"
)
_LOCK = threading.Lock()


def _read_all() -> list[dict[str, Any]]:
    if not QUEUE_PATH.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for raw in QUEUE_PATH.read_text().splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                out.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    except OSError:
        return []
    return out


def _write_all(rows: list[dict[str, Any]]) -> None:
    QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = QUEUE_PATH.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(QUEUE_PATH)


def list_intents(include_done: bool = True) -> list[dict[str, Any]]:
    with _LOCK:
        rows = _read_all()
    if not include_done:
        rows = [r for r in rows if r.get("status") == "pending"]
    rows.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return rows


def enqueue(url: str, action: str = "navigate", note: str = "") -> dict[str, Any]:
    entry = {
        "id": uuid.uuid4().hex[:12],
        "created_at": time.time(),
        "url": url.strip(),
        "action": action.strip() or "navigate",
        "note": note.strip(),
        "status": "pending",
    }
    with _LOCK:
        rows = _read_all()
        rows.append(entry)
        _write_all(rows)
    return entry


def mark_done(intent_id: str, status: str = "done") -> bool:
    if status not in ("done", "cancelled", "pending"):
        return False
    with _LOCK:
        rows = _read_all()
        hit = False
        for r in rows:
            if r.get("id") == intent_id:
                r["status"] = status
                if status != "pending":
                    r["done_at"] = time.time()
                hit = True
                break
        if hit:
            _write_all(rows)
    return hit


def pending_count() -> int:
    return sum(1 for r in list_intents() if r.get("status") == "pending")
