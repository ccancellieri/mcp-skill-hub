"""In-memory + JSONL question queue with answer signalling.

A "question" is asked by a hook (e.g. auto_approve when ask_user_on_ambiguous
is true), persisted to disk, and surfaced in the UI. The user answers via
HTTP POST; the hook short-polls the answer file and acts on it.

Files (under ~/.claude/mcp-skill-hub/state/):
  questions.jsonl  — append-only history (open + answered)
  answers/<id>.json  — single-shot answer payload, removed once consumed

API:
  ask(prompt, command, tool_name, timeout=3.0) -> dict | None
  list_open() -> list[dict]
  list_recent(limit=20) -> list[dict]
  answer(qid, decision, reason="") -> bool
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

STATE_DIR = Path.home() / ".claude" / "mcp-skill-hub" / "state"
QFILE = STATE_DIR / "questions.jsonl"
ADIR = STATE_DIR / "answers"
_LOCK = threading.Lock()
# in-process subscriber events for SSE — set whenever queue mutates.
_NOTIFY: list[asyncio.Event] = []


def _read_all() -> list[dict[str, Any]]:
    if not QFILE.exists():
        return []
    out: list[dict[str, Any]] = []
    for raw in QFILE.read_text().splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return out


def _write_all(rows: list[dict[str, Any]]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = QFILE.with_suffix(".jsonl.tmp")
    with tmp.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(QFILE)


def _notify_all() -> None:
    for ev in list(_NOTIFY):
        try:
            ev.set()
        except Exception:
            pass


def subscribe() -> asyncio.Event:
    ev = asyncio.Event()
    _NOTIFY.append(ev)
    return ev


def unsubscribe(ev: asyncio.Event) -> None:
    try:
        _NOTIFY.remove(ev)
    except ValueError:
        pass


def enqueue_question(prompt: str, command: str = "",
                     tool_name: str = "") -> dict[str, Any]:
    entry = {
        "id": uuid.uuid4().hex[:12],
        "created_at": time.time(),
        "prompt": prompt.strip(),
        "command": command,
        "tool_name": tool_name,
        "status": "open",
    }
    with _LOCK:
        rows = _read_all()
        rows.append(entry)
        _write_all(rows)
    _notify_all()
    return entry


def list_open() -> list[dict[str, Any]]:
    with _LOCK:
        rows = _read_all()
    rows = [r for r in rows if r.get("status") == "open"]
    rows.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return rows


def list_recent(limit: int = 20) -> list[dict[str, Any]]:
    with _LOCK:
        rows = _read_all()
    rows.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return rows[:limit]


def answer(qid: str, decision: str, reason: str = "") -> bool:
    if decision not in ("allow", "deny"):
        return False
    with _LOCK:
        rows = _read_all()
        hit = False
        for r in rows:
            if r.get("id") == qid and r.get("status") == "open":
                r["status"] = "answered"
                r["decision"] = decision
                r["reason"] = reason
                r["answered_at"] = time.time()
                hit = True
                break
        if hit:
            _write_all(rows)
            ADIR.mkdir(parents=True, exist_ok=True)
            (ADIR / f"{qid}.json").write_text(json.dumps(
                {"id": qid, "decision": decision, "reason": reason,
                 "answered_at": time.time()}
            ))
    if hit:
        _notify_all()
    return hit


def poll_answer(qid: str, timeout: float = 3.0,
                interval: float = 0.2) -> dict[str, Any] | None:
    """Block up to ``timeout`` waiting for an answer file. Consumes it."""
    deadline = time.time() + max(0.0, timeout)
    path = ADIR / f"{qid}.json"
    while time.time() < deadline:
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                data = None
            try:
                path.unlink()
            except OSError:
                pass
            return data
        time.sleep(interval)
    return None


async def stream_events() -> AsyncIterator[str]:
    """Yield SSE-formatted strings whenever the queue changes."""
    ev = subscribe()
    try:
        # initial snapshot
        yield "event: snapshot\ndata: " + json.dumps(list_open()) + "\n\n"
        while True:
            try:
                await asyncio.wait_for(ev.wait(), timeout=15.0)
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue
            ev.clear()
            yield "event: change\ndata: " + json.dumps(list_open()) + "\n\n"
    finally:
        unsubscribe(ev)
