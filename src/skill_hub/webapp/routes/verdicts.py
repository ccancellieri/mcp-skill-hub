"""Verdicts route — command_verdicts CRUD, filters, bulk actions."""
from __future__ import annotations

import json
import shlex
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

import sys
_HOOKS_DIR = Path(__file__).resolve().parents[4] / "hooks"
if str(_HOOKS_DIR) not in sys.path:
    sys.path.insert(0, str(_HOOKS_DIR))
import verdict_cache  # noqa: E402

router = APIRouter()

ALLOW_YML = Path.home() / ".claude" / "skill-hub-allow.yml"


def _fetch_rows(
    tool: str = "",
    source: str = "",
    decision: str = "",
    min_hits: int = 0,
    q: str = "",
    limit: int = 500,
) -> list[dict]:
    conn = verdict_cache.connect()
    sql = "SELECT * FROM command_verdicts WHERE 1=1"
    params: list = []
    if tool:
        sql += " AND tool_name = ?"
        params.append(tool)
    if source:
        sql += " AND source = ?"
        params.append(source)
    if decision:
        sql += " AND decision = ?"
        params.append(decision)
    if min_hits:
        sql += " AND hit_count >= ?"
        params.append(min_hits)
    if q:
        sql += " AND command LIKE ?"
        params.append(f"%{q}%")
    sql += " ORDER BY last_used_at DESC LIMIT ?"
    params.append(limit)
    rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def _distinct(col: str) -> list[str]:
    conn = verdict_cache.connect()
    rows = conn.execute(
        f"SELECT DISTINCT {col} FROM command_verdicts WHERE {col} IS NOT NULL "
        f"ORDER BY {col}"
    ).fetchall()
    return [r[0] for r in rows]


@router.get("/verdicts", response_class=HTMLResponse)
def verdicts_page(request: Request) -> Any:
    rows = _fetch_rows()
    # inline stats
    total = len(rows)
    allow_n = sum(1 for r in rows if r.get("decision") == "allow")
    deny_n  = sum(1 for r in rows if r.get("decision") == "deny")
    total_hits = sum(r.get("hit_count") or 0 for r in rows)
    avg_conf = (
        sum(r.get("confidence") or 0 for r in rows) / total
        if total else 0.0
    )
    pinned_n = sum(1 for r in rows if r.get("pinned"))
    stats = {
        "total": total,
        "allow": allow_n,
        "deny": deny_n,
        "total_hits": total_hits,
        "avg_conf": round(avg_conf, 3),
        "pinned": pinned_n,
    }
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "verdicts.html",
        {
            "rows": rows,
            "stats": stats,
            "tools": _distinct("tool_name"),
            "sources": _distinct("source"),
            "active_tab": "verdicts",
        },
    )


@router.get("/verdicts/partial", response_class=HTMLResponse)
def verdicts_partial(
    request: Request,
    tool: str = "",
    source: str = "",
    decision: str = "",
    min_hits: int = 0,
    q: str = "",
) -> Any:
    rows = _fetch_rows(tool, source, decision, min_hits, q)
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request, "_verdict_rows.html", {"rows": rows}
    )


def _row_html(templates, request, row: dict) -> HTMLResponse:
    return templates.TemplateResponse(
        request, "_verdict_row.html", {"r": row}
    )


@router.post("/verdicts/{cmd_hash}/delete", response_class=HTMLResponse)
def verdict_delete(cmd_hash: str) -> HTMLResponse:
    conn = verdict_cache.connect()
    verdict_cache.delete(conn, cmd_hash)
    return HTMLResponse("")  # row removed


@router.post("/verdicts/{cmd_hash}/flip", response_class=HTMLResponse)
def verdict_flip(cmd_hash: str, request: Request) -> Any:
    conn = verdict_cache.connect()
    row = conn.execute(
        "SELECT * FROM command_verdicts WHERE cmd_hash = ?", (cmd_hash,)
    ).fetchone()
    if not row:
        return HTMLResponse("", status_code=404)
    new = "deny" if row["decision"] == "allow" else "allow"
    verdict_cache.update_decision(conn, cmd_hash, new)
    row = dict(conn.execute(
        "SELECT * FROM command_verdicts WHERE cmd_hash = ?", (cmd_hash,)
    ).fetchone())
    return _row_html(request.app.state.templates, request, row)


@router.post("/verdicts/{cmd_hash}/pin", response_class=HTMLResponse)
def verdict_pin(cmd_hash: str, request: Request) -> Any:
    conn = verdict_cache.connect()
    row = conn.execute(
        "SELECT * FROM command_verdicts WHERE cmd_hash = ?", (cmd_hash,)
    ).fetchone()
    if not row:
        return HTMLResponse("", status_code=404)
    verdict_cache.set_pinned(conn, cmd_hash, not bool(row["pinned"]))
    row = dict(conn.execute(
        "SELECT * FROM command_verdicts WHERE cmd_hash = ?", (cmd_hash,)
    ).fetchone())
    return _row_html(request.app.state.templates, request, row)


class BulkBody(BaseModel):
    hashes: list[str]


@router.post("/verdicts/bulk/delete")
def verdicts_bulk_delete(body: BulkBody) -> JSONResponse:
    conn = verdict_cache.connect()
    n = 0
    for h in body.hashes:
        if verdict_cache.delete(conn, h):
            n += 1
    return JSONResponse({"deleted": n})


def _extract_prefix(command: str) -> str:
    try:
        parts = shlex.split(command)
    except ValueError:
        parts = command.split()
    if not parts:
        return ""
    # Two-word prefix if it looks safe (e.g. `git status`), else first token
    if len(parts) >= 2 and parts[0] in {"git", "npm", "uv", "pnpm", "yarn",
                                        "docker", "kubectl", "pytest", "pip"}:
        return f"{parts[0]} {parts[1]}"
    return parts[0]


def _append_prefixes(new_prefixes: list[str]) -> int:
    if not ALLOW_YML.exists():
        ALLOW_YML.parent.mkdir(parents=True, exist_ok=True)
        ALLOW_YML.write_text("safe_bash_prefixes:\n")
    text = ALLOW_YML.read_text()
    # Parse existing prefixes under safe_bash_prefixes: block
    lines = text.splitlines()
    in_block = False
    existing: set[str] = set()
    block_end_idx = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("safe_bash_prefixes:"):
            in_block = True
            block_end_idx = i
            continue
        if in_block:
            stripped = line.lstrip()
            if stripped.startswith("- "):
                existing.add(stripped[2:].strip())
                block_end_idx = i
            elif line and not line.startswith((" ", "\t")):
                in_block = False
    added = 0
    to_add = [p for p in new_prefixes if p and p not in existing]
    if not to_add:
        return 0
    # Insert after block_end_idx
    insert_lines = [f"  - {p}" for p in to_add]
    if block_end_idx < 0:
        # No block — append at end
        lines.append("safe_bash_prefixes:")
        lines.extend(insert_lines)
    else:
        lines[block_end_idx + 1:block_end_idx + 1] = insert_lines
    ALLOW_YML.write_text("\n".join(lines) + ("\n" if not text.endswith("\n") else ""))
    added = len(to_add)
    return added


@router.post("/verdicts/bulk/promote")
def verdicts_bulk_promote(body: BulkBody) -> JSONResponse:
    conn = verdict_cache.connect()
    prefixes: list[str] = []
    for h in body.hashes:
        row = conn.execute(
            "SELECT tool_name, command FROM command_verdicts WHERE cmd_hash = ?",
            (h,),
        ).fetchone()
        if not row:
            continue
        if row["tool_name"] != "Bash":
            continue
        p = _extract_prefix(row["command"])
        if p:
            prefixes.append(p)
    added = _append_prefixes(sorted(set(prefixes)))
    return JSONResponse({"added": added, "prefixes": sorted(set(prefixes))})
