"""JSON API endpoints for the interactive dashboard.

Each function takes a SkillStore (and optional body dict) and returns a
JSON-serializable dict or list. No HTTP dependencies here — makes these
unit-testable without spinning up a server.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

# Import verdict_cache from the sibling hooks/ directory.
_HOOKS = Path(__file__).resolve().parent.parent.parent / "hooks"
if str(_HOOKS) not in sys.path:
    sys.path.insert(0, str(_HOOKS))
try:
    import verdict_cache  # type: ignore  # noqa: E402
except ImportError:  # pragma: no cover
    verdict_cache = None  # type: ignore

from . import dashboard as _dashboard  # noqa: E402
from . import vector_viz  # noqa: E402

ALLOW_YAML = Path.home() / ".claude" / "skill-hub-allow.yml"


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def _vc_conn() -> sqlite3.Connection | None:
    if verdict_cache is None:
        return None
    try:
        return verdict_cache.connect()
    except sqlite3.Error:
        return None


def api_verdicts_list(store: Any, body: dict | None = None) -> list[dict]:
    conn = _vc_conn()
    if conn is None:
        return []
    rows = conn.execute(
        "SELECT cmd_hash, tool_name, command, decision, source, confidence, "
        "hit_count, created_at, last_used_at, "
        "COALESCE(pinned, 0) as pinned, "
        "CASE WHEN vector IS NOT NULL THEN 1 ELSE 0 END as has_vector "
        "FROM command_verdicts ORDER BY last_used_at DESC LIMIT 2000"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def api_verdicts_delete(store: Any, body: dict) -> dict:
    conn = _vc_conn()
    if conn is None:
        return {"ok": False, "error": "verdict cache unavailable"}
    hashes = body.get("cmd_hashes") or ([body["cmd_hash"]] if body.get("cmd_hash") else [])
    n = 0
    for h in hashes:
        if verdict_cache.delete(conn, h):
            n += 1
    conn.close()
    return {"ok": True, "deleted": n}


def api_verdicts_flip(store: Any, body: dict) -> dict:
    conn = _vc_conn()
    if conn is None:
        return {"ok": False, "error": "verdict cache unavailable"}
    h = body.get("cmd_hash")
    if not h:
        return {"ok": False, "error": "cmd_hash required"}
    row = conn.execute(
        "SELECT decision FROM command_verdicts WHERE cmd_hash = ?", (h,)
    ).fetchone()
    if not row:
        conn.close()
        return {"ok": False, "error": "not found"}
    new = "deny" if row["decision"] == "allow" else "allow"
    verdict_cache.update_decision(conn, h, new)
    conn.close()
    return {"ok": True, "decision": new}


def api_verdicts_pin(store: Any, body: dict) -> dict:
    conn = _vc_conn()
    if conn is None:
        return {"ok": False, "error": "verdict cache unavailable"}
    h = body.get("cmd_hash")
    pinned = bool(body.get("pinned", True))
    if not h:
        return {"ok": False, "error": "cmd_hash required"}
    verdict_cache.set_pinned(conn, h, pinned)
    conn.close()
    return {"ok": True, "pinned": pinned}


def api_verdicts_promote_yaml(store: Any, body: dict) -> dict:
    """Append selected commands as safe_bash_prefixes: to the global allow YAML."""
    conn = _vc_conn()
    if conn is None:
        return {"ok": False, "error": "verdict cache unavailable"}
    hashes = body.get("cmd_hashes") or []
    cmds: list[str] = []
    for h in hashes:
        row = conn.execute(
            "SELECT command, tool_name FROM command_verdicts WHERE cmd_hash = ?", (h,)
        ).fetchone()
        if row and row["tool_name"] == "Bash":
            cmds.append(row["command"])
    conn.close()
    if not cmds:
        return {"ok": False, "error": "no Bash commands selected"}
    try:
        ALLOW_YAML.parent.mkdir(parents=True, exist_ok=True)
        existing = ""
        if ALLOW_YAML.exists():
            existing = ALLOW_YAML.read_text()
        lines = existing.splitlines() if existing else []
        if not any(l.strip().startswith("safe_bash_prefixes:") for l in lines):
            lines.append("safe_bash_prefixes:")
        for c in cmds:
            # Use first token (prefix) not full command to avoid lock-in
            prefix = c.split()[0] if c else ""
            if prefix and not any(prefix in l for l in lines):
                lines.append(f'  - "{prefix}"')
        ALLOW_YAML.write_text("\n".join(lines) + "\n")
    except OSError as e:
        return {"ok": False, "error": str(e)}
    return {"ok": True, "promoted": len(cmds), "yaml": str(ALLOW_YAML)}


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------

def api_tasks_list(store: Any, body: dict | None = None) -> dict:
    body = body or {}
    status = body.get("status", "all")
    q = body.get("q", "").strip()
    if q:
        rows = store.search_tasks_text(q, status=status)
    else:
        rows = store.list_tasks(status=status)
    return {"tasks": [dict(r) for r in rows]}


def api_tasks_rename(store: Any, body: dict) -> dict:
    tid = int(body.get("task_id", 0))
    title = (body.get("title") or "").strip()
    if not tid or not title:
        return {"ok": False, "error": "task_id and title required"}
    ok = store.rename_task_title(tid, title)
    return {"ok": ok}


def api_tasks_delete(store: Any, body: dict) -> dict:
    tid = int(body.get("task_id", 0))
    if not tid:
        return {"ok": False, "error": "task_id required"}
    ok = store.delete_task(tid)
    return {"ok": ok}


def api_tasks_merge(store: Any, body: dict) -> dict:
    ids = body.get("task_ids") or []
    ids = [int(x) for x in ids]
    if len(ids) < 2:
        return {"ok": False, "error": "need >= 2 task_ids"}
    new_id = store.merge_tasks(ids)
    return {"ok": bool(new_id), "new_task_id": new_id}


def api_tasks_teach(store: Any, body: dict) -> dict:
    tid = int(body.get("task_id", 0))
    rule = (body.get("rule") or "").strip()
    action = (body.get("action") or "").strip()
    if not tid or not rule or not action:
        return {"ok": False, "error": "task_id, rule, action required"}
    # We need a rule_vector; lightweight: reuse embed() from skill_hub.
    try:
        from .embeddings import embed
        vec = embed(rule)
    except Exception:  # noqa: BLE001
        vec = [0.0]
    new_id = store.add_teaching(
        rule=rule, rule_vector=vec, action=action,
        target_type="task", target_id=str(tid), weight=1.0,
    )
    return {"ok": True, "teaching_id": new_id}


# ---------------------------------------------------------------------------
# Skills
# ---------------------------------------------------------------------------

def api_skills_usage(store: Any, body: dict | None = None) -> dict:
    return {"skills": store.get_skill_usage_stats()}


def api_skills_for_task(store: Any, body: dict) -> dict:
    tid = int(body.get("task_id", 0))
    return {"skills": store.get_skills_for_task(tid) if tid else []}


# ---------------------------------------------------------------------------
# Vector viz
# ---------------------------------------------------------------------------

def api_vector_viz(store: Any, body: dict | None = None) -> dict:
    body = body or {}
    kind = body.get("kind", "skills")
    conn: sqlite3.Connection = store._conn
    rows: list[dict] = []
    if kind == "skills":
        data = conn.execute(
            "SELECT s.id, s.name, s.plugin, e.vector FROM skills s "
            "JOIN embeddings e ON e.skill_id = s.id"
        ).fetchall()
        for r in data:
            try:
                vec = json.loads(r["vector"])
            except (json.JSONDecodeError, TypeError):
                continue
            rows.append({"id": r["id"], "vector": vec,
                         "label": r["name"], "group": r["plugin"] or ""})
    elif kind == "tasks":
        data = conn.execute(
            "SELECT id, title, status, vector FROM tasks WHERE vector IS NOT NULL"
        ).fetchall()
        for r in data:
            try:
                vec = json.loads(r["vector"])
            except (json.JSONDecodeError, TypeError):
                continue
            rows.append({"id": r["id"], "vector": vec,
                         "label": r["title"], "group": r["status"]})
    elif kind == "teachings":
        data = conn.execute(
            "SELECT id, rule, target_type, rule_vector FROM teachings"
        ).fetchall()
        for r in data:
            try:
                vec = json.loads(r["rule_vector"])
            except (json.JSONDecodeError, TypeError):
                continue
            rows.append({"id": r["id"], "vector": vec,
                         "label": r["rule"], "group": r["target_type"]})
    elif kind == "verdicts":
        vc = _vc_conn()
        if vc:
            data = vc.execute(
                "SELECT cmd_hash, command, source, vector FROM command_verdicts "
                "WHERE vector IS NOT NULL"
            ).fetchall()
            vc.close()
            for r in data:
                try:
                    vec = json.loads(r["vector"])
                except (json.JSONDecodeError, TypeError):
                    continue
                rows.append({"id": r["cmd_hash"], "vector": vec,
                             "label": r["command"][:80], "group": r["source"]})

    projected = vector_viz.project_all(rows)
    return {"kind": kind, "points": projected, "count": len(projected)}


# ---------------------------------------------------------------------------
# Aggregate metrics (initial page load)
# ---------------------------------------------------------------------------

def api_metrics(store: Any, body: dict | None = None) -> dict:
    db = _dashboard._db_metrics(store)
    logm = _dashboard._parse_log()
    vcache = _dashboard._verdict_metrics()
    return {
        "db": _jsonable(db),
        "log": _jsonable({
            "auto_approve": dict(logm["auto_approve"]),
            "auto_approve_tool": dict(logm["auto_approve_tool"].most_common(10)),
            "auto_proceed_fires": logm["auto_proceed_fires"],
            "resume_consumed": logm["resume_consumed"],
            "intercept_errors": logm["intercept_errors"],
            "llm_ms_count": len(logm["llm_ms"]),
            "llm_ms_sum": sum(logm["llm_ms"]),
            "p50": _dashboard._percentile(logm["llm_ms"], 50),
            "p95": _dashboard._percentile(logm["llm_ms"], 95),
            "log_missing": logm["log_missing"],
        }),
        "verdicts": _jsonable(vcache),
        "now": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    return obj


# ---------------------------------------------------------------------------
# Vector classifier preview (used for "test this command" UI)
# ---------------------------------------------------------------------------

def api_vector_classify(store: Any, body: dict) -> dict:
    cmd = (body.get("command") or "").strip()
    if not cmd:
        return {"ok": False, "error": "command required"}
    try:
        from .embeddings import embed
        vec = embed(cmd)
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"embed failed: {e}"}
    conn = _vc_conn()
    if conn is None:
        return {"ok": False, "error": "verdict cache unavailable"}
    cfg = verdict_cache.load_config() if verdict_cache else {}
    thresh = float(cfg.get("vector_autoapprove_threshold", 0.88))
    hit = verdict_cache.search_by_vector(conn, vec, threshold=0.0)  # get best
    conn.close()
    if not hit:
        return {"ok": True, "match": None, "threshold": thresh}
    return {"ok": True, "match": {
        "command": hit["command"],
        "similarity": hit["similarity"],
        "source": hit["source"],
        "would_approve": hit["similarity"] >= thresh,
    }, "threshold": thresh}
