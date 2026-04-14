"""S3 F-SELECT — profile-based plugin curation.

A *profile* is a named ``enabledPlugins`` set (e.g. ``geoid``, ``data-eng``,
``minimal``). Switching a profile writes the profile's plugin map to
``~/.claude/settings.json``; the user restarts Claude Code to pick it up.
We cannot gate plugins at runtime because the Claude Code harness bakes
``enabledPlugins`` into the system prompt at session start.

Storage: ``profiles`` table in the skill-hub DB. ``is_active`` flags which
profile was last switched-into (one row at a time).

Derived data: ``auto_curate_candidates`` computes stale plugins from
``session_log`` — plugins enabled but never used in the last N days.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any

from . import plugin_registry as _plugins


def _row_to_profile(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "name": row["name"],
        "plugins": json.loads(row["plugins_json"]),
        "description": row["description"] or "",
        "is_active": bool(row["is_active"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_profiles(store: Any) -> list[dict[str, Any]]:
    """Return all stored profiles ordered by name."""
    rows = store._conn.execute(
        "SELECT * FROM profiles ORDER BY is_active DESC, name ASC"
    ).fetchall()
    return [_row_to_profile(r) for r in rows]


def get_profile(store: Any, name: str) -> dict[str, Any] | None:
    row = store._conn.execute(
        "SELECT * FROM profiles WHERE name = ?", (name,)
    ).fetchone()
    return _row_to_profile(row) if row else None


def get_active_profile(store: Any) -> dict[str, Any] | None:
    row = store._conn.execute(
        "SELECT * FROM profiles WHERE is_active = 1 LIMIT 1"
    ).fetchone()
    return _row_to_profile(row) if row else None


def create_profile(
    store: Any,
    name: str,
    plugins: dict[str, bool] | list[str],
    description: str = "",
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Upsert a profile. ``plugins`` may be a dict of ``{id: bool}`` (mirrors
    settings.json) or a list of plugin-ids (implicitly all enabled).
    """
    if isinstance(plugins, list):
        plugin_map = {pid: True for pid in plugins}
    else:
        plugin_map = {str(k): bool(v) for k, v in plugins.items()}
    existing = get_profile(store, name)
    if existing and not overwrite:
        raise ValueError(f"profile {name!r} already exists (pass overwrite=True)")
    store._conn.execute(
        """
        INSERT INTO profiles (name, plugins_json, description, is_active, updated_at)
        VALUES (?, ?, ?, 0, datetime('now'))
        ON CONFLICT(name) DO UPDATE SET
            plugins_json = excluded.plugins_json,
            description  = excluded.description,
            updated_at   = datetime('now')
        """,
        (name, json.dumps(plugin_map), description),
    )
    store._conn.commit()
    return get_profile(store, name)  # type: ignore[return-value]


def delete_profile(store: Any, name: str) -> bool:
    cur = store._conn.execute("DELETE FROM profiles WHERE name = ?", (name,))
    store._conn.commit()
    return cur.rowcount > 0


def switch_profile(store: Any, name: str, *, dry_run: bool = False) -> dict[str, Any]:
    """Activate ``name`` and mirror its plugin map into ``~/.claude/settings.json``.

    Returns ``{profile, changed_plugins, needs_restart}``.
    """
    profile = get_profile(store, name)
    if profile is None:
        raise KeyError(f"profile {name!r} not found")

    # Compare to current settings.
    settings = _plugins._load_settings()
    current: dict[str, bool] = dict(settings.get("enabledPlugins") or {})
    target = dict(profile["plugins"])

    changed = {}
    for pid, val in target.items():
        if current.get(pid) != val:
            changed[pid] = {"before": current.get(pid), "after": val}

    if not dry_run:
        settings["enabledPlugins"] = target
        _plugins._save_settings(settings)
        store._conn.execute("UPDATE profiles SET is_active = 0")
        store._conn.execute(
            "UPDATE profiles SET is_active = 1, updated_at = datetime('now')"
            " WHERE name = ?",
            (name,),
        )
        store._conn.commit()

    return {
        "profile": profile["name"],
        "changed_plugins": changed,
        "needs_restart": bool(changed),
        "dry_run": dry_run,
    }


def detect_profile_drift(store: Any) -> dict[str, Any] | None:
    """Compare active profile's plugin set to live ``~/.claude/settings.json``.

    Returns ``None`` if no active profile or no drift; otherwise returns
    ``{profile, missing, unexpected}`` where ``missing`` are plugins the
    profile wants but are disabled/absent, and ``unexpected`` are enabled
    plugins not in the profile.
    """
    active = get_active_profile(store)
    if active is None:
        return None
    target = active["plugins"]
    current = dict(_plugins._load_settings().get("enabledPlugins") or {})

    missing = {k: v for k, v in target.items() if current.get(k) != v}
    unexpected = {k: v for k, v in current.items()
                  if v and k not in target}
    if not missing and not unexpected:
        return None
    return {
        "profile": active["name"],
        "missing": missing,
        "unexpected": unexpected,
    }


def auto_curate_candidates(store: Any, stale_days: int = 14) -> list[dict[str, Any]]:
    """Return plugins currently enabled in settings.json that have no
    ``session_log`` activity in the last ``stale_days``. Each entry:
    ``{plugin_id, last_used_at, sessions_last_window}``.
    """
    current = {k: v for k, v in (_plugins._load_settings().get("enabledPlugins") or {}).items() if v}
    if not current:
        return []

    rows = store._conn.execute(
        """
        SELECT plugin_id,
               MAX(created_at) AS last_used_at,
               COUNT(DISTINCT session_id) AS sessions
        FROM session_log
        WHERE plugin_id IS NOT NULL
          AND created_at >= datetime('now', ?)
        GROUP BY plugin_id
        """,
        (f"-{int(stale_days)} days",),
    ).fetchall()
    recent_pids = {row["plugin_id"]: dict(row) for row in rows}

    candidates: list[dict[str, Any]] = []
    for pid in current:
        hit = _match_recent(pid, recent_pids)
        if hit is None:
            candidates.append({
                "plugin_id": pid,
                "last_used_at": None,
                "sessions_last_window": 0,
                "reason": f"no activity in last {stale_days} days",
            })
    candidates.sort(key=lambda c: c["plugin_id"])
    return candidates


def _match_recent(plugin_id: str, recent: dict[str, dict]) -> dict | None:
    """Plugin IDs in settings.json look like ``name@source``; session_log may
    store either the full id or a short name. Match both."""
    if plugin_id in recent:
        return recent[plugin_id]
    short = plugin_id.split("@", 1)[0]
    for k, v in recent.items():
        if k == short or k.split("@", 1)[0] == short:
            return v
    return None
