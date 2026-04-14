"""Plugin discovery helpers — used by extension-point integrations.

Reads ``extra_plugin_dirs`` from ``~/.claude/mcp-skill-hub/config.json`` and
yields per-plugin manifest records merged with their ``plugin.json`` contents.

This is the single source of truth for the A1, A2, A7, A8, A9, A11 integration
points. Keeps the discovery rules (enabled flag, path resolution, plugin.json
parsing) in one place so webapp/watcher/dashboard all agree on what "an
enabled plugin" is.

See ``docs/plugin-extension-points.md`` for the canonical plugin.json shape.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from . import config as _cfg

_log = logging.getLogger(__name__)


def iter_enabled_plugins() -> Iterator[dict[str, Any]]:
    """Yield {name, path, manifest, source} for each enabled plugin directory.

    ``path`` is the absolute ``Path`` to the plugin's root (the directory that
    contains ``plugin.json``). ``manifest`` is the parsed JSON (empty dict if
    the file is missing/invalid). ``name`` falls back to the directory name.
    """
    cfg = _cfg.load_config() if hasattr(_cfg, "load_config") else {}
    entries = cfg.get("extra_plugin_dirs") or []
    for entry in entries:
        if not entry.get("enabled", True):
            continue
        base = Path(str(entry.get("path", ""))).expanduser()
        if not base.exists():
            continue
        source = entry.get("source", "extra")
        # Each subdirectory with a plugin.json is a plugin; if the base itself
        # has a plugin.json, treat the base itself as the plugin root.
        candidates: list[Path] = []
        if (base / "plugin.json").exists():
            candidates.append(base)
        else:
            candidates.extend(d for d in base.iterdir() if d.is_dir())
        for plugin_dir in candidates:
            manifest: dict[str, Any] = {}
            mf = plugin_dir / "plugin.json"
            if mf.exists():
                try:
                    manifest = json.loads(mf.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError) as exc:
                    _log.warning("plugin manifest unreadable: %s (%s)", mf, exc)
                    manifest = {}
            yield {
                "name": manifest.get("name") or plugin_dir.name,
                "path": plugin_dir,
                "manifest": manifest,
                "source": source,
            }


def load_web_mounts() -> list[dict[str, Any]]:
    """Collect web mount definitions from enabled plugins + manual overrides.

    Precedence: per-plugin ``web_mount`` in plugin.json; an entry in the top
    level config key ``extra_web_mounts`` is treated as a manual override and
    merged (last write wins on ``mount`` key).
    """
    mounts: dict[str, dict[str, Any]] = {}
    for p in iter_enabled_plugins():
        wm = p["manifest"].get("web_mount")
        if not wm:
            continue
        mount = wm.get("mount")
        if not mount:
            continue
        mounts[mount] = {
            "plugin_name": p["name"],
            "plugin_path": str(p["path"]),
            "mount": mount,
            "title": wm.get("title") or p["name"],
            "icon": wm.get("icon"),
            "nav": bool(wm.get("nav", True)),
        }

    # Manual overrides from config.extra_web_mounts
    cfg = _cfg.load_config() if hasattr(_cfg, "load_config") else {}
    for entry in cfg.get("extra_web_mounts") or []:
        mount = entry.get("mount")
        if not mount:
            continue
        mounts[mount] = {
            "plugin_name": Path(str(entry.get("plugin_path", ""))).name,
            "plugin_path": str(Path(str(entry.get("plugin_path", ""))).expanduser()),
            "mount": mount,
            "title": entry.get("title") or mount.strip("/"),
            "icon": entry.get("icon"),
            "nav": bool(entry.get("nav", True)),
        }
    return list(mounts.values())


def register_plugin_vector_indexes(store: Any) -> int:
    """Seed ``vector_index_config`` rows for indexes declared by enabled plugins.

    Phase M2 — plugin.json may declare::

        "vector_indexes": [
          {"name": "career:profile",   "default_level": "L3", "half_life_days": 365},
          {"name": "career:narrative", "default_level": "L2", "half_life_days": 60}
        ]

    Also auto-creates a ``memory:<plugin>`` index row (L2, 30d half-life) for
    any plugin that declares ``memory.reads``.

    Returns the number of index rows inserted/updated. Safe to call repeatedly.
    """
    n = 0
    conn = getattr(store, "_conn", None)
    if conn is None:
        return 0
    for plugin in iter_enabled_plugins():
        manifest = plugin["manifest"]
        rows: list[dict[str, Any]] = list(manifest.get("vector_indexes") or [])
        # Implicit per-plugin memory index for backward-compat A4 reads.
        if manifest.get("memory", {}).get("reads"):
            rows.append({
                "name": f"memory:{plugin['name']}",
                "default_level": "L2",
                "half_life_days": 90.0,
            })
        for idx in rows:
            name = idx.get("name")
            if not name:
                continue
            conn.execute(
                """
                INSERT INTO vector_index_config
                    (name, default_level, half_life_days, chunk_size, chunk_overlap,
                     max_docs, summarizer_prompt, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
                ON CONFLICT(name) DO UPDATE SET
                    default_level     = excluded.default_level,
                    half_life_days    = excluded.half_life_days,
                    chunk_size        = excluded.chunk_size,
                    chunk_overlap     = excluded.chunk_overlap,
                    max_docs          = excluded.max_docs,
                    summarizer_prompt = COALESCE(excluded.summarizer_prompt,
                                                 vector_index_config.summarizer_prompt),
                    updated_at        = datetime('now')
                """,
                (
                    name,
                    idx.get("default_level", "L2"),
                    float(idx.get("half_life_days", 30.0)),
                    int(idx.get("chunk_size", 0)),
                    int(idx.get("chunk_overlap", 0)),
                    int(idx.get("max_docs", 0)),
                    idx.get("summarizer_prompt"),
                ),
            )
            n += 1
    conn.commit()
    return n
