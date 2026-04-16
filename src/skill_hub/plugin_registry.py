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

SETTINGS_PATH = Path.home() / ".claude" / "settings.json"

# Repo-root-relative folder for first-party bundled plugins. Resolved from this
# file's location: src/skill_hub/plugin_registry.py -> ../../../plugins
BUNDLED_PLUGINS_DIR = Path(__file__).resolve().parents[2] / "plugins"


def _iter_bundled_sources() -> Iterator[dict[str, Any]]:
    """Synthetic ``extra_plugin_dirs`` entries for ``<repo>/plugins/*``.

    Each subdirectory containing a ``plugin.json`` becomes an entry. Honours
    the ``bundled_plugins_enabled`` config flag (default True). Bundled
    plugins still respect the per-plugin enabled bit in
    ``~/.claude/settings.json["enabledPlugins"]`` via ``iter_all_plugins``.
    """
    if not bool(_cfg.get("bundled_plugins_enabled")):
        return
    if not BUNDLED_PLUGINS_DIR.exists():
        return
    for d in sorted(BUNDLED_PLUGINS_DIR.iterdir()):
        if not d.is_dir():
            continue
        if not (d / "plugin.json").exists():
            continue
        yield {"path": str(d), "source": "bundled", "enabled": True}


def _load_settings() -> dict:
    if not SETTINGS_PATH.exists():
        return {}
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def _save_settings(settings: dict) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(settings, indent=2))


def _enabled_map() -> dict[str, bool]:
    return dict(_load_settings().get("enabledPlugins") or {})


def _resolve_plugin_key(plugin_name: str, plugins: dict) -> str | None:
    """Map a short name like 'superpowers' to the full 'superpowers@source' key."""
    if plugin_name in plugins:
        return plugin_name
    matches = [k for k in plugins if k.split("@", 1)[0] == plugin_name]
    if len(matches) == 1:
        return matches[0]
    return None


def iter_enabled_plugins() -> Iterator[dict[str, Any]]:
    """Yield {name, path, manifest, source} for each enabled plugin directory.

    ``path`` is the absolute ``Path`` to the plugin's root (the directory that
    contains ``plugin.json``). ``manifest`` is the parsed JSON (empty dict if
    the file is missing/invalid). ``name`` falls back to the directory name.
    """
    cfg = _cfg.load_config() if hasattr(_cfg, "load_config") else {}
    entries = list(_iter_bundled_sources()) + list(cfg.get("extra_plugin_dirs") or [])
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


def iter_all_plugins() -> Iterator[dict[str, Any]]:
    """Yield every known plugin (enabled + disabled) with its `enabled: bool` flag.

    Used by the /control plugins tab — the enabled filter in iter_enabled_plugins
    is based on the *source* directory flag; here we also overlay the per-plugin
    enabled state from ``~/.claude/settings.json["enabledPlugins"]``.
    """
    cfg = _cfg.load_config() if hasattr(_cfg, "load_config") else {}
    enabled_map = _enabled_map()
    # Reuse iter_enabled_plugins' parsing by briefly bypassing the source flag.
    sources = list(_iter_bundled_sources()) + list(cfg.get("extra_plugin_dirs") or [])
    for entry in sources:
        base = Path(str(entry.get("path", ""))).expanduser()
        if not base.exists():
            continue
        source = entry.get("source", "extra")
        source_enabled = bool(entry.get("enabled", True))
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
                except (json.JSONDecodeError, OSError):
                    manifest = {}
            plugin_name = manifest.get("name") or plugin_dir.name
            full_key = next(
                (k for k in enabled_map if k.split("@", 1)[0] == plugin_name),
                plugin_name,
            )
            description = (
                manifest.get("description")
                or manifest.get("summary")
                or ""
            )
            yield {
                "name": plugin_name,
                "full_key": full_key,
                "path": plugin_dir,
                "manifest": manifest,
                "description": description,
                "source": source,
                "source_enabled": source_enabled,
                "enabled": bool(enabled_map.get(full_key, False)) and source_enabled,
            }


def toggle(plugin_name: str, enabled: bool) -> str:
    """Flip a plugin's enabled bit in ~/.claude/settings.json.

    Accepts either the short name (``"superpowers"``) or the fully-qualified
    key (``"superpowers@claude-plugins-official"``). Matches the semantics of
    the MCP ``toggle_plugin`` tool so both call paths agree.
    """
    settings = _load_settings()
    if not settings:
        return f"Settings file not found: {SETTINGS_PATH}"
    plugins: dict = settings.setdefault("enabledPlugins", {})
    key = _resolve_plugin_key(plugin_name, plugins)
    if key is None:
        if enabled:
            key = f"{plugin_name}@claude-plugins-official"
            plugins[key] = True
            _save_settings(settings)
            return f"Added '{key}' as enabled (restart to apply)."
        return f"Plugin '{plugin_name}' not found in settings."
    plugins[key] = enabled
    _save_settings(settings)
    state = "enabled" if enabled else "disabled"
    return f"Plugin '{key}' {state}. Restart Claude Code to apply."


def apply_profile(profile_name: str) -> str:
    """Enable every plugin in the named profile and disable the rest."""
    cfg = _cfg.load_config() if hasattr(_cfg, "load_config") else {}
    profiles = (cfg.get("profiles") or {})
    profile = profiles.get(profile_name)
    if profile is None:
        return f"Profile '{profile_name}' not found. Known: {', '.join(profiles) or '(none)'}"

    target = profile.get("plugins")
    if target == "__all__":
        target_names = None  # enable everything known
    elif isinstance(target, list):
        target_names = set(target)
    else:
        return f"Profile '{profile_name}' has malformed plugins field."

    settings = _load_settings()
    plugins: dict = settings.setdefault("enabledPlugins", {})
    # Iterate over every key we know about (settings + discovered).
    known_keys = set(plugins.keys())
    for p in iter_all_plugins():
        known_keys.add(p["full_key"])
    for key in known_keys:
        short = key.split("@", 1)[0]
        should_enable = target_names is None or short in target_names
        plugins[key] = should_enable
    _save_settings(settings)
    n_on = sum(1 for v in plugins.values() if v)
    return f"Profile '{profile_name}' applied — {n_on} plugins enabled. Restart Claude Code to take effect."


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
