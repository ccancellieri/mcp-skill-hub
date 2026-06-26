"""Dashboard KPI card for the cross-project-knowledge-bridge plugin.

Returns one section showing project count, entity count, and last sync time.
Renders via the shared ``kpi_card`` macro.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))


def _load_config() -> dict:
    mf = _PLUGIN_ROOT / "plugin.json"
    if not mf.exists():
        return {}
    try:
        return json.loads(mf.read_text(encoding="utf-8")).get("config", {})
    except (json.JSONDecodeError, OSError):
        return {}


def _resolve_projects() -> list[dict]:
    config = _load_config()
    projects = config.get("projects", [])
    result = []
    for p in projects:
        path = Path(str(p.get("path", ""))).expanduser()
        tags = p.get("tags", [])
        result.append({
            "name": p.get("name", "unknown"),
            "path": str(path),
            "tags": tags,
            "exists": path.exists(),
        })
    return result


def _count_entities() -> int:
    projects = _resolve_projects()
    count = 0
    for proj in projects:
        if not proj["exists"]:
            continue
        memory_dir = Path(proj["path"]) / ".memory"
        if memory_dir.exists():
            for md_file in memory_dir.glob("*.md"):
                if not md_file.name.startswith("_"):
                    count += 1
    return count


def _last_sync_time() -> str | None:
    sync_marker = _PLUGIN_ROOT / ".sync_marker"
    if sync_marker.exists():
        try:
            ts = float(sync_marker.read_text().strip())
            return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
        except (ValueError, OSError):
            pass
    return None


def get_sections() -> list[dict]:
    projects = _resolve_projects()
    entity_count = _count_entities()
    last_sync = _last_sync_time()

    return [
        {
            "id": "knowledge-bridge",
            "title": "Knowledge Bridge",
            "order": 70,
            "template": "dashboard_kpi.html",
            "context": {
                "project_count": len(projects),
                "entity_count": entity_count,
                "last_sync_at": last_sync,
            },
        }
    ]
