"""Dashboard KPI card for the memory-export plugin.

Returns one section showing last-export timestamp, detected project count,
and exportable hub table count. Renders via the shared ``kpi_card`` macro.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

# Make ``memexp`` importable when this module is loaded standalone by the
# dashboard renderer (which doesn't go through web/app.py's sys.path setup).
_PLUGIN_ROOT = Path(__file__).resolve().parent.parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from memexp import scope, snapshot  # noqa: E402


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _last_export() -> tuple[str | None, str | None]:
    out_dir = snapshot.DEFAULT_EXPORT_DIR
    if not out_dir.exists():
        return None, None
    snaps = sorted(out_dir.glob("snapshot-*.tar.gz"))
    if not snaps:
        return None, None
    latest = snaps[-1]
    ts = datetime.fromtimestamp(latest.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
    return ts, _human_bytes(latest.stat().st_size)


def get_sections() -> list[dict]:
    last_ts, last_size = _last_export()
    project_count = len(scope.list_projects())
    table_count = 0
    if snapshot.DEFAULT_DB_PATH.exists():
        try:
            with sqlite3.connect(snapshot.DEFAULT_DB_PATH) as conn:
                table_count = len(scope.list_exportable_tables(conn))
        except sqlite3.Error:
            table_count = 0
    return [
        {
            "id": "memory-export",
            "title": "Memory Export",
            "order": 80,
            "template": "dashboard_kpi.html",
            "context": {
                "last_export_at": last_ts,
                "last_export_size": last_size,
                "project_count": project_count,
                "table_count": table_count,
            },
        }
    ]
