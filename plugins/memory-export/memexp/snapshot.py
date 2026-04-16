"""Build and restore portable mcp-skill-hub memory snapshots.

A snapshot is a single ``.tar.gz`` containing:
- ``manifest.json`` — schema_version + per-component inventory
- ``hub/tables/<name>.jsonl`` — one JSON-encoded row per line
- ``hub/config.json`` — hub configuration (verbatim copy)
- ``hub/settings_enabled_plugins.json`` — slice of ~/.claude/settings.json
- ``projects/<key>/<file>.md`` — per-project memory files (no private/)
- ``local-skills/<file>.json`` — optional, controlled by export option

Imports apply per-target conflict modes (skip / override / llm). The ``llm``
mode delegates to :mod:`intelligent_merge`.
"""
from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from . import scope

_log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

DEFAULT_HUB_DIR = Path.home() / ".claude" / "mcp-skill-hub"
DEFAULT_DB_PATH = DEFAULT_HUB_DIR / "skill_hub.db"
DEFAULT_CONFIG_PATH = DEFAULT_HUB_DIR / "config.json"
DEFAULT_SETTINGS_PATH = Path.home() / ".claude" / "settings.json"
DEFAULT_LOCAL_SKILLS_DIR = Path.home() / ".claude" / "local-skills"
DEFAULT_EXPORT_DIR = DEFAULT_HUB_DIR / "exports"


# ---------------------------------------------------------------------------
# Public option dataclasses

@dataclass
class ExportOptions:
    """Inputs for :func:`build_snapshot`."""

    project_keys: list[str] = field(default_factory=list)
    table_names: list[str] | None = None  # None = all exportable
    include_local_skills: bool = False
    include_config: bool = True
    include_enabled_plugins: bool = True
    db_path: Path = DEFAULT_DB_PATH
    config_path: Path = DEFAULT_CONFIG_PATH
    settings_path: Path = DEFAULT_SETTINGS_PATH
    local_skills_dir: Path = DEFAULT_LOCAL_SKILLS_DIR
    projects_root: Path = scope.DEFAULT_CLAUDE_PROJECTS_ROOT
    output_dir: Path = DEFAULT_EXPORT_DIR


@dataclass
class RestorePlan:
    """Per-target conflict resolution selections.

    ``hub_modes`` maps table-name → ``"skip" | "override" | "llm"``.
    ``project_modes`` maps project-key → ``{"mode": "...", "llm_tier": "..."}``.
    ``llm_per_target`` overrides the tier per hub-table when mode == "llm".
    """

    hub_modes: dict[str, str] = field(default_factory=dict)
    project_modes: dict[str, dict[str, str]] = field(default_factory=dict)
    llm_per_target: dict[str, str] = field(default_factory=dict)
    default_mode: str = "skip"
    default_llm_tier: str = "tier_cheap"
    db_path: Path = DEFAULT_DB_PATH
    projects_root: Path = scope.DEFAULT_CLAUDE_PROJECTS_ROOT
    local_skills_dir: Path = DEFAULT_LOCAL_SKILLS_DIR
    max_llm_calls: int = 200

    def hub_mode(self, table: str) -> str:
        return self.hub_modes.get(table, self.default_mode)

    def hub_tier(self, table: str) -> str:
        return self.llm_per_target.get(table, self.default_llm_tier)

    def project_mode(self, key: str) -> str:
        entry = self.project_modes.get(key, {})
        return entry.get("mode", self.default_mode)

    def project_tier(self, key: str) -> str:
        entry = self.project_modes.get(key, {})
        return entry.get("llm_tier", self.default_llm_tier)


@dataclass
class TableReport:
    inserted: int = 0
    replaced: int = 0
    llm_merged: int = 0
    skipped: int = 0


@dataclass
class FileReport:
    written: int = 0
    overwritten: int = 0
    llm_merged: int = 0
    skipped: int = 0


@dataclass
class RestoreReport:
    tables: dict[str, TableReport] = field(default_factory=dict)
    files: FileReport = field(default_factory=FileReport)
    errors: list[str] = field(default_factory=list)
    llm_calls: int = 0
    llm_total_latency_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "tables": {k: v.__dict__ for k, v in self.tables.items()},
            "files": self.files.__dict__,
            "errors": self.errors,
            "llm_calls": self.llm_calls,
            "llm_total_latency_ms": self.llm_total_latency_ms,
        }


# ---------------------------------------------------------------------------
# Export


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


def _row_to_dict(cursor: sqlite3.Cursor, row: tuple) -> dict:
    cols = [c[0] for c in cursor.description]
    return dict(zip(cols, row))


def _table_pk_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    """Return the primary-key column names for ``table``.

    Uses ``PRAGMA table_info``. Falls back to ``rowid`` semantics by
    returning the empty list when no PK is declared.
    """
    cur = conn.execute(f"PRAGMA table_info({table})")
    return [row[1] for row in cur.fetchall() if row[5]]  # row[5] = pk index


def _dump_table_to_jsonl(conn: sqlite3.Connection, table: str, dest: Path) -> int:
    """Stream all rows of ``table`` as JSON-Lines into ``dest``. Returns count."""
    cur = conn.execute(f"SELECT * FROM {table}")
    n = 0
    with dest.open("w", encoding="utf-8") as f:
        for row in cur:
            obj = _row_to_dict(cur, row)
            # bytes are not JSON-serialisable; encode as hex with a sentinel.
            for k, v in list(obj.items()):
                if isinstance(v, bytes):
                    obj[k] = {"__bytes_hex__": v.hex()}
            f.write(json.dumps(obj, ensure_ascii=False, default=str))
            f.write("\n")
            n += 1
    return n


def build_snapshot(opts: ExportOptions) -> Path:
    """Build a snapshot tar.gz and return its path."""
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = opts.output_dir / f"snapshot-{_timestamp()}.tar.gz"

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        hub_tables = staging / "hub" / "tables"
        hub_tables.mkdir(parents=True)

        manifest: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "host": _host_id(),
            "included_tables": {},
            "included_projects": [],
            "file_count": 0,
            "db_size_bytes": 0,
        }

        # --- Hub DB tables ---
        if opts.db_path.exists():
            manifest["db_size_bytes"] = opts.db_path.stat().st_size
            with sqlite3.connect(opts.db_path) as conn:
                conn.row_factory = None
                table_set = (
                    set(opts.table_names)
                    if opts.table_names is not None
                    else set(scope.list_exportable_tables(conn))
                )
                # Always exclude ephemeral, even if explicitly listed.
                table_set -= scope.EPHEMERAL_TABLES
                for tname in sorted(table_set):
                    dest = hub_tables / f"{tname}.jsonl"
                    n = _dump_table_to_jsonl(conn, tname, dest)
                    manifest["included_tables"][tname] = n

        # --- Hub config ---
        if opts.include_config and opts.config_path.exists():
            shutil.copy2(opts.config_path, staging / "hub" / "config.json")

        # --- enabledPlugins slice ---
        if opts.include_enabled_plugins and opts.settings_path.exists():
            try:
                settings = json.loads(opts.settings_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                settings = {}
            slim = {"enabledPlugins": settings.get("enabledPlugins", {})}
            (staging / "hub" / "settings_enabled_plugins.json").write_text(
                json.dumps(slim, indent=2), encoding="utf-8"
            )

        # --- Projects (memory/*.md, NEVER private/) ---
        all_projects = {p.key: p for p in scope.list_projects(opts.projects_root)}
        for key in opts.project_keys:
            info = all_projects.get(key)
            if info is None:
                _log.warning("project key not found: %s", key)
                continue
            proj_out = staging / "projects" / key
            proj_out.mkdir(parents=True, exist_ok=True)
            files = scope.exportable_md_files(info.path)
            for src in files:
                if src.name == "MEMORY.md":
                    text = src.read_text(encoding="utf-8")
                    (proj_out / "MEMORY.md").write_text(
                        scope.filter_memory_index(text), encoding="utf-8"
                    )
                else:
                    shutil.copy2(src, proj_out / src.name)
            manifest["included_projects"].append(
                {"key": key, "file_count": len(files)}
            )
            manifest["file_count"] += len(files)

        # --- Local skills (optional) ---
        if opts.include_local_skills and opts.local_skills_dir.is_dir():
            ls_out = staging / "local-skills"
            ls_out.mkdir(parents=True, exist_ok=True)
            for src in opts.local_skills_dir.glob("*.json"):
                shutil.copy2(src, ls_out / src.name)
                manifest["file_count"] += 1

        # --- Manifest ---
        (staging / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

        # --- Bundle ---
        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(staging, arcname=".", filter=_strip_owner)

    return out_path


def _strip_owner(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    return info


def _host_id() -> str:
    import socket

    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


# ---------------------------------------------------------------------------
# Restore

# Hook signature for LLM-driven row merge. Provided by callers (so this
# module stays import-light and testable). Returns the merged row dict.
RowMerger = Callable[[str, dict, dict, dict, str], dict]
FileMerger = Callable[[str, str, str, str], str]


def restore_snapshot(
    tar_path: Path,
    plan: RestorePlan,
    *,
    row_merger: RowMerger | None = None,
    file_merger: FileMerger | None = None,
) -> RestoreReport:
    """Apply ``tar_path`` to the live hub state per ``plan``."""
    report = RestoreReport()

    with tempfile.TemporaryDirectory() as td:
        staging = Path(td)
        with tarfile.open(tar_path, "r:gz") as tar:
            _safe_extract(tar, staging, report)

        manifest_path = staging / "manifest.json"
        if not manifest_path.exists():
            report.errors.append("manifest.json missing — not a valid snapshot")
            return report
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            report.errors.append(f"manifest.json invalid: {e}")
            return report
        if int(manifest.get("schema_version", -1)) != SCHEMA_VERSION:
            report.errors.append(
                f"unsupported schema_version: {manifest.get('schema_version')!r} "
                f"(this build expects {SCHEMA_VERSION})"
            )
            return report

        # --- Hub tables ---
        tables_dir = staging / "hub" / "tables"
        if tables_dir.is_dir() and plan.db_path.exists():
            with sqlite3.connect(plan.db_path) as conn:
                conn.row_factory = None
                for jsonl in sorted(tables_dir.glob("*.jsonl")):
                    table = jsonl.stem
                    if table in scope.EPHEMERAL_TABLES:
                        continue
                    mode = plan.hub_mode(table)
                    tier = plan.hub_tier(table)
                    rep = report.tables.setdefault(table, TableReport())
                    _apply_table(
                        conn, table, jsonl, mode, tier,
                        rep, report, plan, row_merger,
                    )
                conn.commit()

        # --- Project files ---
        projects_dir = staging / "projects"
        if projects_dir.is_dir():
            for proj_dir in sorted(projects_dir.iterdir()):
                if not proj_dir.is_dir():
                    continue
                key = proj_dir.name
                mode = plan.project_mode(key)
                tier = plan.project_tier(key)
                target_dir = plan.projects_root / key / "memory"
                _apply_project_files(
                    proj_dir, target_dir, mode, tier, report, plan, file_merger,
                )

        # --- Local skills (always merge — small, idempotent) ---
        ls_dir = staging / "local-skills"
        if ls_dir.is_dir() and plan.local_skills_dir is not None:
            plan.local_skills_dir.mkdir(parents=True, exist_ok=True)
            for src in ls_dir.glob("*.json"):
                dest = plan.local_skills_dir / src.name
                if dest.exists():
                    report.files.skipped += 1
                else:
                    shutil.copy2(src, dest)
                    report.files.written += 1

    return report


def _safe_extract(tar: tarfile.TarFile, dest: Path, report: RestoreReport) -> None:
    """Extract ``tar`` into ``dest`` while rejecting traversal and private/."""
    for member in tar.getmembers():
        # Reject path traversal and absolute paths.
        target = (dest / member.name).resolve()
        if not str(target).startswith(str(dest.resolve()) + "/") and target != dest.resolve():
            report.errors.append(f"refusing path-traversal entry: {member.name}")
            continue
        # Defence in depth: snapshots must never carry private/ paths.
        if any(part == "private" for part in Path(member.name).parts):
            report.errors.append(f"refusing private/ entry: {member.name}")
            continue
        # ``filter="data"`` strips metadata/symlinks per Python 3.12+ guidance.
        tar.extract(member, dest, filter="data")


def _apply_table(
    conn: sqlite3.Connection,
    table: str,
    jsonl_path: Path,
    mode: str,
    tier: str,
    rep: TableReport,
    report: RestoreReport,
    plan: RestorePlan,
    row_merger: RowMerger | None,
) -> None:
    """Insert/replace/llm-merge one JSONL into one table."""
    pk_cols = _table_pk_columns(conn, table)
    cur = conn.cursor()
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as e:
            report.errors.append(f"{table}: bad JSON line ({e})")
            continue
        # Decode hex-encoded blobs back to bytes.
        for k, v in list(row.items()):
            if isinstance(v, dict) and "__bytes_hex__" in v:
                row[k] = bytes.fromhex(v["__bytes_hex__"])

        cols = list(row.keys())
        placeholders = ", ".join("?" for _ in cols)
        col_list = ", ".join(cols)
        values = [row[c] for c in cols]

        if mode == "override":
            cur.execute(
                f"INSERT OR REPLACE INTO {table} ({col_list}) VALUES ({placeholders})",
                values,
            )
            if cur.rowcount > 0:
                rep.replaced += 1
            continue

        if mode == "llm" and pk_cols and row_merger is not None:
            local = _fetch_existing(conn, table, pk_cols, row)
            if local is None:
                cur.execute(
                    f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders})",
                    values,
                )
                rep.inserted += int(bool(cur.rowcount))
                continue
            if local == row:
                rep.skipped += 1
                continue
            if report.llm_calls >= plan.max_llm_calls:
                report.errors.append(
                    f"{table}: max_llm_calls={plan.max_llm_calls} reached; downgrading to skip"
                )
                rep.skipped += 1
                continue
            pk_dict = {c: row.get(c) for c in pk_cols}
            try:
                merged = row_merger(table, pk_dict, local, row, tier)
            except Exception as e:  # noqa: BLE001
                report.errors.append(f"{table}: row_merger raised: {e}")
                rep.skipped += 1
                continue
            report.llm_calls += 1
            mcols = list(merged.keys())
            mvals = [merged[c] for c in mcols]
            cur.execute(
                f"INSERT OR REPLACE INTO {table} ({', '.join(mcols)}) "
                f"VALUES ({', '.join('?' for _ in mcols)})",
                mvals,
            )
            rep.llm_merged += 1
            continue

        # Default / "skip" / llm-without-merger / pk-less table → INSERT OR IGNORE
        cur.execute(
            f"INSERT OR IGNORE INTO {table} ({col_list}) VALUES ({placeholders})",
            values,
        )
        if cur.rowcount > 0:
            rep.inserted += 1
        else:
            rep.skipped += 1


def _fetch_existing(
    conn: sqlite3.Connection,
    table: str,
    pk_cols: list[str],
    row: dict,
) -> dict | None:
    where = " AND ".join(f"{c} = ?" for c in pk_cols)
    vals = [row.get(c) for c in pk_cols]
    cur = conn.execute(f"SELECT * FROM {table} WHERE {where}", vals)
    fetched = cur.fetchone()
    if fetched is None:
        return None
    cols = [c[0] for c in cur.description]
    out = dict(zip(cols, fetched))
    for k, v in list(out.items()):
        if isinstance(v, bytes):
            out[k] = {"__bytes_hex__": v.hex()}
    return out


def _apply_project_files(
    src_dir: Path,
    dest_dir: Path,
    mode: str,
    tier: str,
    report: RestoreReport,
    plan: RestorePlan,
    file_merger: FileMerger | None,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.iterdir()):
        if not src.is_file():
            continue
        # Defensive — should already be filtered by _safe_extract.
        if "private" in src.parts:
            continue
        dest = dest_dir / src.name
        if not dest.exists():
            shutil.copy2(src, dest)
            report.files.written += 1
            continue
        if mode == "override":
            shutil.copy2(src, dest)
            report.files.overwritten += 1
            continue
        if mode == "llm" and file_merger is not None:
            local_text = dest.read_text(encoding="utf-8", errors="ignore")
            incoming_text = src.read_text(encoding="utf-8", errors="ignore")
            if local_text == incoming_text:
                report.files.skipped += 1
                continue
            if report.llm_calls >= plan.max_llm_calls:
                report.errors.append(
                    f"{src.name}: max_llm_calls reached; skipping"
                )
                report.files.skipped += 1
                continue
            try:
                merged = file_merger(src.name, local_text, incoming_text, tier)
            except Exception as e:  # noqa: BLE001
                report.errors.append(f"{src.name}: file_merger raised: {e}")
                report.files.skipped += 1
                continue
            report.llm_calls += 1
            dest.write_text(merged, encoding="utf-8")
            report.files.llm_merged += 1
            continue
        report.files.skipped += 1


# ---------------------------------------------------------------------------
# Preview helper used by the /preview endpoint


def preview(opts: ExportOptions) -> dict:
    """Return a JSON-serialisable preview of what ``build_snapshot`` would write."""
    out: dict[str, Any] = {
        "tables": {},
        "projects": [],
        "file_count": 0,
        "estimated_bytes": 0,
        "pii_offenders": [],
    }
    if opts.db_path.exists():
        with sqlite3.connect(opts.db_path) as conn:
            tables = (
                opts.table_names
                if opts.table_names is not None
                else scope.list_exportable_tables(conn)
            )
            for t in tables:
                if t in scope.EPHEMERAL_TABLES:
                    continue
                cur = conn.execute(f"SELECT COUNT(*) FROM {t}")
                out["tables"][t] = int(cur.fetchone()[0])
        out["estimated_bytes"] += opts.db_path.stat().st_size

    all_projects = {p.key: p for p in scope.list_projects(opts.projects_root)}
    pii_offenders: list[Path] = []
    for key in opts.project_keys:
        info = all_projects.get(key)
        if info is None:
            continue
        files = scope.exportable_md_files(info.path)
        out["projects"].append({"key": key, "file_count": len(files)})
        out["file_count"] += len(files)
        for f in files:
            try:
                out["estimated_bytes"] += f.stat().st_size
            except OSError:
                pass
        pii_offenders.extend(scope.scan_for_pii(files))
    out["pii_offenders"] = [str(p) for p in pii_offenders]
    return out
