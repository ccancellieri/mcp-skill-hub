"""REST endpoints: preview / download snapshot / import / history."""
from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from memexp import intelligent_merge, scope, snapshot

router = APIRouter()


def _parse_csv_param(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


def _record_history(
    kind: str,
    path: Optional[str] = None,
    tables_json: Optional[str] = None,
    row_count: Optional[int] = None,
    size_bytes: Optional[int] = None,
    conflict_mode: Optional[str] = None,
    status: str = "completed",
    notes: Optional[str] = None,
) -> None:
    """Write one row to export_history in skill_hub.db (best-effort, never raises)."""
    db_path = snapshot.DEFAULT_DB_PATH
    if not db_path.exists():
        return
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT INTO export_history
                    (kind, path, tables_json, row_count, size_bytes,
                     conflict_mode, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (kind, path, tables_json, row_count, size_bytes,
                 conflict_mode, status, notes),
            )
            conn.commit()
    except Exception:  # noqa: BLE001
        pass


@router.get("/preview")
def preview(
    projects: Optional[str] = Query(None, description="comma-separated project keys"),
    tables: Optional[str] = Query(None, description="comma-separated table names"),
    include_local_skills: bool = Query(False),
):
    project_keys = _parse_csv_param(projects)
    table_names = _parse_csv_param(tables) or None
    opts = snapshot.ExportOptions(
        project_keys=project_keys,
        table_names=table_names,
        include_local_skills=include_local_skills,
    )
    return JSONResponse(snapshot.preview(opts))


@router.get("/api/export/preview")
def export_preview_summary():
    """Return per-table row counts and size estimates for the full snapshot scope."""
    db_path = snapshot.DEFAULT_DB_PATH
    tables_info: list[dict] = []
    total_rows = 0
    total_size = 0
    if db_path.exists():
        with sqlite3.connect(db_path) as conn:
            all_tables = scope.list_exportable_tables(conn)
            for t in all_tables:
                try:
                    cur = conn.execute(f"SELECT COUNT(*) FROM {t}")
                    cnt = int(cur.fetchone()[0])
                except Exception:  # noqa: BLE001
                    cnt = 0
                tables_info.append({"name": t, "row_count": cnt})
                total_rows += cnt
        total_size = db_path.stat().st_size
    return JSONResponse({
        "tables": tables_info,
        "total_rows": total_rows,
        "total_size_bytes": total_size,
        "db_exists": db_path.exists(),
    })


@router.get("/api/export/history")
def export_history(limit: int = Query(50, ge=1, le=500)):
    """Return recent export/import operations from export_history, most recent first."""
    db_path = snapshot.DEFAULT_DB_PATH
    if not db_path.exists():
        return JSONResponse({"records": []})
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT id, kind, path, tables_json, row_count, size_bytes,
                       conflict_mode, status, notes, created_at
                FROM export_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return JSONResponse({"records": [dict(r) for r in rows]})
    except sqlite3.OperationalError:
        # Table doesn't exist yet (pre-migration DB)
        return JSONResponse({"records": []})


@router.get("/snapshot.tar.gz")
def download(
    projects: Optional[str] = Query(None),
    tables: Optional[str] = Query(None),
    include_local_skills: bool = Query(False),
):
    project_keys = _parse_csv_param(projects)
    table_names = _parse_csv_param(tables) or None
    opts = snapshot.ExportOptions(
        project_keys=project_keys,
        table_names=table_names,
        include_local_skills=include_local_skills,
    )
    out = snapshot.build_snapshot(opts)
    _record_history(
        kind="export",
        path=str(out),
        tables_json=json.dumps(table_names or []),
        size_bytes=out.stat().st_size if out.exists() else None,
        conflict_mode=None,
        status="completed",
    )
    return FileResponse(
        path=out,
        media_type="application/gzip",
        filename=out.name,
    )


@router.post("/import")
async def import_snapshot(
    file: UploadFile = File(..., description="snapshot tar.gz"),
    plan: str = Form(
        "{}",
        description=(
            "JSON-encoded RestorePlan: "
            "{hub_modes:{table:mode},project_modes:{key:{mode,llm_tier}},"
            "llm_per_target:{table:tier},default_mode,default_llm_tier,max_llm_calls}"
        ),
    ),
):
    try:
        plan_dict = json.loads(plan)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"plan field is not valid JSON: {e}") from e

    restore_plan = snapshot.RestorePlan(
        hub_modes=dict(plan_dict.get("hub_modes") or {}),
        project_modes=dict(plan_dict.get("project_modes") or {}),
        llm_per_target=dict(plan_dict.get("llm_per_target") or {}),
        default_mode=str(plan_dict.get("default_mode") or "skip"),
        default_llm_tier=str(plan_dict.get("default_llm_tier") or "tier_cheap"),
        max_llm_calls=int(plan_dict.get("max_llm_calls") or 200),
    )

    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        try:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = Path(tmp.name)
        finally:
            file.file.close()

    try:
        report = snapshot.restore_snapshot(
            tmp_path,
            restore_plan,
            row_merger=intelligent_merge.merge_row,
            file_merger=intelligent_merge.merge_markdown,
        )
        report_dict = report.to_dict()
        total_rows = sum(
            v.get("inserted", 0) + v.get("replaced", 0) + v.get("llm_merged", 0)
            for v in report_dict.get("tables", {}).values()
        )
        _record_history(
            kind="import",
            path=file.filename,
            tables_json=json.dumps(list(report_dict.get("tables", {}).keys())),
            row_count=total_rows,
            conflict_mode=restore_plan.default_mode,
            status="completed" if not report_dict.get("errors") else "partial",
            notes=(
                f"{len(report_dict['errors'])} error(s)"
                if report_dict.get("errors") else None
            ),
        )
        return JSONResponse(report_dict)
    except Exception as exc:
        _record_history(
            kind="import",
            path=file.filename,
            conflict_mode=restore_plan.default_mode,
            status="error",
            notes=str(exc),
        )
        raise
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
