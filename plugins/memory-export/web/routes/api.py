"""REST endpoints: preview / download snapshot / import."""
from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from memexp import intelligent_merge, snapshot

router = APIRouter()


def _parse_csv_param(raw: Optional[str]) -> list[str]:
    if not raw:
        return []
    return [x.strip() for x in raw.split(",") if x.strip()]


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
        return JSONResponse(report.to_dict())
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass
