"""Experiments panel routes — pipeline preset CRUD + A/B runner."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

router = APIRouter()


class PresetCreate(BaseModel):
    name: str
    description: str = ""
    config: dict = {}


class ExperimentCreate(BaseModel):
    name: str
    preset_a_id: int
    preset_b_id: int
    target_runs: int = 10
    notes: str = ""


@router.get("/experiments", response_class=HTMLResponse)
def experiments_page(request: Request):
    return request.app.state.templates.TemplateResponse(
        "experiments.html", {"request": request}
    )


@router.get("/api/experiments/presets")
def list_presets():
    from skill_hub.store import SkillStore
    store = SkillStore()
    try:
        return store.list_presets()
    finally:
        store.close()


@router.post("/api/experiments/presets")
def create_preset(body: PresetCreate):
    from skill_hub.store import SkillStore
    store = SkillStore()
    try:
        pid = store.save_preset(body.name, body.description, body.config)
        return {"id": pid, "name": body.name}
    finally:
        store.close()


@router.delete("/api/experiments/presets/{preset_id}")
def delete_preset(preset_id: int):
    from skill_hub.store import SkillStore
    store = SkillStore()
    try:
        ok = store.delete_preset(preset_id)
        if not ok:
            raise HTTPException(status_code=404, detail="preset not found or is builtin")
        return {"deleted": True}
    finally:
        store.close()


@router.get("/api/experiments")
def list_experiments():
    from skill_hub.store import SkillStore
    store = SkillStore()
    try:
        exps = store.list_experiments()
        result = []
        for exp in exps:
            stats = store.get_experiment_stats(exp["id"])
            result.append({**exp, "stats": stats})
        return result
    finally:
        store.close()


@router.post("/api/experiments")
def create_experiment(body: ExperimentCreate):
    from skill_hub.store import SkillStore
    store = SkillStore()
    try:
        eid = store.create_experiment(
            body.name, body.preset_a_id, body.preset_b_id, body.target_runs, body.notes
        )
        return {"id": eid, "name": body.name}
    finally:
        store.close()


@router.get("/api/experiments/{experiment_id}/stats")
def experiment_stats(experiment_id: int):
    from skill_hub.store import SkillStore
    store = SkillStore()
    try:
        stats = store.get_experiment_stats(experiment_id)
        return stats
    finally:
        store.close()
