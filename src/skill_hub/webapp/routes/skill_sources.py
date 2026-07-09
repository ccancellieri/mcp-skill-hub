"""Skill import source management and read-only audit page."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from ... import config as _cfg
from ...indexer import index_all
from ...skill_import_audit import (
    LOCAL_JSON,
    SKILL_MD,
    audit_paths,
    default_source_paths,
    repair_importable_skills,
)

router = APIRouter()


def _configured_sources() -> list[dict[str, Any]]:
    sources = _cfg.get("skill_import_sources") or []
    if not isinstance(sources, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in sources:
        if not isinstance(entry, dict):
            continue
        out.append({
            "path": str(entry.get("path") or ""),
            "source": str(entry.get("source") or ""),
            "enabled": bool(entry.get("enabled", True)),
        })
    return out


def _render_page(
    request: Request,
    *,
    message: str = "",
    problems: list[str] | None = None,
    audit: Any | None = None,
    index_errors: list[str] | None = None,
) -> Any:
    sources = _configured_sources()
    rows = sources + [{"path": "", "source": "", "enabled": True}]
    live_sources = _live_skill_dirs()
    return request.app.state.templates.TemplateResponse(
        request,
        "skill_sources.html",
        {
            "active_tab": "skill_sources",
            "sources": rows,
            "configured_count": len(sources),
            "live_sources": live_sources,
            "live_count": len(live_sources),
            "message": message,
            "problems": problems or [],
            "audit": audit,
            "index_errors": index_errors or [],
        },
    )


def _enabled_source_entries(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    sources = cfg.get("skill_import_sources") or []
    if not isinstance(sources, list):
        return []
    entries: list[dict[str, Any]] = []
    for entry in sources:
        if not isinstance(entry, dict) or not entry.get("enabled", True):
            continue
        raw_path = str(entry.get("path") or "").strip()
        if not raw_path:
            continue
        source = str(entry.get("source") or Path(raw_path).name).strip()
        entries.append({"path": raw_path, "source": source, "enabled": True})
    return entries


def _path_key(raw_path: str) -> str:
    return str(Path(raw_path).expanduser())


def _source_paths(sources: list[dict[str, Any]]) -> set[str]:
    return {_path_key(str(src.get("path") or "")) for src in sources if src.get("path")}


def _json_source_paths(report: Any) -> set[str]:
    paths: set[str] = set()
    by_root: dict[str, set[str]] = {}
    for candidate in getattr(report, "candidates", []) or []:
        by_root.setdefault(candidate.source_root, set()).add(candidate.format)
    for root, formats in by_root.items():
        if LOCAL_JSON in formats and SKILL_MD not in formats:
            paths.add(_path_key(root))
    return paths


def _enabled_json_source_paths(cfg: dict[str, Any], report: Any) -> list[str]:
    json_paths = _json_source_paths(report)
    return [
        entry["path"] for entry in _enabled_source_entries(cfg)
        if _path_key(entry["path"]) in json_paths
    ]


def _validate_sources(sources: list[dict[str, Any]]) -> list[str]:
    problems: list[str] = []
    labels: dict[str, str] = {}
    paths: set[str] = set()
    for source in sources:
        raw_path = str(source.get("path") or "").strip()
        label = str(source.get("source") or "").strip()
        path_key = _path_key(raw_path)
        label_key = label.lower()
        if ":" in label:
            problems.append(f"Source label cannot contain ':': {label}")
        if path_key in paths:
            problems.append(f"Duplicate source path: {raw_path}")
        paths.add(path_key)
        if label_key in labels:
            problems.append(f"Duplicate source label: {label}")
        labels[label_key] = raw_path
    return problems


def _live_skill_dirs() -> list[dict[str, Any]]:
    live = _cfg.get("extra_skill_dirs") or []
    if not isinstance(live, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in live:
        if not isinstance(entry, dict):
            continue
        raw_path = str(entry.get("path") or "").strip()
        if not raw_path:
            continue
        out.append({
            "path": raw_path,
            "source": str(entry.get("source") or Path(raw_path).name),
            "enabled": bool(entry.get("enabled", True)),
        })
    return out


def _merge_live_skill_dirs(cfg: dict[str, Any], report: Any) -> tuple[int, list[str]]:
    live = cfg.get("extra_skill_dirs") or []
    if not isinstance(live, list):
        live = []

    current_sources = cfg.get("skill_import_sources") or []
    if not isinstance(current_sources, list):
        current_sources = []
    managed = set(str(path) for path in (cfg.get("skill_import_managed_paths") or []))
    managed_json = str(cfg.get("skill_import_managed_local_json_dir") or "")
    current_paths = _source_paths([entry for entry in current_sources if isinstance(entry, dict)])
    json_paths = _json_source_paths(report)
    enabled_json_paths = _enabled_json_source_paths(cfg, report)
    managed.update(current_paths)

    merged: list[dict[str, Any]] = []
    by_path: dict[str, int] = {}
    for entry in live:
        if not isinstance(entry, dict):
            continue
        raw_path = str(entry.get("path") or "").strip()
        if not raw_path:
            continue
        path_key = _path_key(raw_path)
        if path_key in managed:
            continue
        by_path[raw_path] = len(merged)
        merged.append({
            "path": raw_path,
            "source": str(entry.get("source") or Path(raw_path).name),
            "enabled": bool(entry.get("enabled", True)),
        })

    imported = 0
    notes: list[str] = []
    if managed_json and not enabled_json_paths and _path_key(str(cfg.get("local_skills_dir") or "")) == _path_key(managed_json):
        cfg["local_skills_dir"] = ""
        cfg["skill_import_managed_local_json_dir"] = None
        notes.append(f"local JSON skills directory removed: {managed_json}")

    for entry in _enabled_source_entries(cfg):
        raw_path = entry["path"]
        if _path_key(raw_path) in json_paths:
            cfg["local_skills_dir"] = raw_path
            cfg["skill_import_managed_local_json_dir"] = raw_path
            notes.append(f"local JSON skills directory set to {raw_path}")
            continue
        if raw_path in by_path:
            merged[by_path[raw_path]] = entry
        else:
            by_path[raw_path] = len(merged)
            merged.append(entry)
            imported += 1

    cfg["extra_skill_dirs"] = merged
    cfg["skill_import_managed_paths"] = sorted(current_paths)
    return imported, notes


def _sources_from_form(form: Any) -> list[dict[str, Any]]:
    paths = [str(v).strip() for v in form.getlist("path")]
    labels = [str(v).strip() for v in form.getlist("source")]
    enabled_rows = set(str(v) for v in form.getlist("enabled"))

    sources: list[dict[str, Any]] = []
    for idx, raw_path in enumerate(paths):
        if not raw_path:
            continue
        label = labels[idx] if idx < len(labels) and labels[idx] else Path(raw_path).name
        sources.append({
            "path": raw_path,
            "source": label,
            "enabled": str(idx) in enabled_rows,
        })
    return sources


def _form_has_sources(form: Any) -> bool:
    return "path" in form


def _audit_from_config(cfg: dict[str, Any]) -> Any:
    paths = default_source_paths(
        skill_import_sources=cfg.get("skill_import_sources") or [],
        extra_skill_dirs=cfg.get("extra_skill_dirs") or [],
    )
    return audit_paths(paths)


def _audit_import_problems(report: Any, cfg: dict[str, Any]) -> list[str]:
    problems: list[str] = list(getattr(report, "errors", []) or [])
    candidates = getattr(report, "candidates", []) or []
    importable_roots = {
        _path_key(c.source_root) for c in candidates
        if c.recommendation == "import" and c.format in {SKILL_MD, LOCAL_JSON}
    }
    for entry in _enabled_source_entries(cfg):
        raw_path = entry["path"]
        if _path_key(raw_path) not in importable_roots:
            problems.append(f"No importable SKILL.md or local JSON skill candidates found in source: {raw_path}")
    json_sources = _enabled_json_source_paths(cfg, report)
    if len(json_sources) > 1:
        problems.append(
            "Only one enabled local JSON source is supported by the current backend: "
            + ", ".join(json_sources)
        )
    return problems


def _audit_notes(report: Any) -> list[str]:
    notes: list[str] = []
    normalize = sum(1 for c in getattr(report, "candidates", []) or [] if c.recommendation == "normalize")
    keep_reference = sum(1 for c in getattr(report, "candidates", []) or [] if c.recommendation == "keep_reference")
    if normalize:
        notes.append(f"{normalize} loose Markdown candidate(s) still need normalization.")
    if keep_reference:
        notes.append(f"{keep_reference} reference file(s) were kept out of live import.")
    return notes


def _repair_notes(result: Any) -> list[str]:
    notes: list[str] = []
    if getattr(result, "created", None):
        notes.append(f"Created {len(result.created)} repaired skill wrapper(s).")
    if getattr(result, "skipped", None):
        notes.append(f"Skipped {len(result.skipped)} already repaired skill(s).")
    notes.extend(getattr(result, "errors", []) or [])
    return notes


def _reindex(request: Request) -> tuple[int, list[str]]:
    try:
        return index_all(request.app.state.store)
    except Exception as exc:  # noqa: BLE001 - render recoverable UI state
        return 0, [f"reindex failed: {exc}"]


async def _config_from_optional_form(request: Request) -> tuple[dict[str, Any], bool]:
    form = await request.form()
    cfg = _cfg.load_config()
    has_sources = _form_has_sources(form)
    if has_sources:
        cfg["skill_import_sources"] = _sources_from_form(form)
    return cfg, has_sources


@router.get("/skill-sources", response_class=HTMLResponse)
def skill_sources_page(request: Request) -> Any:
    return _render_page(request)


@router.post("/skill-sources/save", response_class=HTMLResponse)
async def skill_sources_save(request: Request) -> Any:
    form = await request.form()
    cfg = _cfg.load_config()
    cfg["skill_import_sources"] = _sources_from_form(form)
    problems = _validate_sources(cfg["skill_import_sources"])
    if problems:
        return _render_page(request, problems=problems)
    _cfg.save_config(cfg)
    return _render_page(
        request,
        message=f"Saved {len(cfg['skill_import_sources'])} source(s).",
    )


@router.post("/skill-sources/audit", response_class=HTMLResponse)
async def skill_sources_audit(request: Request) -> Any:
    cfg, has_sources = await _config_from_optional_form(request)
    problems = _validate_sources(cfg.get("skill_import_sources") or [])
    if problems:
        return _render_page(request, problems=problems)
    if has_sources:
        _cfg.save_config(cfg)
    return _render_page(
        request,
        message=f"Saved {len(cfg['skill_import_sources'])} source(s); audit refreshed.",
        audit=_audit_from_config(cfg),
    )


@router.post("/skill-sources/import", response_class=HTMLResponse)
async def skill_sources_import(request: Request) -> Any:
    cfg, has_sources = await _config_from_optional_form(request)
    problems = _validate_sources(cfg.get("skill_import_sources") or [])
    report = _audit_from_config(cfg)
    problems.extend(_audit_import_problems(report, cfg))
    if problems:
        return _render_page(request, problems=problems, audit=report)
    imported, notes = _merge_live_skill_dirs(cfg, report)
    _cfg.save_config(cfg)
    indexed, errors = _reindex(request)
    message = f"Imported {imported} source(s); indexed {indexed} item(s)."
    return _render_page(
        request,
        message=message,
        problems=notes,
        audit=_audit_from_config(cfg),
        index_errors=errors,
    )


@router.post("/skill-sources/apply", response_class=HTMLResponse)
async def skill_sources_apply(request: Request) -> Any:
    form = await request.form()
    action = str(form.get("action") or "save")
    cfg = _cfg.load_config()
    cfg["skill_import_sources"] = _sources_from_form(form)
    problems = _validate_sources(cfg["skill_import_sources"])
    if problems:
        return _render_page(request, problems=problems)

    if action == "audit":
        _cfg.save_config(cfg)
        return _render_page(
            request,
            message=f"Saved {len(cfg['skill_import_sources'])} source(s); audit refreshed.",
            audit=_audit_from_config(cfg),
        )

    if action == "import":
        report = _audit_from_config(cfg)
        problems.extend(_audit_import_problems(report, cfg))
        if problems:
            return _render_page(request, problems=problems, audit=report)
        imported, notes = _merge_live_skill_dirs(cfg, report)
        _cfg.save_config(cfg)
        indexed, errors = _reindex(request)
        message = f"Imported {imported} source(s); indexed {indexed} item(s)."
        return _render_page(
            request,
            message=message,
            problems=notes + _audit_notes(report),
            audit=_audit_from_config(cfg),
            index_errors=errors,
        )

    if action == "fix":
        report = _audit_from_config(cfg)
        repair = repair_importable_skills(report)
        repaired_report = _audit_from_config(cfg)
        problems.extend(_audit_import_problems(repaired_report, cfg))
        if problems:
            return _render_page(
                request,
                problems=_repair_notes(repair) + problems,
                audit=repaired_report,
            )
        imported, notes = _merge_live_skill_dirs(cfg, repaired_report)
        _cfg.save_config(cfg)
        indexed, errors = _reindex(request)
        message = (
            f"Fixed {len(repair.created)} skill(s); imported {imported} source(s); "
            f"indexed {indexed} item(s)."
        )
        return _render_page(
            request,
            message=message,
            problems=_repair_notes(repair) + notes,
            audit=repaired_report,
            index_errors=errors,
        )

    if action == "reindex":
        _cfg.save_config(cfg)
        indexed, errors = _reindex(request)
        return _render_page(
            request,
            message=f"Saved {len(cfg['skill_import_sources'])} source(s); reindexed {indexed} item(s).",
            audit=_audit_from_config(cfg),
            index_errors=errors,
        )

    _cfg.save_config(cfg)
    return _render_page(
        request,
        message=f"Saved {len(cfg['skill_import_sources'])} source(s).",
    )
