"""Memory console routes — enumerate and manage HOT + COLD memory segments."""
from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from skill_hub.store import DB_PATH

router = APIRouter()

_PROJECT_KEY = "-Users-ccancellieri-work-code"


def _memory_dir() -> Path:
    return Path.home() / ".claude" / "projects" / _PROJECT_KEY / "memory"


def _db_segment_info(table: str, where: str = "", label: str | None = None) -> dict[str, Any]:
    """Return a DB segment descriptor. Best-effort."""
    from skill_hub.store import SkillStore
    store = SkillStore()
    try:
        wc = store._conn.execute(
            f"SELECT COUNT(*) FROM {table}" + (f" WHERE {where}" if where else "")
        ).fetchone()
        count = wc[0] if wc else 0
        # Rough estimate: row count * avg row size
        size_bytes = count * 512
        return {
            "id": f"db:{table}" + (f"[{where}]" if where else ""),
            "kind": "db",
            "name": label or table,
            "location": f"db:skill_hub.db#{table}" + (f" WHERE {where}" if where else ""),
            "size_bytes": size_bytes,
            "row_count": count,
            "line_count": None,
            "last_modified": None,
            "vectorized": table in ("tasks", "teachings", "vectors"),
            "grade": None,
            "activity_state": None,
        }
    except Exception as exc:
        return {"id": f"db:{table}", "kind": "db", "name": label or table,
                "location": f"db:{table}", "error": str(exc)}
    finally:
        store.close()


def _file_segment_info(path: Path, name: str, segment_id: str | None = None) -> dict[str, Any]:
    """Return a file segment descriptor."""
    try:
        if not path.exists():
            return {"id": segment_id or str(path), "kind": "file", "name": name,
                    "location": str(path), "exists": False, "size_bytes": 0}
        stat = path.stat()
        content = path.read_text(encoding="utf-8", errors="replace")
        lines = content.count("\n") + 1
        return {
            "id": segment_id or str(path),
            "kind": "file",
            "name": name,
            "location": str(path),
            "exists": True,
            "size_bytes": stat.st_size,
            "row_count": None,
            "line_count": lines,
            "last_modified": stat.st_mtime,
            "vectorized": False,
            "grade": None,
        }
    except Exception as exc:
        return {"id": segment_id or str(path), "kind": "file", "name": name,
                "location": str(path), "error": str(exc)}


@router.get("/memory", response_class=HTMLResponse)
def memory_page(request: Request) -> Any:
    templates = request.app.state.templates
    return templates.TemplateResponse(
        request,
        "memory.html",
        {"active_tab": "memory"},
    )


@router.get("/api/memory/segments")
def list_segments() -> list[dict[str, Any]]:
    """List all HOT and COLD memory segments with size/metadata."""
    segments: list[dict[str, Any]] = []
    mem_dir = _memory_dir()

    # HOT: Global CLAUDE.md
    segments.append(_file_segment_info(
        Path.home() / ".claude" / "CLAUDE.md", "Global CLAUDE.md", "global-claude-md"
    ))

    # HOT: MEMORY.md index
    segments.append(_file_segment_info(
        mem_dir / "MEMORY.md", "MEMORY.md index", "memory-index"
    ))

    # HOT: Project CLAUDE.md
    project_claude = Path.home() / "work" / "code" / "CLAUDE.md"
    segments.append(_file_segment_info(project_claude, "Project CLAUDE.md", "project-claude-md"))

    # HOT: Auto-memory files (grouped by prefix)
    if mem_dir.exists():
        for prefix, label in [
            ("project_", "Project notes"),
            ("feedback_", "Feedback directives"),
            ("reference_", "Reference docs"),
            ("user_", "User identity"),
        ]:
            files = sorted(mem_dir.glob(f"{prefix}*.md"))
            existing = [f for f in files if f.exists()]
            total_size = sum(f.stat().st_size for f in existing)
            total_lines = sum(
                f.read_text(encoding="utf-8", errors="replace").count("\n") + 1
                for f in existing
            )
            last_mtime = max((f.stat().st_mtime for f in existing), default=None)
            segments.append({
                "id": f"auto-memory-{prefix.rstrip('_')}",
                "kind": "file-group",
                "name": f"Auto-memory: {label} ({len(existing)} files)",
                "location": str(mem_dir / f"{prefix}*.md"),
                "exists": len(existing) > 0,
                "size_bytes": total_size,
                "row_count": len(existing),
                "line_count": total_lines,
                "last_modified": last_mtime,
                "vectorized": False,
                "grade": None,
            })

    # COLD: DB segments
    segments.append(_db_segment_info("tasks", "status='open'", "DB: Open tasks"))
    segments.append(_db_segment_info("tasks", "status='closed'", "DB: Closed tasks"))
    segments.append(_db_segment_info("skills", "", "DB: Skills index"))
    segments.append(_db_segment_info("teachings", "", "DB: Teachings"))
    segments.append(_db_segment_info("vectors", "", "DB: Vectors (L0-L4)"))
    segments.append(_db_segment_info("pipeline_runs", "", "DB: Pipeline runs"))

    # Compute totals
    hot_bytes = sum(
        s.get("size_bytes") or 0 for s in segments if s.get("kind") in ("file", "file-group")
    )
    cold_bytes = sum(s.get("size_bytes") or 0 for s in segments if s.get("kind") == "db")

    return [
        {"_meta": {
            "hot_bytes": hot_bytes,
            "cold_bytes": cold_bytes,
            "total_segments": len(segments),
        }},
        *segments,
    ]


@router.get("/api/memory/segments/{segment_id:path}")
def get_segment(segment_id: str) -> dict[str, Any]:
    """Get details for a specific segment."""
    all_segs = list_segments()
    for seg in all_segs:
        if seg.get("id") == segment_id:
            return seg
    raise HTTPException(status_code=404, detail=f"segment {segment_id!r} not found")


# ---------------------------------------------------------------------------
# Compact endpoint
# ---------------------------------------------------------------------------

@router.post("/api/memory/segments/all/compact")
def compact_all() -> dict[str, Any]:
    """Enqueue compact jobs for every segment."""
    from skill_hub.store import SkillStore
    store = SkillStore()
    try:
        all_segs = list_segments()
        segs = [s for s in all_segs if not s.get("_meta")]
        job_ids = []
        for seg in segs:
            try:
                jid = store.enqueue_job(
                    "compact_segment",
                    {"segment_id": seg.get("id"), "location": seg.get("location")},
                )
                job_ids.append(jid)
            except Exception as exc:
                pass  # best-effort
        return {"queued": True, "job_count": len(job_ids), "job_ids": job_ids}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        store.close()


@router.post("/api/memory/segments/{segment_id}/compact")
def compact_segment(segment_id: str) -> dict[str, Any]:
    """Enqueue a compact job for the given segment."""
    from skill_hub.store import SkillStore
    store = SkillStore()
    try:
        # Find the segment to get its location
        all_segs = list_segments()
        location = None
        for seg in all_segs:
            if seg.get("id") == segment_id:
                location = seg.get("location")
                break
        job_id = store.enqueue_job(
            "compact_segment",
            {"segment_id": segment_id, "location": location},
        )
        return {"queued": True, "job_id": job_id, "segment_id": segment_id}
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Archive endpoint (file segments only)
# ---------------------------------------------------------------------------

@router.post("/api/memory/segments/{segment_id}/archive")
def archive_segment(segment_id: str) -> dict[str, Any]:
    """Archive a file segment: move to _archive/ sub-directory (best-effort)."""
    try:
        all_segs = list_segments()
        seg = next((s for s in all_segs if s.get("id") == segment_id), None)
        if seg is None:
            return JSONResponse(status_code=404, content={"error": f"segment {segment_id!r} not found"})

        kind = seg.get("kind", "")
        if kind not in ("file", "file-group"):
            return JSONResponse(
                status_code=400,
                content={"error": "archive is only supported for file and file-group segments"},
            )

        location = seg.get("location", "")
        src = Path(location.replace("*", "")).resolve()

        if kind == "file":
            if not src.exists():
                return {"queued": True, "archived": 0, "note": "file not found, nothing to archive"}
            archive_dir = src.parent / "_archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
            dest = archive_dir / f"{src.stem}_{ts}{src.suffix}"
            shutil.move(str(src), str(dest))
            return {"queued": True, "archived": 1, "dest": str(dest)}

        # file-group: archive all matching files
        parent = src if src.is_dir() else src.parent
        # Reconstruct the glob pattern from location (e.g. ".../*.md" → glob "*.md")
        glob_pattern = Path(location).name if "*" in location else "*.md"
        files = sorted(parent.glob(glob_pattern))
        if not files:
            return {"queued": True, "archived": 0, "note": "no files matched"}
        archive_dir = parent / "_archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        archived = []
        for f in files:
            dest = archive_dir / f"{f.stem}_{ts}{f.suffix}"
            try:
                shutil.move(str(f), str(dest))
                archived.append(str(dest))
            except Exception:
                pass
        return {"queued": True, "archived": len(archived), "dests": archived}

    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})
