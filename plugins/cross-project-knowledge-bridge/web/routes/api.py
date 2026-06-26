"""REST endpoints: projects, sync, graph data."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

router = APIRouter()


def _load_config() -> dict:
    plugin_root = Path(__file__).resolve().parent.parent.parent
    mf = plugin_root / "plugin.json"
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


def _get_graph_data() -> dict:
    projects = _resolve_projects()
    nodes = []
    edges = []
    node_ids = set()

    for proj in projects:
        if not proj["exists"]:
            continue
        node_id = f"project:{proj['name']}"
        if node_id not in node_ids:
            nodes.append({
                "id": node_id,
                "type": "project",
                "label": proj["name"],
                "tags": proj["tags"],
            })
            node_ids.add(node_id)

        memory_dir = Path(proj["path"]) / ".memory"
        if not memory_dir.exists():
            continue

        for md_file in memory_dir.glob("*.md"):
            if md_file.name.startswith("_"):
                continue
            entity_id = f"entity:{proj['name']}:{md_file.stem}"
            if entity_id not in node_ids:
                title = md_file.stem.replace("-", " ").replace("_", " ").title()
                nodes.append({
                    "id": entity_id,
                    "type": "entity",
                    "label": title,
                    "project": proj["name"],
                    "file": str(md_file),
                })
                node_ids.add(entity_id)
                edges.append({
                    "source": node_id,
                    "target": entity_id,
                    "type": "contains",
                })

    shared_tags = {}
    for proj in projects:
        for tag in proj.get("tags", []):
            shared_tags.setdefault(tag, []).append(proj["name"])

    for tag, proj_names in shared_tags.items():
        if len(proj_names) > 1:
            tag_node_id = f"tag:{tag}"
            if tag_node_id not in node_ids:
                nodes.append({
                    "id": tag_node_id,
                    "type": "tag",
                    "label": tag,
                })
                node_ids.add(tag_node_id)
            for proj_name in proj_names:
                proj_node_id = f"project:{proj_name}"
                if proj_node_id in node_ids:
                    edges.append({
                        "source": proj_node_id,
                        "target": tag_node_id,
                        "type": "has_tag",
                    })

    return {"nodes": nodes, "edges": edges}


@router.get("/api/projects")
def list_projects(request: Request) -> JSONResponse:
    return JSONResponse(_resolve_projects())


@router.post("/api/sync")
async def trigger_sync(request: Request) -> JSONResponse:
    plugin_root = Path(__file__).resolve().parent.parent.parent
    sync_script = plugin_root / "scripts" / "sync_wiki.py"
    if not sync_script.exists():
        return JSONResponse({"status": "error", "message": "sync script not found"})
    try:
        proc = subprocess.run(
            [sys.executable, str(sync_script)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(plugin_root),
        )
        if proc.returncode != 0:
            return JSONResponse({"status": "error", "message": proc.stderr[:500] or "sync failed"})
        return JSONResponse({"status": "ok", "output": proc.stdout[:500]})
    except subprocess.TimeoutExpired:
        return JSONResponse({"status": "error", "message": "sync timed out"})
    except Exception as exc:
        return JSONResponse({"status": "error", "message": str(exc)})


@router.get("/api/graph")
def get_graph(request: Request) -> JSONResponse:
    return JSONResponse(_get_graph_data())
