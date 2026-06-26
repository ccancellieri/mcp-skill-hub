"""FastAPI sub-app for cross-project knowledge graph visualization.

Mounted at /knowledge-bridge by the main skill-hub webapp. Provides:
- Interactive D3.js knowledge graph showing connections across projects
- Project list with decision/pattern counts
- Sync status and manual trigger buttons
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = PLUGIN_ROOT / "web" / "templates"


def get_app() -> FastAPI:
    app = FastAPI(
        title="cross-project-knowledge-bridge",
        docs_url=None,
        redoc_url=None,
    )
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
    app.state.templates = templates

    router = APIRouter()

    def _get_store():
        parent_store = getattr(app.state, "parent_store", None)
        if parent_store:
            return parent_store
        from skill_hub.store import SkillStore
        return SkillStore()

    def _load_config() -> dict:
        mf = PLUGIN_ROOT / "plugin.json"
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

    def _get_graph_data(store: Any) -> dict:
        config = _load_config()
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

    @router.get("/", response_class=HTMLResponse)
    def index(request: Request) -> Any:
        store = _get_store()
        templates = app.state.templates
        config = _load_config()
        projects = _resolve_projects()
        graph = _get_graph_data(store)
        return templates.TemplateResponse(
            request,
            "graph.html",
            {
                "active_tab": "knowledge-bridge",
                "projects": projects,
                "graph": graph,
                "config": config,
            },
        )

    @router.get("/graph.json")
    def graph_json(request: Request) -> dict:
        store = _get_store()
        return _get_graph_data(store)

    @router.get("/projects")
    def list_projects(request: Request) -> list[dict]:
        return _resolve_projects()

    @router.post("/sync")
    async def trigger_sync(request: Request) -> dict:
        import subprocess
        sync_script = PLUGIN_ROOT / "scripts" / "sync_wiki.py"
        if not sync_script.exists():
            return {"status": "error", "message": "sync script not found"}
        try:
            proc = subprocess.run(
                [sys.executable, str(sync_script)],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=str(PLUGIN_ROOT),
            )
            if proc.returncode != 0:
                return {"status": "error", "message": proc.stderr[:500] or "sync failed"}
            return {"status": "ok", "output": proc.stdout[:500]}
        except subprocess.TimeoutExpired:
            return {"status": "error", "message": "sync timed out"}
        except Exception as exc:
            return {"status": "error", "message": str(exc)}

    @router.get("/decisions/{project_name}")
    def get_decisions(request: Request, project_name: str) -> list[dict]:
        projects = _resolve_projects()
        proj = next((p for p in projects if p["name"] == project_name), None)
        if not proj or not proj["exists"]:
            return []
        decisions_file = Path(proj["path"]) / ".memory" / "decisions.md"
        if not decisions_file.exists():
            return []
        try:
            text = decisions_file.read_text(encoding="utf-8")
            return _parse_decisions(text)
        except OSError:
            return []

    def _parse_decisions(text: str) -> list[dict]:
        import re
        decisions = []
        current = None
        for line in text.splitlines():
            heading = re.match(r"^#+\s+(.+)$", line)
            if heading:
                if current:
                    decisions.append(current)
                current = {"title": heading.group(1).strip(), "body": ""}
            elif current:
                current["body"] += line + "\n"
        if current:
            decisions.append(current)
        return decisions

    app.include_router(router)
    return app
