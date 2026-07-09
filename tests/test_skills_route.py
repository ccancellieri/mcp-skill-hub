from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub import config as cfg_mod  # noqa: E402
from skill_hub.services import registry as reg_mod  # noqa: E402
from skill_hub.services.base import Service, Status  # noqa: E402


class _FakeService(Service):
    name = "noop"
    label = "Noop"
    description = ""

    def status(self) -> Status:
        return "stopped"

    def is_available(self):
        return True, ""

    def start(self):
        return True, ""

    def stop(self):
        return True, ""


class _FakeStore:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self._conn = None

    def get_skill_usage_stats(self) -> list[dict]:
        return [dict(r) for r in self.rows]

    def get_skill(self, skill_id: str) -> dict | None:
        for row in self.rows:
            if row["id"] == skill_id:
                return dict(row)
        return None


def _write_skill(root: Path, name: str) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"""---
name: {name}
description: Use when testing {name}.
---

# {name}
""",
        encoding="utf-8",
    )
    return skill_file


def _client(tmp_path: Path, monkeypatch, store: _FakeStore) -> TestClient:
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", cfg_path)
    cfg_path.write_text(json.dumps({}))
    reg_mod.set_registry(reg_mod.ServiceRegistry([_FakeService()]))

    import skill_hub.webapp.routes.skills as skills_route

    monkeypatch.setattr(skills_route, "CLAUDE_SKILLS_DIR", tmp_path / "claude-skills")
    monkeypatch.setattr(
        skills_route,
        "CLAUDE_PLUGIN_CACHE_DIR",
        tmp_path / "claude-plugins" / "cache",
    )

    from skill_hub.webapp.main import create_app

    return TestClient(create_app(store=store))


def test_skills_page_shows_claude_visibility_and_link_action(tmp_path, monkeypatch):
    agents_skill = _write_skill(tmp_path / "agents" / "skills", "fastapi-optimization-skill")
    claude_skill = _write_skill(tmp_path / "claude-skills", "ogc-api-standards")
    store = _FakeStore([
        {
            "id": "agents:fastapi-optimization-skill",
            "name": "fastapi-optimization-skill",
            "description": "Use for FastAPI optimization.",
            "content": "# fastapi",
            "file_path": str(agents_skill),
            "plugin": "agents",
            "target": "claude",
            "feedback_score": 1.0,
            "injections": 0,
            "helpful": 0,
            "unhelpful": 0,
            "last_used": None,
            "helpful_pct": None,
        },
        {
            "id": "claude-user:ogc-api-standards",
            "name": "ogc-api-standards",
            "description": "Use for OGC standards.",
            "content": "# ogc",
            "file_path": str(claude_skill),
            "plugin": "claude-user",
            "target": "claude",
            "feedback_score": 1.0,
            "injections": 0,
            "helpful": 0,
            "unhelpful": 0,
            "last_used": None,
            "helpful_pct": None,
        },
    ])
    client = _client(tmp_path, monkeypatch, store)

    response = client.get("/skills")

    assert response.status_code == 200
    assert "Skill Hub only" in response.text
    assert "Claude user" in response.text
    assert "/skills/agents:fastapi-optimization-skill/link-claude" in response.text


def test_link_and_unlink_claude_skill_uses_managed_symlink(tmp_path, monkeypatch):
    agents_skill = _write_skill(tmp_path / "agents" / "skills", "fastapi-optimization-skill")
    store = _FakeStore([
        {
            "id": "agents:fastapi-optimization-skill",
            "name": "fastapi-optimization-skill",
            "description": "Use for FastAPI optimization.",
            "content": "# fastapi",
            "file_path": str(agents_skill),
            "plugin": "agents",
            "target": "claude",
            "feedback_score": 1.0,
            "injections": 0,
            "helpful": 0,
            "unhelpful": 0,
            "last_used": None,
            "helpful_pct": None,
        }
    ])
    client = _client(tmp_path, monkeypatch, store)
    link_path = tmp_path / "claude-skills" / "fastapi-optimization-skill"

    response = client.post("/skills/agents:fastapi-optimization-skill/link-claude")

    assert response.status_code == 200
    assert link_path.is_symlink()
    assert link_path.resolve() == agents_skill.parent.resolve()
    assert "Claude link" in response.text
    assert "unlink-claude" in response.text

    response = client.post("/skills/agents:fastapi-optimization-skill/unlink-claude")

    assert response.status_code == 200
    assert not link_path.exists()
    assert "Skill Hub only" in response.text
