"""Route coverage for the deterministic context workspace."""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


def _client(monkeypatch) -> TestClient:
    from skill_hub.services import registry as reg_mod
    from skill_hub.webapp.main import create_app

    reg_mod.set_registry(reg_mod.ServiceRegistry([]))
    return TestClient(create_app(store="store"))


def test_context_page_renders_workspace(monkeypatch):
    client = _client(monkeypatch)

    response = client.get("/context")

    assert response.status_code == 200
    assert "Build context" in response.text
    assert 'name="prompt"' in response.text
    assert "No context built yet" in response.text
    assert 'href="/context"' in response.text
    assert "Claude hook" in response.text
    assert "Codex MCP" in response.text
    assert "OpenClaw plugin" in response.text


def test_context_build_renders_result_and_passes_optional_inputs(monkeypatch):
    import skill_hub.webapp.routes.context as context_route

    received = {}

    def fake_build_context(prompt, *, cwd="", session_id="", task_id=None, store=None, cfg=None):
        received.update(
            prompt=prompt,
            cwd=cwd,
            session_id=session_id,
            task_id=task_id,
            store=store,
            cfg=cfg,
        )
        return {
            "original_prompt": prompt,
            "context": "Use the project convention.",
            "items": [{
                "kind": "memory",
                "title": "Project convention",
                "source": "<script>source</script>",
                "text": "<script>item</script>",
            }],
            "warnings": ["One source was truncated."],
            "elapsed_ms": 18,
            "mode": "deterministic",
            "selected_count": 1,
            "omitted_count": 2,
        }

    monkeypatch.setattr(context_route, "build_context", fake_build_context)
    client = _client(monkeypatch)

    response = client.post(
        "/context/build",
        data={
            "prompt": "Add a context page",
            "cwd": "/work/project",
            "session_id": "session-42",
            "task_id": "7",
        },
    )

    assert response.status_code == 200
    assert received == {
        "prompt": "Add a context page",
        "cwd": "/work/project",
        "session_id": "session-42",
        "task_id": 7,
        "store": "store",
        "cfg": None,
    }
    assert "Original prompt" in response.text
    assert "Add a context page" in response.text
    assert "Raw assembled context" in response.text
    assert "Use the project convention." in response.text
    assert "Project convention" in response.text
    assert "&lt;script&gt;source&lt;/script&gt;" in response.text
    assert "&lt;script&gt;item&lt;/script&gt;" in response.text
    assert "<script>source</script>" not in response.text
    assert "1 selected" in response.text
    assert "2 excluded by limits" in response.text
    assert "18 ms" in response.text
    assert "One source was truncated." in response.text


def test_context_build_rejects_invalid_task_id(monkeypatch):
    import skill_hub.webapp.routes.context as context_route

    monkeypatch.setattr(context_route, "build_context", lambda *args, **kwargs: {})
    client = _client(monkeypatch)

    response = client.post("/context/build", data={"prompt": "hello", "task_id": "nope"})

    assert response.status_code == 422
    assert "Task ID must be a whole number" in response.text


def test_context_page_versions_local_assets(monkeypatch):
    import re

    client = _client(monkeypatch)
    response = client.get("/context")
    assets = re.findall(r'(?:href|src)="(/static/app\.(?:css|js)\?v=[a-f0-9]+)"', response.text)
    assert len(assets) == 2
    for asset in assets:
        assert client.get(asset).status_code == 200


def test_context_build_preserves_prompt_and_rejects_whitespace_only(monkeypatch):
    import skill_hub.webapp.routes.context as context_route

    received = {}

    def fake_build_context(prompt, **kwargs):
        received["prompt"] = prompt
        return {
            "original_prompt": prompt,
            "context": "",
            "items": [],
            "warnings": [],
            "elapsed_ms": 1,
            "mode": "deterministic",
            "selected_count": 0,
            "omitted_count": 0,
        }

    monkeypatch.setattr(context_route, "build_context", fake_build_context)
    client = _client(monkeypatch)

    response = client.post("/context/build", data={"prompt": "  keep this exact  \n"})
    blank = client.post("/context/build", data={"prompt": " \n\t "})

    assert response.status_code == 200
    assert received["prompt"] == "  keep this exact  \n"
    assert blank.status_code == 422
    assert "Prompt is required" in blank.text


def test_context_build_returns_unavailable_result_when_core_fails(monkeypatch):
    import skill_hub.webapp.routes.context as context_route

    def fail_build_context(*args, **kwargs):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr(context_route, "build_context", fail_build_context)
    client = _client(monkeypatch)

    response = client.post("/context/build", data={"prompt": "keep me"})

    assert response.status_code == 200
    assert "keep me" in response.text
    assert "Context is unavailable right now" in response.text


def test_context_build_allows_missing_task_id(monkeypatch):
    import skill_hub.webapp.routes.context as context_route

    received = {}
    monkeypatch.setattr(
        context_route,
        "build_context",
        lambda prompt, **kwargs: received.update(kwargs) or {
            "original_prompt": prompt, "context": "", "items": [], "warnings": [],
            "elapsed_ms": 1, "mode": "deterministic", "selected_count": 0, "omitted_count": 0,
        },
    )
    client = _client(monkeypatch)

    response = client.post("/context/build", data={"prompt": "no task"})

    assert response.status_code == 200
    assert received["task_id"] is None


def test_settings_exposes_context_controls_and_hides_retired_automation(tmp_path, monkeypatch):
    import json
    from skill_hub import config as cfg_mod

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "context_enabled": True,
        "context_max_chars": 12000,
        "context_max_items": 8,
        "context_hook_timeout_s": 2.0,
        "context_project_aliases": {"work": "/work/project"},
        "hook_approval_policy": "native",
        "auto_proceed_enabled": True,
        "router_enabled": True,
        "router_tier2_timeout": 3,
        "improve_prompt_default_chain": ["old-normalizer"],
        "prompt_normalizer_enabled": True,
    }))
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", config_path)
    client = _client(monkeypatch)

    response = client.get("/settings")

    assert response.status_code == 200
    assert "context_enabled" in response.text
    assert "context_max_chars" in response.text
    assert "context_max_items" in response.text
    assert "context_hook_timeout_s" in response.text
    assert "context_project_aliases" in response.text
    assert "hook_approval_policy" in response.text
    assert "Native client manages continuation and permissions" in response.text
    assert "auto_proceed_enabled" not in response.text
    assert "router_enabled" in response.text
    assert "router_tier2_timeout" not in response.text
    assert "improve_prompt_default_chain" not in response.text
    assert "prompt_normalizer_enabled" not in response.text


def test_task_options_accepts_only_known_work_states():
    from fastapi import FastAPI
    from skill_hub.webapp.routes.tasks import router as tasks_router

    class Store:
        options = {}

        def set_task_options(self, task_id, patch):
            self.options.update(patch)
            return task_id == 4

        def get_task_options(self, task_id):
            return self.options

    app = FastAPI()
    app.state.store = Store()
    app.include_router(tasks_router)
    client = TestClient(app)

    valid = client.post("/tasks/4/options", json={"work_state": "waiting_user"})
    invalid = client.post("/tasks/4/options", json={"work_state": "resuming"})

    assert valid.status_code == 200
    assert valid.json()["options"]["work_state"] == "waiting_user"
    assert invalid.status_code == 422
    assert "work_state" in invalid.json()["error"]
