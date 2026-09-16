"""Wire-contract coverage for the deterministic context MCP tool."""
from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import pytest
from fastmcp import Client


@pytest.fixture()
def server_and_store(tmp_path, monkeypatch):
    """Load the real MCP server with local-only configuration and store state."""
    monkeypatch.setenv("HOME", str(tmp_path))

    from skill_hub import config, dashboard
    from skill_hub import envelope
    from skill_hub.services import registry
    from skill_hub.store import SkillStore

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({
        "services": {"auto_reconcile": False},
        "continuous_sweep_enabled": False,
        "reindex_sweep_enabled": False,
    }))
    monkeypatch.setattr(config, "CONFIG_PATH", config_path)
    monkeypatch.setattr(dashboard, "render_interactive", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(registry, "start_reconciler", lambda *_args, **_kwargs: None)

    original_emit_hook = envelope._emit_hook
    try:
        server = importlib.import_module("skill_hub.server")
        store = SkillStore(db_path=tmp_path / "skill_hub.db")
        monkeypatch.setattr(server, "_store", store)
        yield server, store
        store.close()
    finally:
        envelope.set_emit_hook(original_emit_hook)


def _call_prepare_context(server, arguments):
    async def call():
        async with Client(server.mcp) as client:
            tools = await client.list_tools()
            assert any(tool.name == "prepare_context" for tool in tools)
            return await client.call_tool("prepare_context", arguments)

    return asyncio.run(call())


def test_prepare_context_mcp_returns_structured_scoped_evidence(server_and_store):
    server, store = server_and_store
    task_id = store.save_task(
        title="Alpha migration",
        summary="Use the additive migration sequence.",
        vector=[],
        session_id="session-alpha",
        cwd="/projects/alpha",
    )
    prompt = "Plan the migration.\nKeep this second line unchanged."

    result = _call_prepare_context(server, {
        "text": prompt,
        "repo_root": "/projects/alpha",
        "session_id": "session-alpha",
        "task_id": task_id,
    })

    data = result.structured_content
    assert isinstance(data, dict)
    assert data["original_prompt"] == prompt
    assert data["mode"] == "deterministic"
    assert any(item["source"] == f"task:{task_id}" for item in data["items"])
    assert "Alpha migration" in data["context"]


def test_prepare_context_mcp_without_identity_does_not_return_foreign_task(server_and_store):
    server, store = server_and_store
    store.save_task(
        title="Foreign secret task",
        summary="This belongs only to beta.",
        vector=[],
        session_id="session-beta",
        cwd="/projects/beta",
    )

    result = _call_prepare_context(server, {"text": "continue"})

    data = result.structured_content
    assert isinstance(data, dict)
    assert data["original_prompt"] == "continue"
    assert all(item["kind"] != "task" for item in data["items"])
    assert "Foreign secret task" not in data["context"]
