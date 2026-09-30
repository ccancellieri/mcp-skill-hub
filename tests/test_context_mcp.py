"""Wire-contract coverage for the deterministic context MCP tool."""
from __future__ import annotations

import asyncio
import importlib
import json
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError
from mcp.types import Implementation
from skill_hub.store import Skill


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


def _call_prepare_context(server, arguments, *, client_info=None):
    async def call():
        async with Client(server.mcp, client_info=client_info) as client:
            tools = await client.list_tools()
            assert any(tool.name == "prepare_context" for tool in tools)
            return await client.call_tool("prepare_context", arguments)

    return asyncio.run(call())


def test_composer_tools_are_available_without_models(server_and_store):
    server, _store = server_and_store
    async def call():
        async with Client(server.mcp) as client:
            names = {tool.name for tool in await client.list_tools()}
            assert {"prepare_composition", "compose_context", "optimize_prompt_deterministic"} <= names
            result = await client.call_tool("prepare_composition", {"prompt": "Keep this prompt", "project_roots": [], "token_budget": 900})
            data = result.structured_content
            assert data["original_prompt"] == "Keep this prompt"
            composed = await client.call_tool("compose_context", {"draft_id": data["draft_id"], "selected_ids": []})
            assert composed.structured_content["context"] == ""
    asyncio.run(call())


def test_compact_composer_mcp_contract_and_legacy_full_response(server_and_store):
    server, store = server_and_store
    store.upsert_skill(Skill(
        id="backend:postgres", name="postgres", description="Postgres migration guidance.",
        content="# PostgreSQL\nFull instructions.",
        file_path="/trusted/postgres/SKILL.md", plugin="backend",
    ))
    store.save_task(
        title="Postgres migration", summary="Alpha task full text.", vector=[],
        session_id="session-a", cwd="/repos/alpha", repo="alpha",
    )

    async def call():
        async with Client(server.mcp) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            prepare_schema = tools["prepare_composition"].inputSchema
            expand_schema = tools["expand_context_candidate"].inputSchema
            assert prepare_schema["properties"]["detail"]["default"] == "preview"
            assert set(prepare_schema["properties"]["detail"]["enum"]) == {"preview", "index"}
            assert expand_schema["properties"]["detail"]["default"] == "full"
            assert set(expand_schema["properties"]["detail"]["enum"]) == {"full", "compact"}
            candidate_schema = expand_schema["properties"]["candidate_id"]
            assert {part["type"] for part in candidate_schema["anyOf"]} == {"string", "array"}

            prompt = "Postgres migration.\nKeep this exact line."
            prepared = await client.call_tool("prepare_composition", {
                "prompt": prompt, "project_roots": ["/repos/alpha"], "detail": "index",
            })
            data = prepared.structured_content
            assert data["original_prompt"] == prompt
            assert data["needs_review"] is True
            assert all("text" not in item for item in data["candidates"])
            assert "Alpha task full text." not in json.dumps(data)
            assert all(item["project_root"] in {"", "/repos/alpha"} for item in data["candidates"])

            ids = [item["candidate_id"] for item in data["candidates"][:2]]
            batch = await client.call_tool("expand_context_candidate", {
                "draft_id": data["draft_id"], "candidate_id": ids, "detail": "compact",
            })
            assert batch.structured_content["draft_id"] == data["draft_id"]
            assert [item["candidate_id"] for item in batch.structured_content["items"]] == ids
            assert any("Full instructions." in item["text"] for item in batch.structured_content["items"])

            legacy = await client.call_tool("expand_context_candidate", {
                "draft_id": data["draft_id"], "candidate_id": ids[0],
            })
            assert legacy.structured_content["candidate_id"] == ids[0]
            assert "items" not in legacy.structured_content
            assert "features" in legacy.structured_content
            assert "score" in legacy.structured_content

            with pytest.raises(ToolError, match="detail"):
                await client.call_tool("prepare_composition", {
                    "prompt": prompt, "detail": "novel",
                })
            with pytest.raises(ToolError, match="unknown_parameter"):
                await client.call_tool("prepare_composition", {
                    "prompt": prompt, "unknown_parameter": "ignored?",
                })

    asyncio.run(call())


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


def test_prepare_context_mcp_runtime_arguments_are_caller_reported(server_and_store, monkeypatch):
    server, _store = server_and_store
    observed = []
    monkeypatch.setattr(
        "skill_hub.runtime_context.observe_runtime",
        lambda value, **kwargs: observed.append((value, kwargs)),
    )

    result = _call_prepare_context(server, {
        "text": "keep prompt unchanged",
        "session_id": "caller-session",
        "runtime": {
            "client": {"id": "codex"},
            "session": {"id": "caller-session"},
            "model": {"id": "gpt-5"},
            "effort": {"value": "high", "scheme": "reasoning_effort"},
            "provenance": {"model_id": "native_event"},
        },
    }, client_info=Implementation(name="codex", version="1.0"))

    assert result.structured_content["original_prompt"] == "keep prompt unchanged"
    assert observed
    assert observed[0][1]["default_source"] == "caller_reported"
    assert observed[0][1]["honor_provenance"] is True
    assert observed[0][0]["provenance"]["client_id"] == "mcp_client_info"
    assert "model_id" not in observed[0][0]["provenance"]


def test_prepare_context_old_mcp_client_with_session_remains_supported(server_and_store, monkeypatch):
    server, _store = server_and_store
    observed = []
    monkeypatch.setattr(
        "skill_hub.runtime_context.observe_runtime",
        lambda value, **kwargs: observed.append((value, kwargs)),
    )

    result = _call_prepare_context(
        server,
        {"text": "old request", "session_id": "legacy-session"},
        client_info=Implementation(name="codex", version="0.9"),
    )

    assert result.structured_content["original_prompt"] == "old request"
    assert observed[0][0]["session"] == {"id": "legacy-session"}
    assert observed[0][0]["client"] == {"id": "codex", "version": "0.9"}
