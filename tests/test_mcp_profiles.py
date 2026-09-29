"""Process-start MCP profiles as seen by a real MCP client."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


_ROOT = Path(__file__).resolve().parent.parent
_MINIMAL_TOOLS = {
    "prepare_context", "prepare_composition", "compose_context",
    "expand_context_candidate", "search_skills", "get_skill_content",
    "retrieve_compressed", "optimize_prompt_deterministic",
}


def _probe(tmp_path: Path, *args: str) -> dict:
    """Start the server in a fresh process, then use its actual MCP protocol."""
    script = r'''
import asyncio
import json
import sys
from pathlib import Path

from fastmcp import Client
from skill_hub import config, dashboard
from skill_hub.services import registry
from skill_hub.store import Skill, SkillStore

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
config_path = root / "config.json"
config_path.write_text(json.dumps({"services": {"auto_reconcile": False},
                                   "continuous_sweep_enabled": False,
                                   "reindex_sweep_enabled": False}))
config.CONFIG_PATH = config_path
dashboard.render_interactive = lambda *_a, **_kw: None
registry.start_reconciler = lambda *_a, **_kw: None

from skill_hub import server

store = SkillStore(db_path=root / "profile.db")
store.upsert_skill(Skill(id="local:profile-test", name="profile-test",
                         description="Profile test retrieval description",
                         content="FULL SECRET SKILL BODY", file_path="",
                         plugin="", target="claude"))
server._store = store
server.embed_available = lambda: (_ for _ in ()).throw(AssertionError("embedding called"))
server.embed = lambda *_a: (_ for _ in ()).throw(AssertionError("embedding called"))
server.mcp.run = lambda **_kw: None
sys.argv = ["skill-hub", *sys.argv[2:]]
server.main()

async def probe():
    async with Client(server.mcp) as client:
        listed = await client.list_tools()
        names = sorted(t.name for t in listed)
        out = {"names": names, "schemas": [t.model_dump(mode="json") for t in listed],
               "instructions": server.mcp.instructions}
        if "--profile" in sys.argv and "minimal" in sys.argv:
            hit = await client.call_tool("search_skills", {"query": "retrieval"})
            out["search"] = str(hit)
            full = await client.call_tool("get_skill_content", {"skill_id": "local:profile-test"})
            out["full_content"] = full.structured_content
            context = await client.call_tool("prepare_context", {
                "text": "Exact original\nsecond line", "repo_root": str(root)})
            out["original_prompt"] = context.structured_content["original_prompt"]
            draft = await client.call_tool("prepare_composition", {
                "prompt": "Exact original\nsecond line", "project_roots": [str(root)]})
            out["draft_mode"] = draft.structured_content["mode"]
            for label, tool, arguments in (
                ("excluded", "teach", {"rule": "x", "action": "y"}),
                ("training", "prepare_composition", {"prompt": "x", "project_roots": [], "mode": "training"}),
                ("feedback", "compose_context", {"draft_id": draft.structured_content["draft_id"],
                                                  "selected_ids": [], "confirmed": True}),
                ("rerank", "search_skills", {"query": "retrieval", "use_rerank": True}),
                ("inline", "search_skills", {"query": "retrieval", "include_content": True}),
            ):
                try:
                    result = await client.call_tool(tool, arguments)
                    out[label] = {"error": result.is_error or bool((result.structured_content or {}).get("error"))
                                  or any(getattr(item, "text", "").startswith("ERROR:") for item in result.content),
                                  "text": str(result)}
                except Exception as exc:
                    out[label] = {"error": True, "text": str(exc)}
        return out

print(json.dumps(asyncio.run(probe())))
store.close()
'''
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": os.pathsep.join(
        filter(None, (str(_ROOT / "src"), os.environ.get("PYTHONPATH", ""))))}
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), *args],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=45,
    )
    assert process.returncode == 0, process.stderr + process.stdout
    return json.loads(process.stdout.splitlines()[-1])


def test_full_profile_is_default_and_explicit_full_matches(tmp_path):
    default = _probe(tmp_path / "default")
    explicit = _probe(tmp_path / "explicit", "--profile", "full")
    assert default["names"] == explicit["names"]
    assert "teach" in default["names"]
    assert not {"context_learning_status", "record_context_outcome",
                "train_context_selector", "promote_context_selector"} & set(default["names"])


def test_minimal_profile_exposes_only_bounded_everyday_tools(tmp_path):
    result = _probe(tmp_path, "--profile", "minimal")
    assert set(result["names"]) == _MINIMAL_TOOLS
    assert "Unknown tool" in result["excluded"]["text"]
    assert "Profile test retrieval description" in result["search"]
    assert "FULL SECRET SKILL BODY" not in result["search"]
    assert result["full_content"]["content"] == "FULL SECRET SKILL BODY"
    assert result["original_prompt"] == "Exact original\nsecond line"
    assert result["draft_mode"] == "manual"
    for key in ("excluded", "training", "feedback", "rerank", "inline"):
        assert result[key]["error"], (key, result[key])


@pytest.mark.parametrize("profile", ["minimal", "full"])
def test_profile_controls_import_time_background_services(tmp_path, profile):
    script = r'''
import json
import sys
from pathlib import Path

from fastmcp import FastMCP
from skill_hub import config, continuous_sweep, dashboard, reindex_sweep
from skill_hub.services import registry

root = Path(sys.argv[1])
config_path = root / "config.json"
config_path.write_text(json.dumps({"services": {"auto_reconcile": True},
                                   "continuous_sweep_enabled": True,
                                   "reindex_sweep_enabled": True}))
config.CONFIG_PATH = config_path
calls = []
dashboard.render_interactive = lambda *_a, **_kw: calls.append("dashboard")
def record_reconciler(*_a, **_kw):
    calls.append("reconciler")
    return type("Handle", (), {"stop": lambda self: None})()
registry.start_reconciler = record_reconciler
continuous_sweep.start = lambda *_a, **_kw: calls.append("continuous")
reindex_sweep.start = lambda *_a, **_kw: calls.append("reindex")
FastMCP.run = lambda *_a, **_kw: None

from skill_hub import mcp_entry
sys.argv = ["skill-hub", "--profile", sys.argv[2]]
mcp_entry.main()
print(json.dumps(calls))
'''
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": os.pathsep.join(
        filter(None, (str(_ROOT / "src"), os.environ.get("PYTHONPATH", ""))))}
    process = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path), profile],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=45,
    )
    assert process.returncode == 0, process.stderr + process.stdout
    calls = json.loads(process.stdout.splitlines()[-1])
    if profile == "minimal":
        assert calls == []
    else:
        assert set(calls) == {"dashboard", "reconciler", "continuous", "reindex"}


def test_unknown_profile_is_explicit_error(tmp_path):
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": os.pathsep.join(
        filter(None, (str(_ROOT / "src"), os.environ.get("PYTHONPATH", ""))))}
    script = '''
import json
import sys
from skill_hub import mcp_entry
sys.argv = ["skill-hub", "--profile", "unknown"]
try:
    mcp_entry.main()
except SystemExit as exc:
    print(json.dumps({"exit": exc.code, "server_imported": "skill_hub.server" in sys.modules}))
'''
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_ROOT, env=env, capture_output=True, text=True, timeout=45,
    )
    assert process.returncode == 0, process.stderr
    result = json.loads(process.stdout.splitlines()[-1])
    assert result == {"exit": 2, "server_imported": False}
    assert "unknown" in process.stderr
    assert "full" in process.stderr and "minimal" in process.stderr
    assert not (tmp_path / ".claude").exists()

    direct_root = tmp_path / "direct"
    direct_root.mkdir()
    direct = subprocess.run(
        [sys.executable, "-m", "skill_hub.server", "--profile", "unknown"],
        cwd=_ROOT, env={**env, "HOME": str(direct_root)},
        capture_output=True, text=True, timeout=45,
    )
    assert direct.returncode == 2
    assert "unknown" in direct.stderr
    assert not (direct_root / ".claude").exists()
