"""A fresh minimal server can use an explicitly indexed skill without a model."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def test_text_only_cli_bootstraps_minimal_mcp_from_selected_directory(tmp_path):
    skill_dir = tmp_path / "my-skills"
    skill_file = skill_dir / "skills" / "review" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("---\nname: review\ndescription: Review database migrations\n---\n"
                          "# Review\nCheck migration ordering.\n")
    db_path = tmp_path / "fresh.db"
    script = r'''
import asyncio
import json
import sys
from pathlib import Path

from skill_hub import cli, config, indexer, mcp_profile_state
from skill_hub.store import SkillStore

root, db_path, skill_dir = map(Path, sys.argv[1:])
config.CONFIG_PATH = root / "config.json"
config.CONFIG_PATH.write_text(json.dumps({"services": {"auto_reconcile": False},
                                         "continuous_sweep_enabled": False,
                                         "reindex_sweep_enabled": False}))
indexer.embed = lambda *_a, **_kw: (_ for _ in ()).throw(AssertionError("embedding called"))
sys.argv = ["skill-hub-cli", "index_skills_text", "--db", str(db_path),
            "--skill-dir", str(skill_dir)]
cli.main()

mcp_profile_state.profile = "minimal"
from skill_hub import server
server._store = SkillStore(db_path=db_path)
server.embed_available = lambda: (_ for _ in ()).throw(AssertionError("embedding called"))
server.embed = lambda *_a: (_ for _ in ()).throw(AssertionError("embedding called"))
server.mcp.run = lambda **_kw: None
server.main(profile="minimal")

from fastmcp import Client
async def probe():
    async with Client(server.mcp) as client:
        listed = {tool.name for tool in await client.list_tools()}
        hits = await client.call_tool("search_skills", {"query": "database migrations"})
        full = await client.call_tool("get_skill_content", {"skill_id": "my-skills:review"})
        return {"tools": sorted(listed), "search": str(hits),
                "content": full.structured_content["content"]}

print(json.dumps(asyncio.run(probe())))
'''
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([sys.executable, "-c", script, str(tmp_path), str(db_path),
                             str(skill_dir)], cwd=ROOT, env=env, text=True,
                            capture_output=True, timeout=45)
    assert result.returncode == 0, result.stderr + result.stdout
    data = json.loads(result.stdout.splitlines()[-1])
    assert "search_skills" in data["tools"]
    assert "index_skills" not in data["tools"]
    assert "Review database migrations" in data["search"]
    assert "Check migration ordering." in data["content"]
    assert not (tmp_path / ".claude" / "settings.json").exists()


def test_text_only_cli_requires_deliberate_skill_directory(tmp_path):
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([sys.executable, "-m", "skill_hub.cli", "index_skills_text",
                             "--db", str(tmp_path / "fresh.db")], cwd=ROOT, env=env,
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 2
    assert "--skill-dir" in result.stderr
    assert not (tmp_path / "fresh.db").exists()


def test_text_only_cli_exits_nonzero_when_existing_vector_blocks_update(tmp_path):
    from skill_hub.store import Skill, SkillStore

    skill_dir = tmp_path / "my-skills"
    skill_file = skill_dir / "skills" / "review" / "SKILL.md"
    skill_file.parent.mkdir(parents=True)
    skill_file.write_text("---\nname: review\ndescription: New advice\n---\n# Review\nNew body.\n")
    db_path = tmp_path / "existing.db"
    store = SkillStore(db_path=db_path)
    store.upsert_skill(Skill(id="my-skills:review", name="review",
                             description="Old advice", content="Old body.",
                             file_path=str(skill_file), plugin="my-skills", target="claude"),
                       content_hash="old")
    store.upsert_embedding("my-skills:review", "old-model", [0.1, 0.2, 0.3])
    store.close()

    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": str(ROOT / "src")}
    result = subprocess.run([sys.executable, "-m", "skill_hub.cli", "index_skills_text",
                             "--db", str(db_path), "--skill-dir", str(skill_dir)],
                            cwd=ROOT, env=env, text=True, capture_output=True, timeout=20)
    assert result.returncode == 1
    assert "existing embedding" in result.stdout
    reopened = SkillStore(db_path=db_path)
    assert reopened.get_skill_content("my-skills:review") == "Old body."
    reopened.close()
