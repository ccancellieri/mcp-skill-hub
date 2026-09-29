"""Exercise an installed Skill Hub wheel over stdio without a model or live data.

Example::

    PYTHONPATH=/tmp/skill-hub-wheel-target python benchmarks/installed_smoke.py \
        --expected-package-root /tmp/skill-hub-wheel-target

The caller's Python environment must provide FastMCP. The subprocesses resolve
Skill Hub from the supplied installed target, never from this checkout's src/.
This is an installation/integration smoke test, not a task-benefit benchmark.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

EXPECTED_TOOLS = {
    "prepare_context", "prepare_composition", "compose_context",
    "expand_context_candidate", "search_skills", "get_skill_content",
    "retrieve_compressed", "optimize_prompt_deterministic",
}
SKILL_ID = "qa-skills:review"
MARKER = "QA_SYNTHETIC_MIGRATION_RULE"
PROMPT = "Review database migrations.\nPreserve rollback ordering."


def _child_env(home: Path, package_root: Path) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items()
           if key in {"PATH", "TMPDIR", "LANG", "LC_ALL", "SYSTEMROOT"}}
    env.update({
        "HOME": str(home),
        "PYTHONPATH": str(package_root),
        "PYTHONDONTWRITEBYTECODE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    return env


def _resolve_package(env: dict[str, str], cwd: Path) -> tuple[Path, str]:
    script = (
        "import importlib.metadata, json, skill_hub; "
        "print(json.dumps({'path': skill_hub.__file__, "
        "'version': importlib.metadata.version('mcp-skill-hub')}))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=cwd, env=env,
        capture_output=True, text=True, timeout=20, check=True,
    )
    value = json.loads(result.stdout)
    return Path(value["path"]).resolve(), value["version"]


def _text(result: object) -> str:
    return "\n".join(getattr(item, "text", "") for item in result.content)


async def _probe(env: dict[str, str], cwd: Path) -> dict[str, object]:
    transport = StdioTransport(
        command=sys.executable,
        args=["-m", "skill_hub.mcp_entry", "--profile", "minimal"],
        env=env, cwd=str(cwd),
    )
    async with Client(transport, timeout=45, init_timeout=45) as client:
        listed = await client.list_tools()
        names = {tool.name for tool in listed}
        assert names == EXPECTED_TOOLS, f"unexpected minimal tool set: {sorted(names)}"

        search = await client.call_tool("search_skills", {"query": "database migrations"})
        search_text = _text(search)
        assert not search.is_error, search_text
        assert "Review database migrations" in search_text, search_text
        assert MARKER not in search_text, "description search leaked full skill text"

        full = await client.call_tool("get_skill_content", {"skill_id": SKILL_ID})
        assert not full.is_error, _text(full)
        assert MARKER in full.structured_content["content"]

        prepared = await client.call_tool("prepare_context", {"text": PROMPT})
        assert not prepared.is_error, _text(prepared)
        assert prepared.structured_content["original_prompt"] == PROMPT

        draft_result = await client.call_tool("prepare_composition", {
            "prompt": PROMPT, "project_roots": [str(cwd)], "mode": "manual",
        })
        assert not draft_result.is_error, _text(draft_result)
        draft = draft_result.structured_content
        assert draft["original_prompt"] == PROMPT
        assert draft["mode"] == "manual"
        skill = next((item for item in draft["candidates"]
                      if item["source"] == f"skill:{SKILL_ID}"), None)
        assert skill is not None, "indexed skill missing from composition candidates"

        expanded = await client.call_tool("expand_context_candidate", {
            "draft_id": draft["draft_id"], "candidate_id": skill["candidate_id"],
        })
        assert not expanded.is_error, _text(expanded)
        assert MARKER in expanded.structured_content["text"]

        composed = await client.call_tool("compose_context", {
            "draft_id": draft["draft_id"],
            "selected_ids": [skill["candidate_id"]],
            "confirmed": False,
        })
        assert not composed.is_error, _text(composed)
        assert composed.structured_content["original_prompt"] == PROMPT, "composition rewrote prompt"
        assert f"skill:{SKILL_ID}" in composed.structured_content["context"], (
            "composition omitted selected skill reference"
        )

        optimized = await client.call_tool("optimize_prompt_deterministic", {"text": PROMPT})
        assert not optimized.is_error, _text(optimized)
        assert optimized.structured_content["original_prompt"] == PROMPT, "optimizer rewrote prompt"

        return {"tool_count": len(names), "searched_description_only": True,
                "full_text_loaded": True, "prompt_preserved": True,
                "manual_composition": True, "model_calls_requested": 0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-package-root", required=True, type=Path,
                        help="Directory containing the installed wheel package")
    args = parser.parse_args()
    package_root = args.expected_package_root.resolve()
    if not (package_root / "skill_hub").is_dir():
        parser.error("expected package root has no installed skill_hub directory")

    stage = "setup"
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="skill-hub-installed-smoke-") as directory:
            root = Path(directory)
            home = root / "home"
            home.mkdir()
            skill_root = root / "qa-skills"
            skill_file = skill_root / "skills" / "review" / "SKILL.md"
            skill_file.parent.mkdir(parents=True)
            skill_file.write_text(
                "---\nname: review\n"
                "description: Review database migrations and rollback ordering\n"
                "---\n# Review\n" + MARKER + ": preserve rollback ordering.\n",
                encoding="utf-8",
            )
            config_file = home / ".claude" / "mcp-skill-hub" / "config.json"
            config_file.parent.mkdir(parents=True)
            config_file.write_text(json.dumps({
                "services": {"auto_reconcile": False},
                "continuous_sweep_enabled": False,
                "reindex_sweep_enabled": False,
            }), encoding="utf-8")
            env = _child_env(home, package_root)

            stage = "package_origin"
            package_file, version = _resolve_package(env, root)
            if not package_file.is_relative_to(package_root):
                raise AssertionError(f"package resolved outside installed target: {package_file}")

            stage = "text_index"
            indexed = subprocess.run(
                [sys.executable, "-m", "skill_hub.cli", "index_skills_text",
                 "--skill-dir", str(skill_root)],
                cwd=root, env=env, capture_output=True, text=True,
                timeout=60, check=True,
            )
            if "Indexed 1 skills" not in indexed.stdout:
                raise AssertionError(f"unexpected index result: {indexed.stdout[-1000:]}")

            stage = "stdio_mcp"
            checks = asyncio.run(_probe(env, root))
            print(json.dumps({
                "status": "passed", "evidence_level": "installed_stdio_integration_only",
                "package_file": str(package_file), "version": version,
                "elapsed_seconds": round(time.monotonic() - started, 3),
                **checks,
            }, sort_keys=True))
            return 0
    except Exception as exc:  # noqa: BLE001 - report the failing integration stage as JSON
        print(json.dumps({"status": "failed", "stage": stage,
                          "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
