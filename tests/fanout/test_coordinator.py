"""Integration tests for skill_hub.fanout.coordinator.

Uses a real tmp git repo + tmp SkillStore but does NOT launch any Claude
sessions (coordinator only calls `ensure_worktree`, not `launch_session`).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.fanout.coordinator import fanout
from skill_hub.fanout.sources import Issue
from skill_hub.store import SkillStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_repo(tmp_path: Path) -> Path:
    """Build a tmp repo root that the worktree module's project resolver
    can find. The resolver searches `worktree.repo_roots` for `<root>/<name>`."""
    root = tmp_path / "repos"
    project = root / "demo"
    project.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=project, check=True)
    subprocess.run(["git", "-C", str(project), "config", "user.email", "a@b.c"], check=True)
    subprocess.run(["git", "-C", str(project), "config", "user.name", "T"], check=True)
    (project / "README.md").write_text("demo\n")
    subprocess.run(["git", "-C", str(project), "add", "."], check=True)
    subprocess.run(["git", "-C", str(project), "commit", "-q", "-m", "init"], check=True)
    return root


@pytest.fixture
def patch_repo_roots(monkeypatch, tmp_repo: Path):
    """Point skill_hub.worktree at the tmp repos root."""
    from skill_hub import worktree as _wt
    monkeypatch.setattr(_wt, "_default_repo_roots", lambda: [tmp_repo])
    return tmp_repo


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_fanout_dry_run_no_side_effects(patch_repo_roots, tmp_path: Path):
    store = SkillStore(db_path=tmp_path / "store.db")
    result = fanout(
        "text",
        filter="- one\n- two",
        limit=2,
        project="demo",
        dry_run=True,
        use_llm=False,
        store=store,
    )
    assert len(result.task_ids) == 2
    assert all(tid == 0 for tid in result.task_ids)
    assert "Agent(" in result.directive
    # No tasks should have been persisted in dry_run.
    rows = store.list_tasks(status="open")
    assert rows == []


def test_fanout_creates_worktrees_and_tasks(patch_repo_roots, tmp_path: Path):
    store = SkillStore(db_path=tmp_path / "store.db")
    result = fanout(
        "text",
        filter="- fix login\n- add pagination",
        limit=2,
        project="demo",
        dry_run=False,
        use_llm=False,
        store=store,
    )
    assert len(result.task_ids) == 2
    assert all(tid > 0 for tid in result.task_ids)

    # Worktree dirs exist under <repo>/.claude/worktrees/...
    repo = patch_repo_roots / "demo"
    for wp in result.worktree_paths:
        assert Path(wp).exists()
        assert str(repo) in wp

    # Tasks are tagged with the group id and the source.
    rows = store.list_tasks(status="open", tag=f"fanout:{result.group_id}")
    assert len(rows) == 2
    for r in rows:
        assert "src:text" in r["tags"]
        full = store.get_task(r["id"])
        assert full["worktree"]
        spec = json.loads(full["worktree"])
        assert spec["project"] == "demo"
        assert spec["branch"].startswith("cc/")


def test_fanout_requires_project(patch_repo_roots, tmp_path: Path):
    store = SkillStore(db_path=tmp_path / "store.db")
    with pytest.raises(ValueError, match="project"):
        fanout("text", filter="- a", project="", store=store, use_llm=False)


def test_fanout_empty_source_returns_no_directive(patch_repo_roots, tmp_path: Path):
    store = SkillStore(db_path=tmp_path / "store.db")
    result = fanout(
        "text",
        filter="not a bullet list",
        project="demo",
        store=store,
        use_llm=False,
    )
    assert result.task_ids == []
    assert "no tasks created" in result.directive
