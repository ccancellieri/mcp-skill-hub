"""M1 #11 — worktree-aware tasks.

Tests that save_task auto-captures branch + worktree top-level via
``git rev-parse``, that list_tasks exposes a ``worktree_current`` filter,
and that the post-merge backing CLI closes only tasks whose branch is
truly gone.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore
    db_path = tmp_path / "skill_hub.db"
    monkeypatch.setattr("skill_hub.store.DB_PATH", db_path)
    s = SkillStore(db_path=db_path)
    yield s
    s.close()


@pytest.fixture
def git_repo(tmp_path):
    """Initialise a tiny git repo with one commit; return its top-level path."""
    repo = tmp_path / "demo-repo"
    repo.mkdir()
    def _run(*args: str) -> None:
        subprocess.run(["git", "-C", str(repo), *args],
                       check=True, capture_output=True)
    _run("init", "-q", "-b", "main")
    _run("config", "user.email", "t@example.com")
    _run("config", "user.name", "Test")
    (repo / "README.md").write_text("seed\n")
    _run("add", "README.md")
    _run("commit", "-q", "-m", "seed")
    return repo


# ---------------------------------------------------------------------------
# worktree helpers
# ---------------------------------------------------------------------------

def test_git_toplevel_in_repo(git_repo):
    from skill_hub import worktree as _wt
    top = _wt.git_toplevel(str(git_repo))
    assert top is not None
    assert Path(top).resolve() == git_repo.resolve()


def test_git_toplevel_outside_repo(tmp_path):
    from skill_hub import worktree as _wt
    # tmp_path itself is not a git repo on macOS/Linux test runners.
    bare = tmp_path / "no-git"
    bare.mkdir()
    assert _wt.git_toplevel(str(bare)) is None


def test_current_branch_main(git_repo):
    from skill_hub import worktree as _wt
    assert _wt.current_branch(str(git_repo)) == "main"


def test_current_branch_after_checkout(git_repo):
    from skill_hub import worktree as _wt
    subprocess.run(["git", "-C", str(git_repo), "checkout", "-q", "-b",
                    "feature/widget"], check=True, capture_output=True)
    assert _wt.current_branch(str(git_repo)) == "feature/widget"


def test_capture_worktree_context(git_repo):
    from skill_hub import worktree as _wt
    top, branch, cwd = _wt.capture_worktree_context(str(git_repo))
    assert Path(top).resolve() == git_repo.resolve()
    assert branch == "main"
    assert Path(cwd).resolve() == git_repo.resolve()


def test_capture_worktree_context_outside_repo(tmp_path):
    from skill_hub import worktree as _wt
    top, branch, cwd = _wt.capture_worktree_context(str(tmp_path))
    assert top == ""
    assert branch == ""
    # cwd is still resolved, even outside a repo.
    assert cwd != ""


# ---------------------------------------------------------------------------
# save_task auto-capture
# ---------------------------------------------------------------------------

def test_save_task_auto_captures_branch_and_worktree(monkeypatch, store, git_repo):
    """When cwd is inside a git repo, save_task records branch + top-level."""
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)
    monkeypatch.setattr(srv, "embed", lambda *_a, **_kw: [])

    # Switch the test repo onto a feature branch.
    subprocess.run(["git", "-C", str(git_repo), "checkout", "-q", "-b",
                    "feature/auto-capture"], check=True, capture_output=True)

    srv.save_task(title="Capture me", summary="body",
                  cwd=str(git_repo))
    row = store._conn.execute(
        "SELECT cwd, branch FROM tasks WHERE title = ?", ("Capture me",)
    ).fetchone()
    assert Path(row["cwd"]).resolve() == git_repo.resolve()
    assert row["branch"] == "feature/auto-capture"


def test_save_task_outside_repo_falls_back_to_cwd(monkeypatch, store, tmp_path):
    """Outside any repo, branch is blank and cwd is whatever caller passed."""
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)
    monkeypatch.setattr(srv, "embed", lambda *_a, **_kw: [])

    bare = tmp_path / "lonely"
    bare.mkdir()
    srv.save_task(title="No repo", summary="body", cwd=str(bare))
    row = store._conn.execute(
        "SELECT cwd, branch FROM tasks WHERE title = ?", ("No repo",)
    ).fetchone()
    # cwd round-trips even without git; branch is empty (no NULL coercion to
    # avoid breaking the cwd+branch resumable lookup which uses IFNULL).
    assert row["cwd"] == str(bare)
    assert (row["branch"] or "") == ""


def test_save_task_empty_cwd_skips_inspection(monkeypatch, store):
    """No cwd passed -> no git inspection (daemon has no meaningful cwd)."""
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)
    monkeypatch.setattr(srv, "embed", lambda *_a, **_kw: [])

    srv.save_task(title="No cwd", summary="body")
    row = store._conn.execute(
        "SELECT cwd, branch FROM tasks WHERE title = ?", ("No cwd",)
    ).fetchone()
    assert (row["cwd"] or "") == ""
    assert (row["branch"] or "") == ""


# ---------------------------------------------------------------------------
# list_tasks(worktree_current=True)
# ---------------------------------------------------------------------------

def test_list_tasks_worktree_current_filter(monkeypatch, store, git_repo, tmp_path):
    """worktree_current=True only surfaces tasks whose cwd matches."""
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)
    monkeypatch.setattr(srv, "embed", lambda *_a, **_kw: [])

    # Task A: recorded inside git_repo.
    srv.save_task(title="In-repo task", summary="x",
                  cwd=str(git_repo))
    # Task B: outside the repo entirely.
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    srv.save_task(title="Outside task", summary="y",
                  cwd=str(elsewhere))

    out = srv.list_tasks(status="open", worktree_current=True,
                         cwd=str(git_repo))
    assert "In-repo task" in out
    assert "Outside task" not in out
    assert "worktree=" in out


def test_list_tasks_worktree_current_requires_cwd(monkeypatch, store):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    out = srv.list_tasks(status="open", worktree_current=True)
    assert "requires cwd=" in out


def test_list_tasks_worktree_current_outside_repo(monkeypatch, store, tmp_path):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)

    bare = tmp_path / "not-a-repo"
    bare.mkdir()
    out = srv.list_tasks(status="open", worktree_current=True,
                         cwd=str(bare))
    assert "not inside a git repository" in out


def test_list_tasks_worktree_current_empty_result(monkeypatch, store, git_repo):
    from skill_hub import server as srv
    monkeypatch.setattr(srv, "_store", store)
    monkeypatch.setattr(srv, "embed", lambda *_a, **_kw: [])

    out = srv.list_tasks(status="open", worktree_current=True,
                         cwd=str(git_repo))
    assert "No open tasks for current worktree" in out


# ---------------------------------------------------------------------------
# Store-level branch lookup (post-merge hook backing)
# ---------------------------------------------------------------------------

def test_find_open_tasks_by_branch(store):
    a = store.save_task(title="A", summary="x", vector=[],
                        branch="feature/foo")
    b = store.save_task(title="B", summary="y", vector=[],
                        branch="feature/bar")
    c = store.save_task(title="C", summary="z", vector=[],
                        branch="feature/foo")
    store.close_task(c, compact="done")

    rows = store.find_open_tasks_by_branch("feature/foo")
    assert {r["id"] for r in rows} == {a}

    rows = store.find_open_tasks_by_branch("feature/bar")
    assert {r["id"] for r in rows} == {b}

    rows = store.find_open_tasks_by_branch("does-not-exist")
    assert rows == []


def test_find_open_tasks_by_branch_repo_scope(store):
    a = store.save_task(title="A", summary="x", vector=[],
                        branch="feature/foo", repo="alpha")
    store.save_task(title="B", summary="y", vector=[],
                    branch="feature/foo", repo="beta")
    rows = store.find_open_tasks_by_branch("feature/foo", repo="alpha")
    assert {r["id"] for r in rows} == {a}


# ---------------------------------------------------------------------------
# CLI: close_tasks_for_branch
# ---------------------------------------------------------------------------

def test_cli_close_tasks_for_branch_when_branch_gone(
    monkeypatch, store, git_repo, capsys
):
    """Branch deleted -> tasks closed."""
    from skill_hub import cli as _cli
    monkeypatch.setattr(_cli, "SkillStore", lambda *_a, **_kw: store)
    # Prevent the singleton .close() from invalidating our shared store.
    monkeypatch.setattr(store, "close", lambda: None)

    # Create + delete a branch so the existence check fails.
    subprocess.run(["git", "-C", str(git_repo), "checkout", "-q", "-b",
                    "feature/gone"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(git_repo), "checkout", "-q", "main"],
                   check=True, capture_output=True)
    subprocess.run(["git", "-C", str(git_repo), "branch", "-D",
                    "feature/gone"], check=True, capture_output=True)

    tid = store.save_task(title="Doomed", summary="x", vector=[],
                          branch="feature/gone")
    monkeypatch.setattr(sys, "argv",
                        ["skill-hub-cli", "close_tasks_for_branch",
                         "feature/gone", "--cwd", str(git_repo)])
    _cli.main()
    out = capsys.readouterr().out
    assert "Closed 1 task" in out
    row = store._conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "closed"


def test_cli_close_tasks_for_branch_skips_when_branch_exists(
    monkeypatch, store, git_repo, capsys
):
    """Branch still exists -> hook is a no-op (idempotent)."""
    from skill_hub import cli as _cli
    monkeypatch.setattr(_cli, "SkillStore", lambda *_a, **_kw: store)
    monkeypatch.setattr(store, "close", lambda: None)

    subprocess.run(["git", "-C", str(git_repo), "checkout", "-q", "-b",
                    "feature/alive"], check=True, capture_output=True)

    tid = store.save_task(title="Alive", summary="x", vector=[],
                          branch="feature/alive")
    monkeypatch.setattr(sys, "argv",
                        ["skill-hub-cli", "close_tasks_for_branch",
                         "feature/alive", "--cwd", str(git_repo)])
    with pytest.raises(SystemExit) as exc:
        _cli.main()
    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "still exists" in out
    row = store._conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "open"


def test_cli_close_tasks_for_branch_force(
    monkeypatch, store, git_repo, capsys
):
    """--force bypasses the existence check (manual cleanup)."""
    from skill_hub import cli as _cli
    monkeypatch.setattr(_cli, "SkillStore", lambda *_a, **_kw: store)
    monkeypatch.setattr(store, "close", lambda: None)

    subprocess.run(["git", "-C", str(git_repo), "checkout", "-q", "-b",
                    "feature/alive"], check=True, capture_output=True)

    tid = store.save_task(title="Force", summary="x", vector=[],
                          branch="feature/alive")
    monkeypatch.setattr(sys, "argv",
                        ["skill-hub-cli", "close_tasks_for_branch",
                         "feature/alive", "--cwd", str(git_repo), "--force"])
    _cli.main()
    out = capsys.readouterr().out
    assert "Closed 1 task" in out
    row = store._conn.execute(
        "SELECT status FROM tasks WHERE id = ?", (tid,)
    ).fetchone()
    assert row["status"] == "closed"


# ---------------------------------------------------------------------------
# Hook shipped on disk
# ---------------------------------------------------------------------------

def test_post_merge_hook_exists_and_executable():
    hook = ROOT / "hooks" / "post-merge.sh"
    assert hook.exists(), f"missing {hook}"
    import os
    assert os.access(hook, os.X_OK), f"{hook} is not executable"
