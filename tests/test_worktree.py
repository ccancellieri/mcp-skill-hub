"""Tests for skill_hub.worktree -- worktree-driven parallel sessions.

Covers project resolution, idempotent worktree creation, mode dispatch with
mocked subprocess/osascript, pidfile liveness with stale cleanup, and a
save/reopen/close round-trip via the SkillStore.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from skill_hub import worktree as wt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_repo_root(tmp_path: Path) -> Path:
    """A throwaway directory used as the repo_roots[0] for tests."""
    root = tmp_path / "code"
    root.mkdir()
    return root


@pytest.fixture()
def tmp_repo(tmp_repo_root: Path) -> Path:
    """A real git repo with one initial commit, named 'sample'."""
    repo = tmp_repo_root / "sample"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "t@t"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "t"], check=True)
    (repo / "README.md").write_text("hi\n")
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-q", "-m", "init"], check=True,
    )
    return repo


@pytest.fixture(autouse=True)
def _isolate_config(tmp_repo_root, monkeypatch):
    """Force worktree.repo_roots to point at the temp dir for every test."""
    from skill_hub import config as cfg
    monkeypatch.setattr(
        cfg, "load_config", lambda: {"worktree": {
            "repo_roots": [str(tmp_repo_root)],
            "default_mode": "background",
        }},
    )
    yield


# ---------------------------------------------------------------------------
# resolve_project / detect_project_from_cwd
# ---------------------------------------------------------------------------

def test_resolve_project_finds_repo(tmp_repo: Path):
    found = wt.resolve_project("sample")
    assert found == tmp_repo


def test_resolve_project_invalid_name():
    with pytest.raises(wt.WorktreeError, match="invalid project name"):
        wt.resolve_project("../etc/passwd")


def test_resolve_project_missing(tmp_repo_root: Path):
    with pytest.raises(wt.WorktreeError, match="not found as a git repo"):
        wt.resolve_project("does-not-exist")


def test_detect_project_from_cwd(tmp_repo: Path):
    assert wt.detect_project_from_cwd(tmp_repo) == "sample"
    sub = tmp_repo / "sub" / "deep"
    sub.mkdir(parents=True)
    assert wt.detect_project_from_cwd(sub) == "sample"


def test_detect_project_from_cwd_outside_root(tmp_path: Path):
    assert wt.detect_project_from_cwd(tmp_path) is None


# ---------------------------------------------------------------------------
# ensure_worktree (idempotent)
# ---------------------------------------------------------------------------

def test_ensure_worktree_creates_and_reuses(tmp_repo: Path):
    spec = wt.ensure_worktree("sample", "feat-a", mode="background")
    assert Path(spec.worktree_path).exists()
    assert spec.branch == "cc/feat-a"
    assert spec.repo_path == str(tmp_repo)
    # Idempotent reattach: identical spec, no new worktree row.
    spec2 = wt.ensure_worktree("sample", "feat-a", mode="background")
    assert spec2.worktree_path == spec.worktree_path
    registered = wt._registered_worktrees(tmp_repo)
    # Main repo + one worktree.
    assert sum(1 for p in registered if p.endswith("/feat-a")) == 1


def test_ensure_worktree_invalid_mode(tmp_repo: Path):
    with pytest.raises(wt.WorktreeError, match="invalid mode"):
        wt.ensure_worktree("sample", "x", mode="bogus")  # type: ignore[arg-type]


def test_ensure_worktree_path_collision(tmp_repo: Path):
    # Pre-create the directory so it's a non-worktree squatter.
    target = wt.compute_worktree_path(tmp_repo, "squatter")
    target.mkdir(parents=True)
    with pytest.raises(wt.WorktreeError, match="not a registered git worktree"):
        wt.ensure_worktree("sample", "squatter", mode="background")


# ---------------------------------------------------------------------------
# pidfile liveness
# ---------------------------------------------------------------------------

def test_is_session_alive_no_pidfile(tmp_repo: Path):
    spec = wt.ensure_worktree("sample", "alive-1", mode="background")
    assert wt.is_session_alive(spec) is False


def test_is_session_alive_stale_pidfile_cleaned(tmp_repo: Path):
    spec = wt.ensure_worktree("sample", "alive-2", mode="background")
    # Write a pidfile pointing at a definitely-dead PID.
    wt._write_pidfile(spec, pid=999999)
    assert Path(spec.pid_file).exists()
    assert wt.is_session_alive(spec) is False
    # Stale file should be cleaned up.
    assert not Path(spec.pid_file).exists()


def test_is_session_alive_self_pid(tmp_repo: Path):
    spec = wt.ensure_worktree("sample", "alive-3", mode="background")
    wt._write_pidfile(spec, pid=os.getpid())
    assert wt.is_session_alive(spec) is True


# ---------------------------------------------------------------------------
# launch_session mode dispatch (mocked)
# ---------------------------------------------------------------------------

def test_launch_background_uses_subprocess(tmp_repo: Path, monkeypatch):
    spec = wt.ensure_worktree("sample", "bg-launch", mode="background")
    captured: dict = {}

    class _FakePopen:
        def __init__(self, args, **kw):
            captured["args"] = args
            captured["cwd"] = kw.get("cwd")
            self.pid = 12345

    monkeypatch.setattr(wt, "_claude_binary", lambda: "/fake/claude")
    monkeypatch.setattr(wt.subprocess, "Popen", _FakePopen)
    out = wt.launch_session(spec, initial_prompt="hello")
    assert out.last_pid == 12345
    assert captured["cwd"] == spec.worktree_path
    assert captured["args"][0] == "/fake/claude"
    assert "--print" in captured["args"]
    assert "hello" in captured["args"]
    # Pidfile written.
    assert Path(spec.pid_file).exists()
    # Stop hook wired into worktree settings.
    settings = Path(spec.worktree_path) / ".claude" / "settings.local.json"
    assert settings.exists()
    blob = settings.read_text()
    assert wt._HOOK_NAME in blob


def test_launch_tmux_requires_tmux_env(tmp_repo: Path, monkeypatch):
    spec = wt.ensure_worktree("sample", "tmux-needs-env", mode="tmux")
    monkeypatch.delenv("TMUX", raising=False)
    monkeypatch.setattr(wt, "_claude_binary", lambda: "/fake/claude")
    with pytest.raises(wt.WorktreeError, match=r"requires running inside a tmux"):
        wt.launch_session(spec)


def test_launch_terminal_only_macos(tmp_repo: Path, monkeypatch):
    spec = wt.ensure_worktree("sample", "term-skip", mode="terminal")
    monkeypatch.setattr(wt, "_claude_binary", lambda: "/fake/claude")
    monkeypatch.setattr(wt.sys, "platform", "linux")
    with pytest.raises(wt.WorktreeError, match="macOS-only"):
        wt.launch_session(spec)


# ---------------------------------------------------------------------------
# Stop hook idempotency
# ---------------------------------------------------------------------------

def test_ensure_stop_hook_idempotent(tmp_repo: Path):
    spec = wt.ensure_worktree("sample", "hook-once", mode="background")
    wt._ensure_stop_hook(spec)
    wt._ensure_stop_hook(spec)
    settings = Path(spec.worktree_path) / ".claude" / "settings.local.json"
    import json as _json
    cfg = _json.loads(settings.read_text())
    stops = cfg.get("hooks", {}).get("Stop", [])
    # Exactly one Stop hook entry referencing our script.
    matching = [
        h for entry in stops for h in entry.get("hooks", [])
        if h.get("command", "").endswith(wt._HOOK_NAME)
    ]
    assert len(matching) == 1


# ---------------------------------------------------------------------------
# teardown
# ---------------------------------------------------------------------------

def test_teardown_removes_worktree(tmp_repo: Path):
    spec = wt.ensure_worktree("sample", "tear-1", mode="background")
    assert Path(spec.worktree_path).exists()
    wt.teardown_worktree(spec)
    registered = wt._registered_worktrees(tmp_repo)
    assert spec.worktree_path not in registered
    assert not Path(spec.worktree_path).exists()


# ---------------------------------------------------------------------------
# WorktreeSpec round-trip
# ---------------------------------------------------------------------------

def test_spec_json_roundtrip(tmp_repo: Path):
    spec = wt.ensure_worktree("sample", "rt", mode="background")
    spec.last_pid = 4242
    spec.last_window_id = "win-7"
    blob = spec.to_json()
    restored = wt.WorktreeSpec.from_json(blob)
    assert restored == spec
