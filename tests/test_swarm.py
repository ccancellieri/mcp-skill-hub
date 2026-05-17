"""Tests for skill_hub.swarm -- swarm-lite multi-Claude dispatcher.

Covers the issue m4-#20 acceptance:

* a 2-claim swarm with a dummy ``claude`` binary works
* per-claim log capture is verified
* reap correctly transitions claim status (running -> done/failed)

The dummy binary is a tiny POSIX shell script the test writes into ``tmp_path``
and points the swarm at via ``claude_binary=...``. This keeps the test
hermetic — no real Claude Code install required.
"""
from __future__ import annotations

import stat
from pathlib import Path

import pytest

from skill_hub import swarm


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_dummy_claude(path: Path, *, exit_code: int = 0, sleep_ms: int = 0) -> Path:
    """Write a POSIX shell script that mimics ``claude --print``.

    It echoes its args (so the test can assert prompt delivery), echoes its
    cwd (so the test can assert the worktree was picked up), optionally
    sleeps, then exits with ``exit_code``.
    """
    script = "#!/bin/sh\n" "echo \"cwd=$PWD\"\n" "echo \"args=$*\"\n"
    if sleep_ms:
        script += f"sleep {sleep_ms / 1000:.3f}\n"
    script += f"exit {exit_code}\n"
    path.write_text(script)
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture()
def dummy_claude(tmp_path: Path) -> Path:
    """A dummy `claude` binary that exits 0 immediately."""
    return _write_dummy_claude(tmp_path / "claude", exit_code=0)


@pytest.fixture()
def worktrees(tmp_path: Path) -> tuple[Path, Path]:
    """Two empty 'worktree' directories (real dirs are enough for the test)."""
    a = tmp_path / "wt-a"
    b = tmp_path / "wt-b"
    a.mkdir()
    b.mkdir()
    return a, b


@pytest.fixture(autouse=True)
def swarm_log_root(tmp_path: Path, monkeypatch) -> Path:
    """Redirect the default swarm log root to tmp so tests don't leak."""
    root = tmp_path / "swarm-logs"
    monkeypatch.setattr(swarm, "SWARM_LOG_DIR", root)
    return root


# ---------------------------------------------------------------------------
# Acceptance: 2-claim swarm with a dummy claude binary
# ---------------------------------------------------------------------------


def test_swarm_launch_two_claims(dummy_claude: Path, worktrees: tuple[Path, Path]):
    wt_a, wt_b = worktrees
    claims = [
        swarm.Claim(claim_id="c-1", worktree_path=str(wt_a), task_summary="fix bug 1"),
        swarm.Claim(claim_id="c-2", worktree_path=str(wt_b), task_summary="fix bug 2"),
    ]
    transitions: list[tuple[str, str]] = []

    def cb(claim_id: str, status: str, **_) -> None:
        transitions.append((claim_id, status))

    handles = swarm.swarm_launch(
        claims,
        claude_binary=str(dummy_claude),
        status_callback=cb,
    )

    assert len(handles) == 2
    assert {h.claim_id for h in handles} == {"c-1", "c-2"}
    for h in handles:
        assert h.status == "running"
        assert h.pid > 0
        assert Path(h.log_path).parent.is_dir()
    # status_callback fired with "running" for each.
    assert sorted(transitions) == [("c-1", "running"), ("c-2", "running")]

    # Reap with a small timeout so the dummy children all exit cleanly.
    reap_transitions: list[tuple[str, str, int]] = []

    def reap_cb(claim_id: str, status: str, returncode: int = -1, **_) -> None:
        reap_transitions.append((claim_id, status, returncode))

    swarm.swarm_reap(handles, timeout=5.0, status_callback=reap_cb)

    for h in handles:
        assert h.status == "done", h
        assert h.returncode == 0, h
        assert h.finished_at is not None
    assert sorted(reap_transitions) == [
        ("c-1", "done", 0),
        ("c-2", "done", 0),
    ]


def test_log_capture_records_cwd_and_args(
    dummy_claude: Path, worktrees: tuple[Path, Path]
):
    wt_a, _ = worktrees
    handles = swarm.swarm_launch(
        [swarm.Claim(claim_id="logcap", worktree_path=str(wt_a),
                     task_summary="check logging")],
        claude_binary=str(dummy_claude),
    )
    swarm.swarm_reap(handles, timeout=5.0)
    log = Path(handles[0].log_path).read_text()
    # Dummy binary prints cwd and args -- both must land in the log.
    assert f"cwd={wt_a.resolve()}" in log or f"cwd={wt_a}" in log
    assert "--print" in log
    assert "check logging" in log  # task_summary made it into the prompt


def test_reap_marks_nonzero_as_failed(
    tmp_path: Path, worktrees: tuple[Path, Path]
):
    wt_a, _ = worktrees
    failing = _write_dummy_claude(tmp_path / "claude-fail", exit_code=2)
    handles = swarm.swarm_launch(
        [swarm.Claim(claim_id="bad", worktree_path=str(wt_a))],
        claude_binary=str(failing),
    )
    swarm.swarm_reap(handles, timeout=5.0)
    assert handles[0].status == "failed"
    assert handles[0].returncode == 2


def test_reap_status_callback_only_on_transition(
    tmp_path: Path, worktrees: tuple[Path, Path]
):
    wt_a, _ = worktrees
    slow = _write_dummy_claude(tmp_path / "claude-slow", exit_code=0, sleep_ms=300)
    handles = swarm.swarm_launch(
        [swarm.Claim(claim_id="slow", worktree_path=str(wt_a))],
        claude_binary=str(slow),
    )
    fired: list[tuple[str, str]] = []

    def cb(claim_id: str, status: str, **_) -> None:
        fired.append((claim_id, status))

    # First sweep should *not* mark it done yet (process is still sleeping).
    swarm.swarm_reap(handles, timeout=0.0, status_callback=cb)
    assert fired == []
    assert handles[0].status == "running"

    # Now wait it out.
    swarm.swarm_reap(handles, timeout=5.0, status_callback=cb)
    assert fired == [("slow", "done")]
    assert handles[0].status == "done"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_swarm_launch_rejects_empty(dummy_claude: Path):
    with pytest.raises(swarm.SwarmError, match="empty"):
        swarm.swarm_launch([], claude_binary=str(dummy_claude))


def test_swarm_launch_rejects_duplicate_claim_ids(
    dummy_claude: Path, worktrees: tuple[Path, Path]
):
    wt_a, wt_b = worktrees
    with pytest.raises(swarm.SwarmError, match="duplicate"):
        swarm.swarm_launch(
            [
                swarm.Claim(claim_id="same", worktree_path=str(wt_a)),
                swarm.Claim(claim_id="same", worktree_path=str(wt_b)),
            ],
            claude_binary=str(dummy_claude),
        )


def test_swarm_launch_rejects_missing_worktree(dummy_claude: Path, tmp_path: Path):
    with pytest.raises(swarm.SwarmError, match="worktree_path does not exist"):
        swarm.swarm_launch(
            [swarm.Claim(claim_id="x", worktree_path=str(tmp_path / "nope"))],
            claude_binary=str(dummy_claude),
        )


def test_swarm_launch_accepts_dict_claims(
    dummy_claude: Path, worktrees: tuple[Path, Path]
):
    wt_a, _ = worktrees
    handles = swarm.swarm_launch(
        [{"claim_id": "dictc", "worktree_path": str(wt_a), "task_summary": "x"}],
        claude_binary=str(dummy_claude),
    )
    swarm.swarm_reap(handles, timeout=5.0)
    assert handles[0].claim_id == "dictc"
    assert handles[0].status == "done"


def test_swarm_launch_dict_missing_keys_errors(dummy_claude: Path):
    with pytest.raises(swarm.SwarmError, match="claim_id"):
        swarm.swarm_launch(
            [{"worktree_path": "/tmp"}],
            claude_binary=str(dummy_claude),
        )


def test_resolve_claude_binary_raises_when_missing(monkeypatch):
    monkeypatch.setattr(swarm.shutil, "which", lambda _name: None)
    with pytest.raises(swarm.SwarmError, match="not found on PATH"):
        swarm._resolve_claude_binary()


# ---------------------------------------------------------------------------
# Group_id + log dir layout
# ---------------------------------------------------------------------------


def test_group_id_creates_subdir(
    dummy_claude: Path, worktrees: tuple[Path, Path], swarm_log_root: Path
):
    wt_a, _ = worktrees
    handles = swarm.swarm_launch(
        [swarm.Claim(claim_id="g1", worktree_path=str(wt_a))],
        claude_binary=str(dummy_claude),
        group_id="my-batch",
    )
    swarm.swarm_reap(handles, timeout=5.0)
    expected = swarm_log_root / "my-batch" / "g1.log"
    assert handles[0].log_path == str(expected)
    assert expected.exists()


def test_swarm_handle_dict_roundtrip(
    dummy_claude: Path, worktrees: tuple[Path, Path]
):
    """server.py round-trips handles through dict; verify that survives."""
    wt_a, _ = worktrees
    handles = swarm.swarm_launch(
        [swarm.Claim(claim_id="rt", worktree_path=str(wt_a))],
        claude_binary=str(dummy_claude),
    )
    # Simulate persistence: handle -> dict -> handle.
    as_dict = handles[0].to_dict()
    restored = swarm.SwarmHandle(**as_dict)
    assert restored.claim_id == "rt"
    assert restored.pid == handles[0].pid
    assert restored.status == "running"
    # Reap the original handle (we are the child's parent).
    swarm.swarm_reap(handles, timeout=5.0)
    assert handles[0].status == "done"
