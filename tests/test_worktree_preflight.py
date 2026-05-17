"""Tests for skill_hub.worktree_preflight (M3-1 collision check)."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest

from skill_hub import worktree as wt
from skill_hub import worktree_preflight as wp


# ---------------------------------------------------------------------------
# Fixtures (mirror tests/test_worktree.py)
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_repo_root(tmp_path: Path) -> Path:
    root = tmp_path / "code"
    root.mkdir()
    return root


@pytest.fixture()
def tmp_repo(tmp_repo_root: Path) -> Path:
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
    from skill_hub import config as cfg
    monkeypatch.setattr(
        cfg, "load_config", lambda: {"worktree": {
            "repo_roots": [str(tmp_repo_root)],
            "default_mode": "background",
        }},
    )
    # Default: no `gh` available, so tests don't depend on the user's env.
    monkeypatch.setattr(wp, "_gh_available", lambda: False)
    yield


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def test_normalize_issue_number_accepts_int_and_hash_string():
    assert wp._normalize_issue_number(15) == 15
    assert wp._normalize_issue_number("15") == 15
    assert wp._normalize_issue_number("#15") == 15


@pytest.mark.parametrize("bad", [0, -1, "abc", "", "#"])
def test_normalize_issue_number_rejects_garbage(bad):
    with pytest.raises(ValueError):
        wp._normalize_issue_number(bad)


# ---------------------------------------------------------------------------
# Clean state
# ---------------------------------------------------------------------------

def test_preflight_clean_state_is_safe(tmp_repo: Path):
    res = wp.preflight(15, project="sample")
    assert res.safe is True
    assert res.worktrees == []
    assert res.branches == []
    assert res.pull_requests == []
    assert res.issue_prefix == "issue-15-"
    assert res.branch_prefix == "cc/issue-15-"
    assert res.repo_path == str(tmp_repo)
    # gh-skipped warning is fine; should not be treated as a collision.
    assert any("gh CLI not installed" in w for w in res.warnings)


def test_format_result_safe(tmp_repo: Path):
    res = wp.preflight(42, project="sample")
    out = wp.format_result(res)
    assert "safe to start" in out
    assert "worktree_preflight #42" in out
    assert "(sample)" in out


# ---------------------------------------------------------------------------
# Worktree collision
# ---------------------------------------------------------------------------

def test_preflight_detects_existing_worktree(tmp_repo: Path):
    # Create a worktree whose name matches the issue prefix.
    spec = wt.ensure_worktree("sample", "issue-15-foo", mode="background")
    res = wp.preflight(15, project="sample")
    assert res.safe is False
    assert spec.worktree_path in res.worktrees
    # Branch is also there (created alongside the worktree).
    assert any(b.startswith("cc/issue-15-") for b in res.branches)
    out = wp.format_result(res)
    assert "collision detected" in out
    assert spec.worktree_path in out


def test_preflight_unrelated_worktree_does_not_collide(tmp_repo: Path):
    wt.ensure_worktree("sample", "issue-99-bar", mode="background")
    res = wp.preflight(15, project="sample")
    assert res.safe is True


# ---------------------------------------------------------------------------
# Branch-only collision (e.g. worktree was removed but branch kept)
# ---------------------------------------------------------------------------

def test_preflight_detects_branch_without_worktree(tmp_repo: Path):
    # Create the branch directly, no worktree.
    subprocess.run(
        ["git", "-C", str(tmp_repo), "branch", "cc/issue-15-leftover"],
        check=True,
    )
    res = wp.preflight(15, project="sample")
    assert res.safe is False
    assert res.worktrees == []
    assert "cc/issue-15-leftover" in res.branches


# ---------------------------------------------------------------------------
# Open PR collision (mocked gh)
# ---------------------------------------------------------------------------

def test_preflight_detects_open_pr(tmp_repo: Path, monkeypatch):
    monkeypatch.setattr(wp, "_gh_available", lambda: True)

    issue_json = json.dumps({"title": "the issue", "state": "OPEN"})
    pr_json = json.dumps([{
        "number": 7,
        "title": "WIP: fix the thing",
        "url": "https://x/pull/7",
        "headRefName": "cc/issue-15-fix-the-thing",
    }])

    def _fake_run_gh(args):
        if "issue" in args and "view" in args:
            return 0, issue_json, ""
        if "pr" in args and "list" in args:
            return 0, pr_json, ""
        return 1, "", "unexpected"

    monkeypatch.setattr(wp, "_run_gh", _fake_run_gh)

    res = wp.preflight(15, project="sample", repo="owner/name")
    assert res.safe is False
    assert res.issue_title == "the issue"
    assert res.issue_state == "OPEN"
    assert len(res.pull_requests) == 1
    pr = res.pull_requests[0]
    assert pr["number"] == 7
    assert pr["head"] == "cc/issue-15-fix-the-thing"


def test_preflight_filters_prs_with_non_matching_head(tmp_repo: Path, monkeypatch):
    # gh's search is fuzzy on `head:` — preflight() must double-check the
    # prefix locally to avoid false positives.
    monkeypatch.setattr(wp, "_gh_available", lambda: True)
    pr_json = json.dumps([
        {"number": 1, "title": "ok",   "url": "u1",
         "headRefName": "cc/issue-15-real"},
        {"number": 2, "title": "noise", "url": "u2",
         "headRefName": "cc/issue-150-other"},  # different issue, same prefix-like
    ])

    def _fake_run_gh(args):
        if "issue" in args and "view" in args:
            return 0, json.dumps({"title": "", "state": ""}), ""
        if "pr" in args and "list" in args:
            return 0, pr_json, ""
        return 1, "", ""

    monkeypatch.setattr(wp, "_run_gh", _fake_run_gh)
    res = wp.preflight(15, project="sample", repo="owner/name")
    heads = [pr["head"] for pr in res.pull_requests]
    assert heads == ["cc/issue-15-real"]


def test_preflight_gh_unauthenticated_is_warning_not_collision(
    tmp_repo: Path, monkeypatch,
):
    monkeypatch.setattr(wp, "_gh_available", lambda: True)

    def _fake_run_gh(args):
        return 4, "", "error: You are not authenticated. Run gh auth login."

    monkeypatch.setattr(wp, "_run_gh", _fake_run_gh)
    res = wp.preflight(15, project="sample", repo="owner/name")
    # No collisions and no PRs, but warnings explain why.
    assert res.safe is True
    assert any("not authenticated" in w for w in res.warnings)


def test_preflight_gh_timeout_does_not_fail(tmp_repo: Path, monkeypatch):
    monkeypatch.setattr(wp, "_gh_available", lambda: True)

    def _fake_run_gh(args):
        return 124, "", f"gh timed out after {wp._GH_TIMEOUT_S}s"

    monkeypatch.setattr(wp, "_run_gh", _fake_run_gh)
    res = wp.preflight(15, project="sample", repo="owner/name")
    assert res.safe is True
    assert any("timed out" in w for w in res.warnings)


# ---------------------------------------------------------------------------
# Performance — sub-second on a fresh repo
# ---------------------------------------------------------------------------

def test_preflight_subsecond_clean_state(tmp_repo: Path):
    # _gh_available is patched to False by the autouse fixture so we measure
    # only the local git calls.
    t0 = time.monotonic()
    res = wp.preflight(15, project="sample")
    elapsed = time.monotonic() - t0
    assert res.safe is True
    # Generous margin; on a clean repo this is typically well under 100 ms.
    assert elapsed < 1.0, f"preflight took {elapsed:.3f}s, expected < 1s"


# ---------------------------------------------------------------------------
# to_dict serialization
# ---------------------------------------------------------------------------

def test_result_to_dict_includes_safe_flag(tmp_repo: Path):
    res = wp.preflight(15, project="sample")
    d = res.to_dict()
    assert d["safe"] is True
    assert d["issue_number"] == 15
    assert d["branch_prefix"] == "cc/issue-15-"
