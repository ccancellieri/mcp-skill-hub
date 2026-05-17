"""Unit tests for skill_hub.fanout.sources."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SRC = Path(__file__).resolve().parent.parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.fanout.sources import (
    GitHubSource,
    Issue,
    TextSource,
    get_source,
)


# ---------------------------------------------------------------------------
# TextSource
# ---------------------------------------------------------------------------

def test_text_source_parses_bullets():
    src = TextSource()
    issues = src.fetch(filter="- one\n- two\n- three")
    assert [i.title for i in issues] == ["one", "two", "three"]
    assert all(i.source == "text" for i in issues)
    assert {i.id for i in issues} == {"text:0001", "text:0002", "text:0003"}


def test_text_source_parses_numbered():
    src = TextSource()
    issues = src.fetch(filter="1. alpha\n2) beta")
    assert [i.title for i in issues] == ["alpha", "beta"]


def test_text_source_continuation_lines_become_body():
    src = TextSource()
    issues = src.fetch(filter="- first\n  more detail\n- second")
    assert len(issues) == 2
    assert issues[0].title == "first"
    assert "more detail" in issues[0].body
    assert issues[1].body == ""


def test_text_source_empty_returns_empty():
    assert TextSource().fetch(filter="") == []
    assert TextSource().fetch(filter="not a bullet\njust text") == []


def test_text_source_limit():
    src = TextSource()
    issues = src.fetch(filter="- a\n- b\n- c\n- d", limit=2)
    assert len(issues) == 2


# ---------------------------------------------------------------------------
# GitHubSource (mocked subprocess)
# ---------------------------------------------------------------------------

_GH_JSON = json.dumps([
    {
        "number": 11,
        "title": "Add pagination",
        "body": "We need cursor pagination on /items.",
        "labels": [{"name": "feature"}, {"name": "api"}],
        "url": "https://github.com/x/y/issues/11",
        "state": "OPEN",
    },
    {
        "number": 22,
        "title": "Race in worker pool",
        "body": "Two workers grab same job.",
        "labels": [{"name": "bug"}],
        "url": "https://github.com/x/y/issues/22",
        "state": "OPEN",
    },
])


class _DummyProc:
    def __init__(self, stdout: str, returncode: int = 0, stderr: str = ""):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_github_source_parses_json():
    with patch("skill_hub.fanout.sources.shutil.which", return_value="/usr/bin/gh"), \
         patch("skill_hub.fanout.sources.subprocess.run",
               return_value=_DummyProc(_GH_JSON)) as run:
        issues = GitHubSource().fetch(filter="is:open label:bug", limit=5,
                                      repo="x/y")
    args = run.call_args[0][0]
    assert args[:3] == ["gh", "issue", "list"]
    assert "--search" in args and "--limit" in args and "--repo" in args
    assert [i.id for i in issues] == ["gh:11", "gh:22"]
    assert issues[0].labels == ["feature", "api"]
    assert issues[1].title == "Race in worker pool"


def test_github_source_raises_without_gh_cli():
    with patch("skill_hub.fanout.sources.shutil.which", return_value=None):
        with pytest.raises(RuntimeError, match="gh"):
            GitHubSource().fetch()


def test_github_source_auth_error_friendly():
    with patch("skill_hub.fanout.sources.shutil.which", return_value="/usr/bin/gh"), \
         patch("skill_hub.fanout.sources.subprocess.run",
               return_value=_DummyProc("", returncode=1,
                                       stderr="gh auth status: not authenticated")):
        with pytest.raises(RuntimeError, match="not authenticated"):
            GitHubSource().fetch()


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_get_source_known():
    assert isinstance(get_source("text"), TextSource)
    assert isinstance(get_source("gh"), GitHubSource)
    assert isinstance(get_source("github"), GitHubSource)


def test_get_source_unknown_raises():
    with pytest.raises(ValueError, match="unknown fanout source"):
        get_source("nope")
