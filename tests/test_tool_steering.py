"""Tests for G2 — proactive, deterministic tool steering.

The hint nudges Claude toward codegraph queries + compact shell output, but
ONLY when the prompt shows search intent inside a codegraph-indexed repo, so
it never adds per-prompt noise (which would defeat the token-reduction goal).
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest  # noqa: E402


@pytest.fixture()
def indexed_repo(tmp_path):
    (tmp_path / ".codegraph").mkdir()
    return str(tmp_path)


def test_fires_on_search_intent_in_indexed_repo(indexed_repo):
    from skill_hub import cli

    hint = cli._tool_steering_hint("where is the SkillStore class defined?", cwd=indexed_repo)
    assert hint is not None
    assert "codegraph_search" in hint


def test_fires_on_grep_keyword(indexed_repo):
    from skill_hub import cli

    assert cli._tool_steering_hint("grep for register_plugin", cwd=indexed_repo) is not None


def test_silent_without_search_intent(indexed_repo):
    from skill_hub import cli

    # No search/grep verb -> no steering, even in an indexed repo.
    assert cli._tool_steering_hint("please refactor the login flow", cwd=indexed_repo) is None


def test_silent_when_not_indexed(tmp_path):
    from skill_hub import cli

    # Search intent but no .codegraph index -> nothing to steer toward.
    assert cli._tool_steering_hint("where is the config loaded?", cwd=str(tmp_path)) is None


def test_silent_when_flag_disabled(indexed_repo, monkeypatch):
    from skill_hub import cli, config

    real_get = config.get
    monkeypatch.setattr(
        config, "get",
        lambda key: False if key == "tool_steering_enabled" else real_get(key),
    )
    assert cli._tool_steering_hint("where is the store?", cwd=indexed_repo) is None


def test_hook_classify_and_execute_forwards_session_cwd(tmp_path, monkeypatch):
    """Production entry point: hook_classify_and_execute() must forward its
    own ``cwd`` argument (the session cwd carried by the hook payload) to
    _tool_steering_hint -- NOT os.getcwd(), which is meaningless when this
    code runs detached from the session's working directory.
    """
    from skill_hub import cli, config as cfg

    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(cli, "_local_mode", False)

    captured: dict = {}

    def _spy(message, cwd=None):
        captured["cwd"] = cwd
        return None

    monkeypatch.setattr(cli, "_tool_steering_hint", _spy)
    # Stub out the LLM-backed context stage so this stays deterministic and
    # never touches Ollama or the real skill store.
    monkeypatch.setattr(cli, "_dynamic_context_stage", lambda *a, **k: None)

    session_cwd = str(tmp_path / "project")
    cli.hook_classify_and_execute("hi", session_id="s1", cwd=session_cwd)

    assert "cwd" in captured, "_tool_steering_hint was never called"
    assert captured["cwd"] == session_cwd
