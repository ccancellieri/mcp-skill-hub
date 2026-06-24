"""Tests for tool-usage activity logging + per-interception token savings.

Covers:
- ``activity_log._fmt_saved`` / ``log_pipeline_end(saved=...)`` — the ``<<``
  line surfaces an estimated token saving (``saved~1.8k`` / ``saved~420tok``).
- ``activity_log.append_line`` — direct file append used by hook subprocesses.
- PostToolUse observer helpers: ``_mcp_short``, ``_describe_tool`` (grep
  detection), ``_codegraph_indexed``, and ``_emit_tool_activity`` (TOOL line +
  codegraph HINT only when grep runs in a ``.codegraph/``-indexed repo).
"""
from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
HOOKS = Path(__file__).resolve().parent.parent / "hooks"
sys.path.insert(0, str(SRC))


@pytest.fixture()
def observer():
    """Load the PostToolUse observer hook module by path."""
    sys.path.insert(0, str(HOOKS))
    spec = importlib.util.spec_from_file_location(
        "post_tool_observer", HOOKS / "post_tool_observer.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── activity_log: token-saving formatting ──────────────────────────────────

def test_fmt_saved_thresholds():
    from skill_hub import activity_log as al
    assert al._fmt_saved(420) == "saved~420tok"
    assert al._fmt_saved(999) == "saved~999tok"
    assert al._fmt_saved(1000) == "saved~1.0k"
    assert al._fmt_saved(1800) == "saved~1.8k"


def test_pipeline_end_emits_saved():
    from skill_hub import activity_log as al
    logger = al.get_logger()
    captured: list[str] = []
    handler = logging.Handler()
    handler.emit = lambda rec: captured.append(rec.getMessage())  # type: ignore[assignment]
    logger.addHandler(handler)
    try:
        al.log_pipeline_end("LOCAL", reason="task:save_memory", saved=1800)
        al.log_pipeline_end("CLAUDE", reason="passthrough")  # saved omitted
    finally:
        logger.removeHandler(handler)
    assert any("saved~1.8k" in m for m in captured)
    # A pipeline end without a saving must not print a saved~ token.
    assert not any("passthrough" in m and "saved~" in m for m in captured)


def test_append_line_writes_to_file(tmp_path, monkeypatch):
    from skill_hub import activity_log as al
    log_file = tmp_path / "activity.log"
    monkeypatch.setattr(al, "LOG_DIR", tmp_path)
    monkeypatch.setattr(al, "LOG_FILE", log_file)
    al.append_line("TOOL  Bash    grep -rn foo src/")
    text = log_file.read_text()
    assert "TOOL  Bash    grep -rn foo src/" in text
    # HH:MM:SS prefix present.
    assert text.split("  ", 1)[0].count(":") == 2


# ── observer: tool description + grep detection ────────────────────────────

def test_mcp_short_strips_redundant_prefix(observer):
    assert observer._mcp_short("mcp__codegraph__codegraph_search") == "codegraph:search"
    assert observer._mcp_short("mcp__skill-hub__search_skills") == "skill-hub:search_skills"
    assert observer._mcp_short("Bash") == "Bash"


def test_describe_tool_grep_detection(observer):
    name, detail, greplike = observer._describe_tool("Bash", {"command": "grep -rn foo src/"})
    assert name == "Bash" and "grep" in detail and greplike is True
    # rg via pipe is also grep-like.
    _, _, piped = observer._describe_tool("Bash", {"command": "cat x | rg foo"})
    assert piped is True
    # A non-search Bash command is not grep-like.
    _, _, plain = observer._describe_tool("Bash", {"command": "ls -la"})
    assert plain is False
    # The built-in Grep tool is always grep-like.
    _, _, gtool = observer._describe_tool("Grep", {"pattern": "foo", "path": "src"})
    assert gtool is True
    # codegraph (the preferred path) is never flagged.
    cname, _, cgrep = observer._describe_tool(
        "mcp__codegraph__codegraph_search", {"query": "foo"}
    )
    assert cname == "codegraph:search" and cgrep is False


def test_codegraph_indexed_detects_marker(observer, tmp_path):
    (tmp_path / ".codegraph").mkdir()
    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)
    assert observer._codegraph_indexed(str(nested)) is True
    assert observer._codegraph_indexed(str(tmp_path)) is True


def test_codegraph_indexed_absent(observer, tmp_path):
    assert observer._codegraph_indexed(str(tmp_path)) is False


def test_emit_tool_activity_logs_and_hints(observer, tmp_path, monkeypatch):
    from skill_hub import activity_log as al
    log_file = tmp_path / "repo" / "activity.log"
    log_file.parent.mkdir(parents=True)
    monkeypatch.setattr(al, "LOG_DIR", log_file.parent)
    monkeypatch.setattr(al, "LOG_FILE", log_file)

    repo = tmp_path / "indexed"
    (repo / ".codegraph").mkdir(parents=True)
    cfg = {"log_tool_usage": True, "tool_usage_codegraph_hint": True}

    # grep inside an indexed repo → TOOL line + HINT.
    observer._emit_tool_activity("Bash", {"command": "grep -rn foo src/"}, str(repo), cfg)
    text = log_file.read_text()
    assert "TOOL  Bash" in text
    assert "prefer codegraph_search" in text

    # codegraph call → TOOL line, no HINT.
    log_file.write_text("")
    observer._emit_tool_activity(
        "mcp__codegraph__codegraph_search", {"query": "foo"}, str(repo), cfg
    )
    text = log_file.read_text()
    assert "TOOL  codegraph:search" in text
    assert "prefer codegraph_search" not in text


def test_codegraph_outcome_classification(observer):
    # Non-empty payload -> ok.
    ok = observer._codegraph_outcome({"content": "def foo(): ... 12 callers"}, "PostToolUse")
    assert ok == "ok"
    # Empty / "no results" -> empty (the signal that pushes agents to grep).
    assert observer._codegraph_outcome({"content": "No matches found"}, "PostToolUse") == "empty"
    assert observer._codegraph_outcome({"content": ""}, "PostToolUse") == "empty"
    # Error field or failure event -> fail.
    assert observer._codegraph_outcome({"is_error": True}, "PostToolUse") == "fail"
    assert observer._codegraph_outcome({"content": "x" * 50}, "PostToolUseFailure") == "fail"


def test_emit_tool_activity_tags_codegraph_outcome(observer, tmp_path, monkeypatch):
    from skill_hub import activity_log as al
    log_file = tmp_path / "activity.log"
    monkeypatch.setattr(al, "LOG_DIR", tmp_path)
    monkeypatch.setattr(al, "LOG_FILE", log_file)
    metered: list = []
    monkeypatch.setattr(observer, "_meter_codegraph",
                        lambda tool, outcome: metered.append((tool, outcome)))
    cfg = {"log_tool_usage": True}

    observer._emit_tool_activity(
        "mcp__codegraph__codegraph_search", {"query": "foo"}, str(tmp_path), cfg,
        {"content": "No results"}, "PostToolUse",
    )
    text = log_file.read_text()
    assert "TOOL  codegraph:search" in text and "[empty]" in text
    assert metered == [("codegraph:search", "empty")]


def test_emit_tool_activity_no_hint_without_index(observer, tmp_path, monkeypatch):
    from skill_hub import activity_log as al
    log_file = tmp_path / "activity.log"
    monkeypatch.setattr(al, "LOG_DIR", tmp_path)
    monkeypatch.setattr(al, "LOG_FILE", log_file)
    cfg = {"log_tool_usage": True, "tool_usage_codegraph_hint": True}
    # No .codegraph/ here → grep is logged but not flagged.
    observer._emit_tool_activity("Bash", {"command": "grep foo"}, str(tmp_path), cfg)
    text = log_file.read_text()
    assert "TOOL  Bash" in text
    assert "prefer codegraph_search" not in text


def test_emit_tool_activity_respects_disable_flag(observer, tmp_path, monkeypatch):
    from skill_hub import activity_log as al
    log_file = tmp_path / "activity.log"
    monkeypatch.setattr(al, "LOG_DIR", tmp_path)
    monkeypatch.setattr(al, "LOG_FILE", log_file)
    observer._emit_tool_activity(
        "Bash", {"command": "grep foo"}, str(tmp_path), {"log_tool_usage": False}
    )
    assert not log_file.exists() or log_file.read_text() == ""
