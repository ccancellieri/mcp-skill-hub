"""Tests for ``verdict_cache.active_task_id`` / ``task_tag`` helpers.

These back the per-task log-scoping feature: hook ``log()`` helpers prepend a
``task=<id>`` token whenever ``~/.claude/mcp-skill-hub/state/active_task.json``
points at an open task, so the dashboard can filter log lines per task.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HOOKS = Path(__file__).resolve().parent.parent / "hooks"
sys.path.insert(0, str(HOOKS))

import verdict_cache  # noqa: E402


def _point_marker_at(tmp_path: Path, monkeypatch) -> Path:
    marker = tmp_path / "active_task.json"
    monkeypatch.setattr(verdict_cache, "ACTIVE_TASK_MARKER", marker)
    return marker


def test_active_task_id_missing_marker(tmp_path, monkeypatch):
    _point_marker_at(tmp_path, monkeypatch)
    assert verdict_cache.active_task_id() is None
    assert verdict_cache.task_tag() == ""


def test_active_task_id_invalid_json(tmp_path, monkeypatch):
    marker = _point_marker_at(tmp_path, monkeypatch)
    marker.write_text("{not valid json")
    assert verdict_cache.active_task_id() is None
    assert verdict_cache.task_tag() == ""


def test_active_task_id_missing_field(tmp_path, monkeypatch):
    marker = _point_marker_at(tmp_path, monkeypatch)
    marker.write_text(json.dumps({"session_id": "abc"}))
    assert verdict_cache.active_task_id() is None


def test_active_task_id_bool_rejected(tmp_path, monkeypatch):
    marker = _point_marker_at(tmp_path, monkeypatch)
    marker.write_text(json.dumps({"task_id": True}))
    assert verdict_cache.active_task_id() is None


def test_active_task_id_int(tmp_path, monkeypatch):
    marker = _point_marker_at(tmp_path, monkeypatch)
    marker.write_text(json.dumps({"task_id": 102, "title": "x"}))
    assert verdict_cache.active_task_id() == 102
    assert verdict_cache.task_tag() == "task=102 "


def test_active_task_id_string_digits(tmp_path, monkeypatch):
    marker = _point_marker_at(tmp_path, monkeypatch)
    marker.write_text(json.dumps({"task_id": "17"}))
    assert verdict_cache.active_task_id() == 17


def test_tagged_line_format(tmp_path, monkeypatch):
    marker = _point_marker_at(tmp_path, monkeypatch)
    marker.write_text(json.dumps({"task_id": 102}))
    line = f"[23:47:12] AUTO_APPROVE {verdict_cache.task_tag()}Bash decision=approve"
    assert "task=102" in line
    assert line.startswith("[23:47:12] AUTO_APPROVE task=102 Bash")
