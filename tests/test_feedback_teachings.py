"""Unit tests for skill_hub.feedback_teachings (issue #114)."""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest

from skill_hub import feedback_teachings as _mod
from skill_hub.store import SkillStore


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    s = SkillStore(db_path=tmp_path / "skill_hub.db")
    yield s
    s.close()


def _write_feedback(directory: Path, filename: str, content: str) -> Path:
    p = directory / filename
    p.write_text(content, encoding="utf-8")
    return p


_FULL_FEEDBACK = """\
---
name: Don't end responses with clarifying questions
description: Stop hook fires when assistant ends with a clarifying question
type: feedback
---
Don't end responses with clarifying questions when context is sufficient to act.

**Why:** The stop hook is configured to fire on this pattern.

**How to apply:** State your assumption and proceed.
"""

_NO_WHY_FEEDBACK = """\
---
name: Use constants over literals
description: Always prefer constants or enums over raw string literals
type: feedback
---
Always prefer library constants and enums over inline string literals.
"""

_NO_FRONTMATTER_FEEDBACK = "This file has no frontmatter.\nJust a raw rule.\n"


# ---------------------------------------------------------------------------
# _parse_feedback_file
# ---------------------------------------------------------------------------


class TestParseFeedbackFile:
    def test_extracts_rule_and_why(self, tmp_path):
        path = _write_feedback(tmp_path, "feedback_full.md", _FULL_FEEDBACK)
        result = _mod._parse_feedback_file(path)
        assert result is not None
        assert "clarifying questions" in result["rule"]
        assert "stop hook" in result["why"]

    def test_extracts_how_to_apply(self, tmp_path):
        path = _write_feedback(tmp_path, "feedback_full.md", _FULL_FEEDBACK)
        result = _mod._parse_feedback_file(path)
        assert result is not None
        assert "State your assumption" in result["how_to_apply"]

    def test_missing_how_to_apply_is_empty(self, tmp_path):
        path = _write_feedback(tmp_path, "feedback_nowhy.md", _NO_WHY_FEEDBACK)
        result = _mod._parse_feedback_file(path)
        assert result is not None
        assert result["how_to_apply"] == ""

    def test_missing_why_falls_back_to_description(self, tmp_path):
        path = _write_feedback(tmp_path, "feedback_nowhy.md", _NO_WHY_FEEDBACK)
        result = _mod._parse_feedback_file(path)
        assert result is not None
        assert result["why"] == "Always prefer constants or enums over raw string literals"

    def test_no_frontmatter_uses_body_directly(self, tmp_path):
        path = _write_feedback(tmp_path, "feedback_nofm.md", _NO_FRONTMATTER_FEEDBACK)
        result = _mod._parse_feedback_file(path)
        assert result is not None
        assert "no frontmatter" in result["rule"]
        assert result["name"] == "feedback_nofm"

    def test_empty_file_returns_none(self, tmp_path):
        path = _write_feedback(tmp_path, "feedback_empty.md", "")
        assert _mod._parse_feedback_file(path) is None

    def test_missing_file_returns_none(self, tmp_path):
        assert _mod._parse_feedback_file(tmp_path / "feedback_missing.md") is None

    def test_rule_capped_at_500_chars(self, tmp_path):
        content = "---\nname: long\ndescription: desc\ntype: feedback\n---\n" + "x" * 1000
        path = _write_feedback(tmp_path, "feedback_long.md", content)
        result = _mod._parse_feedback_file(path)
        assert result is not None
        assert len(result["rule"]) == 500


# ---------------------------------------------------------------------------
# _build_action / _normalize_rule
# ---------------------------------------------------------------------------


class TestBuildAction:
    def test_why_and_how_combined(self):
        item = {"name": "test", "why": "because reasons", "how_to_apply": "use it here"}
        action = _mod._build_action(item)
        assert "Why: because reasons" in action
        assert "How to apply: use it here" in action

    def test_neither_falls_back_to_name(self):
        item = {"name": "my-rule", "why": "", "how_to_apply": ""}
        assert "my-rule" in _mod._build_action(item)


def test_normalize_rule_collapses_whitespace_and_case():
    assert _mod._normalize_rule("Some   Rule\ntext") == _mod._normalize_rule("some rule text")


# ---------------------------------------------------------------------------
# convert() — conversion, dedup, embedding handling
# ---------------------------------------------------------------------------


class TestConvert:
    def test_converts_new_file_into_teaching(self, tmp_path, store, monkeypatch):
        _write_feedback(tmp_path, "feedback_a.md", _FULL_FEEDBACK)
        monkeypatch.setattr(_mod, "embed_available", lambda: False)

        result = _mod.convert(store, tmp_path)

        assert result == {"found": 1, "converted": 1, "duplicates": 0, "unparsed": 0, "errors": 0}
        rows = store.list_teachings()
        assert len(rows) == 1
        assert "clarifying questions" in rows[0]["rule"]
        assert rows[0]["target_type"] == "global"
        assert rows[0]["target_id"] == "global"

    def test_recursive_scan_across_subdirectories(self, tmp_path, store, monkeypatch):
        sub = tmp_path / "project-a" / "memory"
        sub.mkdir(parents=True)
        _write_feedback(sub, "feedback_nested.md", _FULL_FEEDBACK)
        monkeypatch.setattr(_mod, "embed_available", lambda: False)

        result = _mod.convert(store, tmp_path)
        assert result["found"] == 1
        assert result["converted"] == 1

    def test_missing_memory_root_is_a_noop(self, tmp_path, store):
        result = _mod.convert(store, tmp_path / "does-not-exist")
        assert result == {"found": 0, "converted": 0, "duplicates": 0, "unparsed": 0, "errors": 0}

    def test_unparsed_file_is_counted_not_converted(self, tmp_path, store, monkeypatch):
        _write_feedback(tmp_path, "feedback_empty.md", "")
        monkeypatch.setattr(_mod, "embed_available", lambda: False)

        result = _mod.convert(store, tmp_path)
        assert result["found"] == 1
        assert result["unparsed"] == 1
        assert result["converted"] == 0
        assert store.list_teachings() == []

    def test_duplicate_rule_is_skipped(self, tmp_path, store, monkeypatch):
        _write_feedback(tmp_path, "feedback_a.md", _FULL_FEEDBACK)
        monkeypatch.setattr(_mod, "embed_available", lambda: False)

        first = _mod.convert(store, tmp_path)
        assert first["converted"] == 1

        second = _mod.convert(store, tmp_path)
        assert second["converted"] == 0
        assert second["duplicates"] == 1
        assert len(store.list_teachings()) == 1

    def test_two_files_same_rule_text_only_one_inserted(self, tmp_path, store, monkeypatch):
        _write_feedback(tmp_path, "feedback_a.md", _FULL_FEEDBACK)
        _write_feedback(tmp_path, "feedback_b.md", _FULL_FEEDBACK)
        monkeypatch.setattr(_mod, "embed_available", lambda: False)

        result = _mod.convert(store, tmp_path)
        assert result["converted"] == 1
        assert result["duplicates"] == 1
        assert len(store.list_teachings()) == 1

    def test_rerun_is_idempotent(self, tmp_path, store, monkeypatch):
        _write_feedback(tmp_path, "feedback_a.md", _FULL_FEEDBACK)
        _write_feedback(tmp_path, "feedback_b.md", _NO_WHY_FEEDBACK)
        monkeypatch.setattr(_mod, "embed_available", lambda: False)

        _mod.convert(store, tmp_path)
        _mod.convert(store, tmp_path)
        _mod.convert(store, tmp_path)

        assert len(store.list_teachings()) == 2

    def test_embed_unavailable_inserts_vectorless(self, tmp_path, store, monkeypatch):
        _write_feedback(tmp_path, "feedback_a.md", _FULL_FEEDBACK)
        monkeypatch.setattr(_mod, "embed_available", lambda: False)

        _mod.convert(store, tmp_path)

        row = store._conn.execute("SELECT rule_vector FROM teachings").fetchone()
        assert row["rule_vector"] == "[]"

    def test_embed_failure_inserts_vectorless(self, tmp_path, store, monkeypatch):
        _write_feedback(tmp_path, "feedback_a.md", _FULL_FEEDBACK)
        monkeypatch.setattr(_mod, "embed_available", lambda: True)

        def _boom(text):
            raise RuntimeError("no model")

        monkeypatch.setattr(_mod, "embed", _boom)

        result = _mod.convert(store, tmp_path)
        assert result["converted"] == 1
        row = store._conn.execute("SELECT rule_vector FROM teachings").fetchone()
        assert row["rule_vector"] == "[]"

    def test_embed_success_stores_vector(self, tmp_path, store, monkeypatch):
        _write_feedback(tmp_path, "feedback_a.md", _FULL_FEEDBACK)
        monkeypatch.setattr(_mod, "embed_available", lambda: True)
        monkeypatch.setattr(_mod, "embed", lambda text: [0.1, 0.2, 0.3])

        _mod.convert(store, tmp_path)

        row = store._conn.execute("SELECT rule_vector FROM teachings").fetchone()
        assert row["rule_vector"] == "[0.1, 0.2, 0.3]"
