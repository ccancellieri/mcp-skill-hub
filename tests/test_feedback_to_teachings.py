"""Unit tests for scripts/feedback_to_teachings.py."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure project src is importable
SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Import the module under test
SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))

import feedback_to_teachings as _mod


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_feedback_file(
    directory: Path,
    filename: str,
    content: str,
) -> Path:
    """Write a feedback_*.md file in *directory*."""
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

_NO_FRONTMATTER_FEEDBACK = """\
This file has no frontmatter.
It just contains a raw rule.
"""

_ONLY_FRONTMATTER_FEEDBACK = """\
---
name: some-rule
description: a rule with only frontmatter
type: feedback
---
"""


# ---------------------------------------------------------------------------
# _parse_feedback_file
# ---------------------------------------------------------------------------


class TestParseFeedbackFile:
    def test_full_file_extracts_rule_and_why(self, tmp_path: Path):
        path = _make_feedback_file(tmp_path, "feedback_full.md", _FULL_FEEDBACK)
        result = _mod._parse_feedback_file(path)
        assert result is not None
        assert "clarifying questions" in result["rule"]
        assert "stop hook" in result["why"]
        assert result["name"] == "Don't end responses with clarifying questions"

    def test_rule_capped_at_500_chars(self, tmp_path: Path):
        long_body = "x" * 1000
        content = "---\nname: long\ndescription: desc\ntype: feedback\n---\n" + long_body
        path = _make_feedback_file(tmp_path, "feedback_long.md", content)
        result = _mod._parse_feedback_file(path)
        assert result is not None
        assert len(result["rule"]) == 500

    def test_missing_why_falls_back_to_description(self, tmp_path: Path):
        path = _make_feedback_file(tmp_path, "feedback_nowhy.md", _NO_WHY_FEEDBACK)
        result = _mod._parse_feedback_file(path)
        assert result is not None
        assert result["why"] == "Always prefer constants or enums over raw string literals"

    def test_no_frontmatter_uses_body_directly(self, tmp_path: Path):
        path = _make_feedback_file(tmp_path, "feedback_nofm.md", _NO_FRONTMATTER_FEEDBACK)
        result = _mod._parse_feedback_file(path)
        assert result is not None
        assert "no frontmatter" in result["rule"]
        # Name falls back to stem
        assert result["name"] == "feedback_nofm"

    def test_only_frontmatter_no_body_is_handled(self, tmp_path: Path):
        path = _make_feedback_file(tmp_path, "feedback_onlyfm.md", _ONLY_FRONTMATTER_FEEDBACK)
        result = _mod._parse_feedback_file(path)
        # No body content — should fall back to description (non-empty) and succeed
        assert result is not None
        assert result["rule"] == "a rule with only frontmatter"
        assert result["description"] == "a rule with only frontmatter"

    def test_empty_file_returns_none(self, tmp_path: Path):
        path = _make_feedback_file(tmp_path, "feedback_empty.md", "")
        result = _mod._parse_feedback_file(path)
        assert result is None

    def test_missing_file_returns_none(self, tmp_path: Path):
        path = tmp_path / "feedback_missing.md"
        result = _mod._parse_feedback_file(path)
        assert result is None

    def test_name_extracted_from_frontmatter(self, tmp_path: Path):
        path = _make_feedback_file(tmp_path, "feedback_full.md", _FULL_FEEDBACK)
        result = _mod._parse_feedback_file(path)
        assert result is not None
        assert result["name"] == "Don't end responses with clarifying questions"

    def test_description_extracted_from_frontmatter(self, tmp_path: Path):
        path = _make_feedback_file(tmp_path, "feedback_full.md", _FULL_FEEDBACK)
        result = _mod._parse_feedback_file(path)
        assert result is not None
        assert "clarifying question" in result["description"]


# ---------------------------------------------------------------------------
# run() — dry-run mode
# ---------------------------------------------------------------------------


class TestDryRunMode:
    def test_dry_run_makes_no_db_calls(self, tmp_path: Path):
        _make_feedback_file(tmp_path, "feedback_test.md", _FULL_FEEDBACK)

        store_mock = MagicMock()
        with (
            patch("feedback_to_teachings.SkillStore", return_value=store_mock),
            patch("feedback_to_teachings.embed_available", return_value=False),
        ):
            count = _mod.run(memory_dir=tmp_path, dry_run=True)

        store_mock.add_teaching.assert_not_called()
        assert count == 1

    def test_dry_run_returns_correct_count(self, tmp_path: Path):
        _make_feedback_file(tmp_path, "feedback_a.md", _FULL_FEEDBACK)
        _make_feedback_file(tmp_path, "feedback_b.md", _NO_WHY_FEEDBACK)

        count = _mod.run(memory_dir=tmp_path, dry_run=True)
        assert count == 2

    def test_dry_run_empty_dir_returns_zero(self, tmp_path: Path):
        count = _mod.run(memory_dir=tmp_path, dry_run=True)
        assert count == 0

    def test_dry_run_skips_non_feedback_files(self, tmp_path: Path):
        # Create a non-feedback_ file — should NOT be counted
        (tmp_path / "project_something.md").write_text("content", encoding="utf-8")
        _make_feedback_file(tmp_path, "feedback_real.md", _FULL_FEEDBACK)

        count = _mod.run(memory_dir=tmp_path, dry_run=True)
        assert count == 1


# ---------------------------------------------------------------------------
# run() — live mode
# ---------------------------------------------------------------------------


class TestLiveMode:
    def _make_store_mock(self) -> MagicMock:
        store = MagicMock()
        store.add_teaching.return_value = 1
        return store

    def test_add_teaching_called_for_each_file(self, tmp_path: Path):
        _make_feedback_file(tmp_path, "feedback_a.md", _FULL_FEEDBACK)
        _make_feedback_file(tmp_path, "feedback_b.md", _NO_WHY_FEEDBACK)
        store_mock = self._make_store_mock()

        with (
            patch("feedback_to_teachings.SkillStore", return_value=store_mock),
            patch("feedback_to_teachings.embed_available", return_value=False),
            patch("feedback_to_teachings._SKILL_HUB_AVAILABLE", True),
        ):
            count = _mod.run(memory_dir=tmp_path, dry_run=False)

        assert store_mock.add_teaching.call_count == 2
        assert count == 2

    def test_add_teaching_receives_correct_rule(self, tmp_path: Path):
        _make_feedback_file(tmp_path, "feedback_a.md", _FULL_FEEDBACK)
        store_mock = self._make_store_mock()

        with (
            patch("feedback_to_teachings.SkillStore", return_value=store_mock),
            patch("feedback_to_teachings.embed_available", return_value=False),
            patch("feedback_to_teachings._SKILL_HUB_AVAILABLE", True),
        ):
            _mod.run(memory_dir=tmp_path, dry_run=False)

        call_kwargs = store_mock.add_teaching.call_args
        rule_arg = call_kwargs.kwargs.get("rule") or call_kwargs.args[0]
        assert "clarifying questions" in rule_arg

    def test_add_teaching_uses_global_target(self, tmp_path: Path):
        _make_feedback_file(tmp_path, "feedback_a.md", _FULL_FEEDBACK)
        store_mock = self._make_store_mock()

        with (
            patch("feedback_to_teachings.SkillStore", return_value=store_mock),
            patch("feedback_to_teachings.embed_available", return_value=False),
            patch("feedback_to_teachings._SKILL_HUB_AVAILABLE", True),
        ):
            _mod.run(memory_dir=tmp_path, dry_run=False)

        call_kwargs = store_mock.add_teaching.call_args
        kw = call_kwargs.kwargs
        assert kw.get("target_type") == "global"
        assert kw.get("target_id") == "global"

    def test_add_teaching_action_from_why(self, tmp_path: Path):
        _make_feedback_file(tmp_path, "feedback_a.md", _FULL_FEEDBACK)
        store_mock = self._make_store_mock()

        with (
            patch("feedback_to_teachings.SkillStore", return_value=store_mock),
            patch("feedback_to_teachings.embed_available", return_value=False),
            patch("feedback_to_teachings._SKILL_HUB_AVAILABLE", True),
        ):
            _mod.run(memory_dir=tmp_path, dry_run=False)

        call_kwargs = store_mock.add_teaching.call_args
        action_arg = call_kwargs.kwargs.get("action")
        assert action_arg is not None
        assert "stop hook" in action_arg

    def test_add_teaching_action_falls_back_to_name(self, tmp_path: Path):
        # File with no why and no description => action falls back to name
        no_why_no_desc = "---\nname: my-rule\ndescription: \ntype: feedback\n---\nThe rule body.\n"
        _make_feedback_file(tmp_path, "feedback_b.md", no_why_no_desc)
        store_mock = self._make_store_mock()

        with (
            patch("feedback_to_teachings.SkillStore", return_value=store_mock),
            patch("feedback_to_teachings.embed_available", return_value=False),
            patch("feedback_to_teachings._SKILL_HUB_AVAILABLE", True),
        ):
            _mod.run(memory_dir=tmp_path, dry_run=False)

        call_kwargs = store_mock.add_teaching.call_args
        action_arg = call_kwargs.kwargs.get("action")
        assert action_arg is not None
        assert "my-rule" in action_arg

    def test_missing_why_action_uses_description(self, tmp_path: Path):
        _make_feedback_file(tmp_path, "feedback_nowhy.md", _NO_WHY_FEEDBACK)
        store_mock = self._make_store_mock()

        with (
            patch("feedback_to_teachings.SkillStore", return_value=store_mock),
            patch("feedback_to_teachings.embed_available", return_value=False),
            patch("feedback_to_teachings._SKILL_HUB_AVAILABLE", True),
        ):
            _mod.run(memory_dir=tmp_path, dry_run=False)

        call_kwargs = store_mock.add_teaching.call_args
        action_arg = call_kwargs.kwargs.get("action")
        assert "constants or enums" in action_arg

    def test_return_count_matches_files(self, tmp_path: Path):
        for name in ["feedback_x.md", "feedback_y.md", "feedback_z.md"]:
            _make_feedback_file(tmp_path, name, _NO_WHY_FEEDBACK)
        store_mock = self._make_store_mock()

        with (
            patch("feedback_to_teachings.SkillStore", return_value=store_mock),
            patch("feedback_to_teachings.embed_available", return_value=False),
            patch("feedback_to_teachings._SKILL_HUB_AVAILABLE", True),
        ):
            count = _mod.run(memory_dir=tmp_path, dry_run=False)

        assert count == 3


# ---------------------------------------------------------------------------
# Embedding handling
# ---------------------------------------------------------------------------


class TestEmbeddingHandling:
    def test_embed_failure_still_saves_with_zero_vector(self, tmp_path: Path):
        _make_feedback_file(tmp_path, "feedback_a.md", _FULL_FEEDBACK)
        store_mock = MagicMock()
        store_mock.add_teaching.return_value = 5

        with (
            patch("feedback_to_teachings.SkillStore", return_value=store_mock),
            patch("feedback_to_teachings.embed_available", return_value=True),
            patch("feedback_to_teachings.embed", side_effect=RuntimeError("no model")),
            patch("feedback_to_teachings._SKILL_HUB_AVAILABLE", True),
        ):
            count = _mod.run(memory_dir=tmp_path, dry_run=False)

        assert count == 1
        store_mock.add_teaching.assert_called_once()
        call_kwargs = store_mock.add_teaching.call_args
        vec_arg = call_kwargs.kwargs.get("rule_vector")
        assert isinstance(vec_arg, list)
        # All zeros for fallback vector
        assert all(v == 0.0 for v in vec_arg)

    def test_embed_success_passes_vector(self, tmp_path: Path):
        _make_feedback_file(tmp_path, "feedback_a.md", _FULL_FEEDBACK)
        store_mock = MagicMock()
        store_mock.add_teaching.return_value = 6
        fake_vec = [0.1, 0.2, 0.3]

        with (
            patch("feedback_to_teachings.SkillStore", return_value=store_mock),
            patch("feedback_to_teachings.embed_available", return_value=True),
            patch("feedback_to_teachings.embed", return_value=fake_vec),
            patch("feedback_to_teachings._SKILL_HUB_AVAILABLE", True),
        ):
            _mod.run(memory_dir=tmp_path, dry_run=False)

        call_kwargs = store_mock.add_teaching.call_args
        vec_arg = call_kwargs.kwargs.get("rule_vector")
        assert vec_arg == fake_vec
