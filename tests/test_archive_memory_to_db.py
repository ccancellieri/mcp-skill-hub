"""Unit tests for scripts/archive_memory_to_db.py."""

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

import archive_memory_to_db as _mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory_dir(tmp_path: Path) -> Path:
    """Return a temporary directory seeded with a minimal MEMORY.md."""
    return tmp_path


def _make_memory_md(directory: Path, lines: list[str]) -> Path:
    """Write a MEMORY.md in *directory* with the given bullet *lines*."""
    header = "# Memory Index\n\n## Projects\n"
    body = "\n".join(lines) + "\n"
    p = directory / "MEMORY.md"
    p.write_text(header + body, encoding="utf-8")
    return p


def _make_md_file(directory: Path, filename: str, content: str) -> Path:
    """Write a .md side-file in *directory*."""
    p = directory / filename
    p.write_text(content, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# _is_archiveable_by_description
# ---------------------------------------------------------------------------


class TestIsArchiveableByDescription:
    def test_done_marker(self):
        assert _mod._is_archiveable_by_description("DONE: finished the work")

    def test_shipped_marker(self):
        assert _mod._is_archiveable_by_description("SHIPPED 2026-04-14: something")

    def test_deferred_marker(self):
        assert _mod._is_archiveable_by_description("DEFERRED Phase B: some notes")

    def test_complete_marker(self):
        assert _mod._is_archiveable_by_description("COMPLETE: leveled/recursive")

    def test_closed_marker(self):
        assert _mod._is_archiveable_by_description("CLOSED: archived")

    def test_case_insensitive(self):
        assert _mod._is_archiveable_by_description("done: finished")

    def test_no_marker(self):
        assert not _mod._is_archiveable_by_description("still active — WIP")

    def test_empty_string(self):
        assert not _mod._is_archiveable_by_description("")

    def test_partial_word_not_matched(self):
        # "undone" should NOT match the word boundary
        assert not _mod._is_archiveable_by_description("undone work here")


# ---------------------------------------------------------------------------
# _is_archiveable_by_body
# ---------------------------------------------------------------------------


class TestIsArchiveableByBody:
    def test_done_at_start_of_line(self):
        body = "---\nname: foo\n---\nDONE 2026-04-16. Merged feature branch."
        assert _mod._is_archiveable_by_body(body)

    def test_shipped_at_start_of_line(self):
        body = "Some preamble\nSHIPPED commit abc123: thing done."
        assert _mod._is_archiveable_by_body(body)

    def test_deferred_at_start_of_line(self):
        body = "DEFERRED Phase B\nSome details"
        assert _mod._is_archiveable_by_body(body)

    def test_complete_in_heading(self):
        body = "# COMPLETE: OGC dimensions hierarchical"
        assert _mod._is_archiveable_by_body(body)

    def test_done_in_h2_heading(self):
        body = "## DONE — multi-driver refactor"
        assert _mod._is_archiveable_by_body(body)

    def test_marker_mid_paragraph_not_matched(self):
        # "DONE" buried mid-sentence in a paragraph body line (not start of line, not heading)
        body = "This was once DONE but now it is active again\nActive work here"
        # NOTE: this SHOULD match because the word DONE appears at start of content
        # per regex: _ARCHIVE_STATUS_RE.match(stripped) — "This was once DONE" won't match
        assert not _mod._is_archiveable_by_body(body)

    def test_empty_body(self):
        assert not _mod._is_archiveable_by_body("")

    def test_active_entry(self):
        body = "Active project with no status marker\nStill in progress"
        assert not _mod._is_archiveable_by_body(body)

    def test_indented_done_matches(self):
        # Indented "DONE" should still match after strip()
        body = "  DONE 2026-04-15: shipped commit abc"
        assert _mod._is_archiveable_by_body(body)


# ---------------------------------------------------------------------------
# _parse_entries
# ---------------------------------------------------------------------------


class TestParseEntries:
    def test_basic_entry(self):
        text = "# Index\n- [my_project.md](my_project.md) — some description\n"
        entries = _mod._parse_entries(text)
        assert len(entries) == 1
        assert entries[0]["title"] == "my_project.md"
        assert entries[0]["filename"] == "my_project.md"
        assert entries[0]["description"] == "some description"

    def test_entry_without_description(self):
        text = "- [file.md](file.md)\n"
        entries = _mod._parse_entries(text)
        assert len(entries) == 1
        assert entries[0]["description"] == ""

    def test_multiple_entries(self):
        text = (
            "- [a.md](a.md) — DONE: first\n"
            "- [b.md](b.md) — active\n"
            "- [c.md](c.md) — SHIPPED: third\n"
        )
        entries = _mod._parse_entries(text)
        assert len(entries) == 3
        assert entries[0]["filename"] == "a.md"
        assert entries[2]["filename"] == "c.md"

    def test_non_entry_lines_ignored(self):
        text = "# Header\n\n## Section\nSome prose line\n- [x.md](x.md) — entry\n"
        entries = _mod._parse_entries(text)
        assert len(entries) == 1

    def test_em_dash_variants(self):
        # Both — (em dash U+2014) and – (en dash U+2013) and - should work
        for dash in ["—", "–", "-"]:
            text = f"- [f.md](f.md) {dash} desc\n"
            entries = _mod._parse_entries(text)
            assert len(entries) == 1, f"failed for dash={dash!r}"
            assert entries[0]["description"] == "desc"


# ---------------------------------------------------------------------------
# Dry-run mode: no filesystem changes
# ---------------------------------------------------------------------------


class TestDryRunMode:
    def test_no_filesystem_changes(self, memory_dir: Path):
        _make_md_file(
            memory_dir, "project_done.md",
            "DONE 2026-04-16. Finished everything."
        )
        memory_path = _make_memory_md(
            memory_dir,
            ["- [project_done.md](project_done.md) — DONE: finished"],
        )

        before_text = memory_path.read_text(encoding="utf-8")
        archive_dir = memory_dir / "_archive"

        _mod.run(memory_path=memory_path, dry_run=True)

        # MEMORY.md must be unchanged
        assert memory_path.read_text(encoding="utf-8") == before_text
        # _archive/ must NOT be created
        assert not archive_dir.exists()
        # Original .md file must still be in place
        assert (memory_dir / "project_done.md").exists()

    def test_nothing_to_archive_dry_run(self, memory_dir: Path):
        _make_md_file(
            memory_dir, "project_active.md",
            "Still active — WIP"
        )
        memory_path = _make_memory_md(
            memory_dir,
            ["- [project_active.md](project_active.md) — active work in progress"],
        )
        before_text = memory_path.read_text(encoding="utf-8")

        count = _mod.run(memory_path=memory_path, dry_run=True)
        assert count == 0
        assert memory_path.read_text(encoding="utf-8") == before_text


# ---------------------------------------------------------------------------
# Live mode: filesystem changes
# ---------------------------------------------------------------------------


class TestLiveMode:
    def _make_store_mock(self):
        store = MagicMock()
        store.save_task.return_value = 42
        store.close_task.return_value = True
        return store

    def test_archive_dir_created(self, memory_dir: Path):
        _make_md_file(memory_dir, "done_project.md", "DONE 2026-04-16. Merged.")
        memory_path = _make_memory_md(
            memory_dir,
            ["- [done_project.md](done_project.md) — DONE: done"],
        )
        store_mock = self._make_store_mock()

        with (
            patch("archive_memory_to_db.SkillStore", return_value=store_mock),
            patch("archive_memory_to_db.embed_available", return_value=False),
        ):
            _mod.run(memory_path=memory_path, dry_run=False)

        assert (memory_dir / "_archive").is_dir()

    def test_md_file_moved_to_archive(self, memory_dir: Path):
        _make_md_file(memory_dir, "done_project.md", "DONE 2026-04-16. Merged.")
        memory_path = _make_memory_md(
            memory_dir,
            ["- [done_project.md](done_project.md) — DONE: done"],
        )
        store_mock = self._make_store_mock()

        with (
            patch("archive_memory_to_db.SkillStore", return_value=store_mock),
            patch("archive_memory_to_db.embed_available", return_value=False),
        ):
            _mod.run(memory_path=memory_path, dry_run=False)

        assert not (memory_dir / "done_project.md").exists()
        assert (memory_dir / "_archive" / "done_project.md").exists()

    def test_memory_md_line_removed(self, memory_dir: Path):
        _make_md_file(memory_dir, "done_project.md", "DONE 2026-04-16. Merged.")
        _make_md_file(memory_dir, "active_project.md", "Still active — WIP")
        memory_path = _make_memory_md(
            memory_dir,
            [
                "- [done_project.md](done_project.md) — DONE: done",
                "- [active_project.md](active_project.md) — active",
            ],
        )
        store_mock = self._make_store_mock()

        with (
            patch("archive_memory_to_db.SkillStore", return_value=store_mock),
            patch("archive_memory_to_db.embed_available", return_value=False),
        ):
            _mod.run(memory_path=memory_path, dry_run=False)

        remaining = memory_path.read_text(encoding="utf-8")
        assert "done_project.md" not in remaining
        assert "active_project.md" in remaining

    def test_save_and_close_task_called(self, memory_dir: Path):
        _make_md_file(memory_dir, "shipped.md", "SHIPPED commit abc. Done.")
        memory_path = _make_memory_md(
            memory_dir,
            ["- [shipped.md](shipped.md) — SHIPPED: stuff"],
        )
        store_mock = self._make_store_mock()

        with (
            patch("archive_memory_to_db.SkillStore", return_value=store_mock),
            patch("archive_memory_to_db.embed_available", return_value=False),
        ):
            _mod.run(memory_path=memory_path, dry_run=False)

        store_mock.save_task.assert_called_once()
        store_mock.close_task.assert_called_once_with(
            task_id=42, compact=pytest.approx("SHIPPED: stuff", abs=None), compact_vector=None
        )

    def test_return_count(self, memory_dir: Path):
        for name in ["a.md", "b.md"]:
            _make_md_file(memory_dir, name, "DONE: done")
        memory_path = _make_memory_md(
            memory_dir,
            [
                "- [a.md](a.md) — DONE: first",
                "- [b.md](b.md) — DONE: second",
            ],
        )
        store_mock = self._make_store_mock()

        with (
            patch("archive_memory_to_db.SkillStore", return_value=store_mock),
            patch("archive_memory_to_db.embed_available", return_value=False),
        ):
            count = _mod.run(memory_path=memory_path, dry_run=False)

        assert count == 2


# ---------------------------------------------------------------------------
# Embedding failure handled gracefully
# ---------------------------------------------------------------------------


class TestEmbeddingFailure:
    def test_embed_failure_saves_without_vector(self, memory_dir: Path):
        _make_md_file(memory_dir, "done.md", "DONE: done")
        memory_path = _make_memory_md(
            memory_dir,
            ["- [done.md](done.md) — DONE: done"],
        )
        store_mock = MagicMock()
        store_mock.save_task.return_value = 7
        store_mock.close_task.return_value = True

        with (
            patch("archive_memory_to_db.SkillStore", return_value=store_mock),
            patch("archive_memory_to_db.embed_available", return_value=True),
            patch("archive_memory_to_db.embed", side_effect=RuntimeError("no model")),
        ):
            count = _mod.run(memory_path=memory_path, dry_run=False)

        assert count == 1
        # save_task must still be called, with empty vector
        call_kwargs = store_mock.save_task.call_args
        assert call_kwargs is not None
        # save_task is always called with keyword arguments; positional fallback
        # (args[2]) is not reliable — use kwargs exclusively.
        vec_arg = call_kwargs.kwargs.get("vector")
        if vec_arg is None and len(call_kwargs.args) > 2:
            vec_arg = call_kwargs.args[2]
        assert vec_arg == []

    def test_embed_success_passes_vector(self, memory_dir: Path):
        _make_md_file(memory_dir, "done.md", "DONE: done")
        memory_path = _make_memory_md(
            memory_dir,
            ["- [done.md](done.md) — DONE: done"],
        )
        store_mock = MagicMock()
        store_mock.save_task.return_value = 8
        store_mock.close_task.return_value = True
        fake_vec = [0.1, 0.2, 0.3]

        with (
            patch("archive_memory_to_db.SkillStore", return_value=store_mock),
            patch("archive_memory_to_db.embed_available", return_value=True),
            patch("archive_memory_to_db.embed", return_value=fake_vec),
        ):
            _mod.run(memory_path=memory_path, dry_run=False)

        call_kwargs = store_mock.save_task.call_args
        vec_arg = call_kwargs.kwargs.get("vector") or call_kwargs.args[2]
        assert vec_arg == fake_vec


# ---------------------------------------------------------------------------
# Missing .md file
# ---------------------------------------------------------------------------


class TestMissingMdFile:
    def test_missing_file_skipped(self, memory_dir: Path):
        # Do NOT create the .md file
        memory_path = _make_memory_md(
            memory_dir,
            ["- [nonexistent.md](nonexistent.md) — DONE: was here"],
        )
        store_mock = MagicMock()

        # The entry has "DONE" in description so _should_archive returns (True, "")
        # The .md file doesn't exist so body is ""; file won't be moved
        # But task WILL be saved (with empty body) — that is acceptable behaviour.
        # The key requirement: no crash, no archive dir created for it unless we
        # actually proceed to live mode.
        with (
            patch("archive_memory_to_db.SkillStore", return_value=store_mock),
            patch("archive_memory_to_db.embed_available", return_value=False),
        ):
            # Should not raise
            _mod.run(memory_path=memory_path, dry_run=False)

        # _archive/ exists but nonexistent.md is obviously not there
        assert not (memory_dir / "_archive" / "nonexistent.md").exists()

    def test_active_missing_file_skipped_gracefully(self, memory_dir: Path, capsys):
        # An active entry whose file is missing: no crash, warning printed
        memory_path = _make_memory_md(
            memory_dir,
            ["- [missing_active.md](missing_active.md) — active work"],
        )
        _mod.run(memory_path=memory_path, dry_run=True)
        # No exception; the WARN line goes to stderr but we just check no crash
