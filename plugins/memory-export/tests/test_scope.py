"""Tests for memexp.scope."""
from __future__ import annotations

from pathlib import Path

import pytest

from memexp import scope


def _make_project(root: Path, key: str, *, files: dict[str, str], private: dict[str, str] | None = None) -> Path:
    proj = root / key / "memory"
    proj.mkdir(parents=True)
    for name, body in files.items():
        (proj / name).write_text(body, encoding="utf-8")
    if private:
        (proj / "private").mkdir()
        for name, body in private.items():
            (proj / "private" / name).write_text(body, encoding="utf-8")
    return proj


def test_list_projects_counts_top_and_private(tmp_path):
    _make_project(
        tmp_path,
        "proj-a",
        files={"a.md": "alpha", "MEMORY.md": "# index"},
        private={"secret.md": "shhh"},
    )
    _make_project(
        tmp_path,
        "proj-b",
        files={"b.md": "beta"},
    )

    projects = scope.list_projects(tmp_path)
    by_key = {p.key: p for p in projects}
    assert by_key["proj-a"].top_level_md_count == 2
    assert by_key["proj-a"].private_md_count == 1
    assert by_key["proj-a"].has_memory_index is True
    assert by_key["proj-b"].top_level_md_count == 1
    assert by_key["proj-b"].private_md_count == 0
    assert by_key["proj-b"].has_memory_index is False


def test_list_projects_skips_non_memory_dirs(tmp_path):
    (tmp_path / "proj-c").mkdir()
    (tmp_path / "proj-c" / "not-memory").mkdir()
    assert scope.list_projects(tmp_path) == []


def test_filter_memory_index_drops_private_links_and_section():
    text = (
        "# Index\n"
        "- [keep](public.md) — kept line\n"
        "- [drop](private/secret.md) — gone\n"
        "## Private Projects\n"
        "- [also drop](./private/other.md)\n"
        "## Other\n"
        "- [keep too](work.md)\n"
    )
    out = scope.filter_memory_index(text)
    assert "public.md" in out
    assert "work.md" in out
    assert "private" not in out
    assert "Private Projects" not in out


def test_filter_memory_index_idempotent():
    text = "# X\n- [a](file.md)\n"
    once = scope.filter_memory_index(text)
    twice = scope.filter_memory_index(once)
    assert once == twice


def test_scan_for_pii_flags_default_tokens(tmp_path):
    a = tmp_path / "a.md"
    a.write_text("notes about Glicemia readings and dosing", encoding="utf-8")
    b = tmp_path / "b.md"
    b.write_text("clean public notes", encoding="utf-8")
    offenders = scope.scan_for_pii([a, b])
    assert a in offenders
    assert b not in offenders


def test_scan_for_pii_extra_tokens(tmp_path):
    a = tmp_path / "a.md"
    a.write_text("contains secret-token-xyz inside", encoding="utf-8")
    assert scope.scan_for_pii([a]) == []
    assert scope.scan_for_pii([a], extra_tokens=("secret-token",)) == [a]


def test_list_exportable_tables_excludes_ephemeral():
    import sqlite3

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE skills (id TEXT)")
    conn.execute("CREATE TABLE teachings (id TEXT)")
    conn.execute("CREATE TABLE session_log (id INTEGER)")  # ephemeral
    conn.execute("CREATE TABLE response_cache (id INTEGER)")  # ephemeral
    out = scope.list_exportable_tables(conn)
    assert "skills" in out
    assert "teachings" in out
    assert "session_log" not in out
    assert "response_cache" not in out
