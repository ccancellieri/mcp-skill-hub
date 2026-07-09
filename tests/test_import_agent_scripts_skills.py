"""Tests for scripts/import_agent_scripts_skills.py."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import import_agent_scripts_skills as importer  # noqa: E402


def _write_skill(root: Path, slug: str, body: str = "Body\n") -> Path:
    skill_dir = root / "skills" / slug
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {slug}
description: Test skill {slug}.
---

{body}
""",
        encoding="utf-8",
    )
    return skill_dir


def test_discover_finds_direct_skill_dirs_only(tmp_path: Path):
    _write_skill(tmp_path, "alpha")
    nested = tmp_path / "skills" / "alpha" / "references"
    nested.mkdir()
    (nested / "SKILL.md").write_text("# not standalone\n", encoding="utf-8")

    found = importer.discover_skill_dirs(tmp_path)

    assert [p.relative_to(tmp_path).as_posix() for p in found] == ["skills/alpha"]


def test_run_copies_full_skill_directory_with_marker(tmp_path: Path):
    source_skill = _write_skill(tmp_path, "alpha")
    (source_skill / "references").mkdir()
    (source_skill / "references" / "guide.md").write_text("Guide\n", encoding="utf-8")

    dest = tmp_path / "out"
    result = importer.run(tmp_path, dest)

    target = dest / "agent-scripts__alpha" / "SKILL.md"
    assert result.written == [target]
    assert target.exists()
    assert importer.IMPORT_MARKER in target.read_text(encoding="utf-8")
    assert (dest / "agent-scripts__alpha" / "references" / "guide.md").read_text(
        encoding="utf-8"
    ) == "Guide\n"


def test_run_is_idempotent(tmp_path: Path):
    _write_skill(tmp_path, "alpha")
    dest = tmp_path / "out"

    first = importer.run(tmp_path, dest)
    assert first.written

    target = dest / "agent-scripts__alpha" / "SKILL.md"
    mtime = target.stat().st_mtime_ns

    second = importer.run(tmp_path, dest)

    assert second.skipped == [target]
    assert not second.written
    assert not second.overwrote
    assert target.stat().st_mtime_ns == mtime


def test_changed_owned_skill_is_overwritten(tmp_path: Path):
    _write_skill(tmp_path, "alpha", "Original\n")
    dest = tmp_path / "out"
    importer.run(tmp_path, dest)

    src = tmp_path / "skills" / "alpha" / "SKILL.md"
    src.write_text(
        src.read_text(encoding="utf-8").replace("Original", "Updated"),
        encoding="utf-8",
    )

    second = importer.run(tmp_path, dest)

    target = dest / "agent-scripts__alpha" / "SKILL.md"
    assert second.overwrote == [target]
    assert "Updated" in target.read_text(encoding="utf-8")


def test_foreign_destination_is_preserved(tmp_path: Path):
    _write_skill(tmp_path, "alpha")
    dest = tmp_path / "out"
    target_dir = dest / "agent-scripts__alpha"
    target_dir.mkdir(parents=True)
    foreign = target_dir / "SKILL.md"
    foreign.write_text("# hand-authored\n", encoding="utf-8")

    result = importer.run(tmp_path, dest)

    assert result.skipped_foreign == [foreign]
    assert foreign.read_text(encoding="utf-8") == "# hand-authored\n"


def test_dry_run_writes_nothing(tmp_path: Path):
    _write_skill(tmp_path, "alpha")

    result = importer.run(tmp_path, tmp_path / "out", dry_run=True)

    assert result.written
    assert not (tmp_path / "out").exists()


def test_security_notes_classify_sensitive_skill(tmp_path: Path):
    _write_skill(
        tmp_path,
        "release",
        "Use 1Password, git push, and git reset --hard only after approval.\n",
    )

    result = importer.run(tmp_path, tmp_path / "out", dry_run=True)

    assert result.security_notes["release"] == [
        "credential workflow reference",
        "git push workflow",
        "destructive git reset reference",
    ]


def test_marker_check_does_not_follow_outside_runtime_imports(tmp_path: Path):
    src = (_SCRIPTS / "import_agent_scripts_skills.py").read_text(encoding="utf-8")
    for forbidden in ("import agent_scripts", "from agent_scripts"):
        assert forbidden not in src


def test_main_returns_error_for_missing_root(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    rc = importer.main(
        ["--source", str(tmp_path / "missing"), "--dest", str(tmp_path / "out")]
    )

    assert rc == 1
    assert "source not found" in capsys.readouterr().err


def test_fixture_like_repo_with_package_files_imports_only_skills(tmp_path: Path):
    repo = tmp_path / "repo"
    _write_skill(repo, "alpha")
    (repo / "package.json").write_text("{}", encoding="utf-8")
    shutil.copytree(repo, tmp_path / "copy")

    result = importer.run(tmp_path / "copy", tmp_path / "out")

    assert [p.parent.name for p in result.written] == ["agent-scripts__alpha"]
