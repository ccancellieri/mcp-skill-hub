"""Tests for scripts/import_ruflo_skills.py."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

# Make the script importable as a module without executing it.
_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import import_ruflo_skills as importer  # noqa: E402

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "ruflo-fake"


# ---------------------------------------------------------------------------
# Constraint: no runtime ruflo / claude-flow import
# ---------------------------------------------------------------------------


def test_no_runtime_ruflo_import():
    """The importer source must not import ruflo / claude-flow at runtime."""
    src = (_SCRIPTS / "import_ruflo_skills.py").read_text(encoding="utf-8")
    # The doc comment legitimately mentions ruflo, so check imports, not free text.
    for forbidden in (
        "import ruflo",
        "from ruflo",
        "import claude_flow",
        "from claude_flow",
    ):
        assert forbidden not in src, f"forbidden import sneaked in: {forbidden}"


# ---------------------------------------------------------------------------
# detect_ruflo_root
# ---------------------------------------------------------------------------


class TestDetectRufloRoot:
    def test_explicit_path_wins(self, tmp_path: Path):
        explicit = tmp_path / "explicit"
        explicit.mkdir()
        env_root = tmp_path / "env"
        env_root.mkdir()
        got = importer.detect_ruflo_root(
            explicit,
            env={"RUFLO_ROOT": str(env_root)},
            candidates=(tmp_path / "missing",),
        )
        assert got == explicit

    def test_explicit_nonexistent_returns_none(self, tmp_path: Path):
        assert importer.detect_ruflo_root(tmp_path / "nope") is None

    def test_env_var_used_when_no_explicit(self, tmp_path: Path):
        env_root = tmp_path / "env"
        env_root.mkdir()
        got = importer.detect_ruflo_root(
            None,
            env={"RUFLO_ROOT": str(env_root)},
            candidates=(tmp_path / "missing",),
        )
        assert got == env_root

    def test_candidates_fallback(self, tmp_path: Path):
        c1 = tmp_path / "c1"
        c2 = tmp_path / "c2"
        c2.mkdir()
        got = importer.detect_ruflo_root(None, env={}, candidates=(c1, c2))
        assert got == c2

    def test_nothing_found(self, tmp_path: Path):
        got = importer.detect_ruflo_root(
            None, env={}, candidates=(tmp_path / "a", tmp_path / "b")
        )
        assert got is None


# ---------------------------------------------------------------------------
# parse_frontmatter
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_basic(self):
        fm, body = importer.parse_frontmatter(
            "---\nname: foo\ndescription: bar\n---\nhello\n"
        )
        assert fm == {"name": "foo", "description": "bar"}
        assert body == "hello\n"

    def test_quoted_values_unquoted(self):
        fm, _ = importer.parse_frontmatter(
            "---\nname: foo\nargument-hint: \"a|b\"\n---\nx\n"
        )
        assert fm["argument-hint"] == "a|b"

    def test_no_frontmatter(self):
        fm, body = importer.parse_frontmatter("# Heading\n\nbody\n")
        assert fm == {}
        assert body == "# Heading\n\nbody\n"

    def test_indented_continuation_ignored(self):
        fm, _ = importer.parse_frontmatter(
            "---\nname: foo\nnested:\n  child: skipped\n---\nx\n"
        )
        assert fm == {"name": "foo"}


# ---------------------------------------------------------------------------
# discover_skill_files — must include skills/, exclude agents/
# ---------------------------------------------------------------------------


def test_discover_finds_skills_only():
    files = importer.discover_skill_files(FIXTURES)
    rels = sorted(p.relative_to(FIXTURES).as_posix() for p in files)
    assert rels == [
        "plugins/ruflo-core/0.2.1/skills/no-frontmatter/SKILL.md",
        "plugins/ruflo-core/0.2.1/skills/witness/SKILL.md",
        "plugins/ruflo-swarm/skills/monitor-stream/SKILL.md",
        "plugins/ruflo-swarm/skills/swarm-init/SKILL.md",
    ]


# ---------------------------------------------------------------------------
# _infer_plugin
# ---------------------------------------------------------------------------


class TestInferPlugin:
    def test_versioned_path(self):
        p = Path("/x/plugins/ruflo-core/0.2.1/skills/witness/SKILL.md")
        assert importer._infer_plugin(p) == "ruflo-core"

    def test_unversioned_path(self):
        p = Path("/x/plugins/ruflo-swarm/skills/swarm-init/SKILL.md")
        assert importer._infer_plugin(p) == "ruflo-swarm"

    def test_fallback(self):
        p = Path("/lone/SKILL.md")
        assert importer._infer_plugin(p) == "ruflo"


# ---------------------------------------------------------------------------
# End-to-end run() against the fixture
# ---------------------------------------------------------------------------


def test_run_against_fixture(tmp_path: Path):
    dest = tmp_path / "out"
    result = importer.run(FIXTURES, dest)

    # Three valid skills should land. The no-frontmatter one is reported as error.
    written_rel = sorted(p.relative_to(dest).as_posix() for p in result.written)
    assert written_rel == [
        "imported_ruflo/ruflo-core__witness/SKILL.md",
        "imported_ruflo/ruflo-swarm__monitor-stream/SKILL.md",
        "imported_ruflo/ruflo-swarm__swarm-init/SKILL.md",
    ]
    assert any("no-frontmatter" in e for e in result.errors), result.errors

    # The rendered manifest must keep canonical frontmatter and the import marker.
    witness = (dest / "imported_ruflo" / "ruflo-core__witness" / "SKILL.md").read_text()
    assert witness.startswith("---\n")
    assert "name: witness" in witness
    assert "description: Sign, verify" in witness
    assert "argument-hint:" in witness
    assert "allowed-tools:" in witness
    assert "source: ruflo:ruflo-core:witness" in witness
    assert importer.IMPORT_MARKER in witness
    # extra-field is not in the preserved set
    assert "extra-field" not in witness


def test_run_is_idempotent(tmp_path: Path):
    dest = tmp_path / "out"

    first = importer.run(FIXTURES, dest)
    assert first.written, "first run should write files"
    assert not first.skipped
    assert not first.overwrote

    # Capture mtimes so we can prove nothing got rewritten.
    paths = [
        dest / "imported_ruflo" / "ruflo-swarm__swarm-init" / "SKILL.md",
        dest / "imported_ruflo" / "ruflo-swarm__monitor-stream" / "SKILL.md",
        dest / "imported_ruflo" / "ruflo-core__witness" / "SKILL.md",
    ]
    mtimes = {p: p.stat().st_mtime_ns for p in paths}

    second = importer.run(FIXTURES, dest)
    assert not second.written, "no new writes on re-run"
    assert not second.overwrote, "no overwrites on re-run"
    skipped_rel = sorted(p.relative_to(dest).as_posix() for p in second.skipped)
    assert skipped_rel == [
        "imported_ruflo/ruflo-core__witness/SKILL.md",
        "imported_ruflo/ruflo-swarm__monitor-stream/SKILL.md",
        "imported_ruflo/ruflo-swarm__swarm-init/SKILL.md",
    ]

    for p, mt in mtimes.items():
        assert p.stat().st_mtime_ns == mt, f"file was rewritten on idempotent run: {p}"


def test_dry_run_makes_no_writes(tmp_path: Path):
    dest = tmp_path / "out"
    result = importer.run(FIXTURES, dest, dry_run=True)
    assert result.written  # would-write list populated
    assert not dest.exists() or not any(dest.rglob("SKILL.md"))


def test_overwrite_when_source_changes(tmp_path: Path):
    """If a source skill changes, the importer rewrites the destination."""
    # Copy the fixture into a writable temp location so we can mutate it.
    src_root = tmp_path / "ruflo"
    shutil.copytree(FIXTURES, src_root)
    dest = tmp_path / "out"

    importer.run(src_root, dest)
    target = dest / "imported_ruflo" / "ruflo-swarm__monitor-stream" / "SKILL.md"
    assert target.exists()

    # Mutate the source description
    src_file = src_root / "plugins/ruflo-swarm/skills/monitor-stream/SKILL.md"
    src_file.write_text(
        src_file.read_text().replace(
            "Live-stream swarm events and agent activity in real time",
            "Updated description for monitor-stream",
        ),
        encoding="utf-8",
    )

    second = importer.run(src_root, dest)
    overwrote_rel = [p.relative_to(dest).as_posix() for p in second.overwrote]
    assert "imported_ruflo/ruflo-swarm__monitor-stream/SKILL.md" in overwrote_rel
    assert "Updated description for monitor-stream" in target.read_text()


def test_foreign_destination_is_preserved(tmp_path: Path):
    """Hand-authored skills at the dest path must NOT be overwritten."""
    dest = tmp_path / "out"
    target_dir = dest / "imported_ruflo" / "ruflo-swarm__swarm-init"
    target_dir.mkdir(parents=True)
    foreign = target_dir / "SKILL.md"
    foreign.write_text("# hand-authored — keep me\n", encoding="utf-8")

    result = importer.run(FIXTURES, dest)
    assert foreign.read_text() == "# hand-authored — keep me\n"
    assert foreign in result.skipped_foreign


def test_run_with_missing_root_reports_error(tmp_path: Path):
    result = importer.run(
        tmp_path / "does-not-exist",
        tmp_path / "out",
    )
    assert not result.written
    assert result.errors
    assert "no ruflo install found" in result.errors[0]


# ---------------------------------------------------------------------------
# main() CLI entry
# ---------------------------------------------------------------------------


def test_main_returns_0_on_success(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    rc = importer.main([
        "--ruflo-root", str(FIXTURES),
        "--dest", str(tmp_path / "out"),
    ])
    # The fixture contains one no-frontmatter file -> errors -> rc=1.
    # That's correct behaviour: we surface parse failures non-silently.
    assert rc == 1
    out = capsys.readouterr()
    assert "written:" in out.out
    assert "no-frontmatter" in out.err


def test_main_dry_run_writes_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    dest = tmp_path / "out"
    importer.main([
        "--ruflo-root", str(FIXTURES),
        "--dest", str(dest),
        "--dry-run",
    ])
    out = capsys.readouterr().out
    assert "[dry-run]" in out
    assert not dest.exists() or not any(dest.rglob("SKILL.md"))
