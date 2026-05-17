"""Tests for skill_hub.sync_check — M3 issue #16.

Covers acceptance criteria:
- Synthetic primary diff + follower files → detection accurate.
- No false positives for symbols that exist in both primary and follower.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest

from skill_hub import sync_check as sc


# ---------------------------------------------------------------------------
# Pure-grep path (no git): use ``removed_symbols=`` to bypass the diff step
# ---------------------------------------------------------------------------


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_detects_stale_ref_in_follower(tmp_path: Path):
    primary = tmp_path / "primary"
    follower = tmp_path / "follower"
    # Primary no longer defines OldClass.
    _write(primary / "src" / "new.py", "class NewClass:\n    pass\n")
    # Follower still imports OldClass.
    _write(
        follower / "src" / "foo.py",
        "from primary import OldClass\n\n\ndef use():\n    return OldClass()\n",
    )

    report = sc.sync_check(
        primary=primary,
        followers=[follower],
        removed_symbols=["OldClass"],
    )

    assert report.error is None
    assert report.removed_symbols == ["OldClass"]
    # Both lines (import + usage) should be detected.
    assert len(report.findings) == 2
    symbols = {f.symbol for f in report.findings}
    assert symbols == {"OldClass"}
    paths = {f.path for f in report.findings}
    assert paths == {"src/foo.py"}
    line_nos = sorted(f.line_no for f in report.findings)
    assert line_nos == [1, 5]


def test_no_false_positive_when_symbol_exists_in_primary(tmp_path: Path):
    """Acceptance: a symbol present in both primary and follower must NOT flag."""
    primary = tmp_path / "primary"
    follower = tmp_path / "follower"
    # Primary still has KeptClass (it was *moved*, not removed).
    _write(primary / "src" / "moved.py", "class KeptClass:\n    pass\n")
    _write(follower / "src" / "ref.py", "from primary import KeptClass\n")

    report = sc.sync_check(
        primary=primary,
        followers=[follower],
        removed_symbols=["KeptClass"],
    )

    assert report.removed_symbols == []  # filtered out: still in primary
    assert report.findings == []


def test_render_format_matches_acceptance_example(tmp_path: Path):
    """Issue example: ``stale ref "OldClass" in follower/src/foo.py:42``."""
    primary = tmp_path / "primary"
    follower = tmp_path / "follower"
    _write(primary / "x.py", "# nothing relevant\n")
    # Pad to line 42.
    _write(
        follower / "src" / "foo.py",
        "\n" * 41 + "from primary import OldClass\n",
    )

    report = sc.sync_check(
        primary=primary,
        followers=[follower],
        removed_symbols=["OldClass"],
    )

    assert len(report.findings) == 1
    finding = report.findings[0]
    assert finding.line_no == 42
    rendered = finding.render()
    assert rendered.startswith('stale ref "OldClass" in ')
    assert rendered.endswith("/src/foo.py:42")


def test_multiple_followers_each_scanned(tmp_path: Path):
    primary = tmp_path / "primary"
    follower_a = tmp_path / "follower_a"
    follower_b = tmp_path / "follower_b"
    _write(primary / "x.py", "# nothing\n")
    _write(follower_a / "a.py", "import GoneSymbol\n")
    _write(follower_b / "b.py", "from x import GoneSymbol\n")

    report = sc.sync_check(
        primary=primary,
        followers=[follower_a, follower_b],
        removed_symbols=["GoneSymbol"],
    )

    assert len(report.findings) == 2
    followers_with_hits = {f.follower for f in report.findings}
    assert len(followers_with_hits) == 2
    assert len(report.followers_scanned) == 2


def test_word_boundary_avoids_substring_match(tmp_path: Path):
    """``Foo`` must not match ``Foobar`` — word-boundary enforcement."""
    primary = tmp_path / "primary"
    follower = tmp_path / "follower"
    _write(primary / "x.py", "# nothing\n")
    _write(follower / "y.py", "x = Foobar\ny = MyFoo\nz = Foo_helper\n")

    report = sc.sync_check(
        primary=primary,
        followers=[follower],
        removed_symbols=["Foo"],
    )

    assert report.findings == []


def test_word_boundary_matches_attribute_access(tmp_path: Path):
    """``mod.Foo`` is a legitimate hit — the dot is a word boundary."""
    primary = tmp_path / "primary"
    follower = tmp_path / "follower"
    _write(primary / "x.py", "# nothing\n")
    _write(follower / "y.py", "return mod.OldClass()\n")

    report = sc.sync_check(
        primary=primary,
        followers=[follower],
        removed_symbols=["OldClass"],
    )

    assert len(report.findings) == 1
    assert report.findings[0].symbol == "OldClass"


def test_excluded_dirs_not_scanned(tmp_path: Path):
    primary = tmp_path / "primary"
    follower = tmp_path / "follower"
    _write(primary / "x.py", "# nothing\n")
    # A node_modules / .venv hit should be ignored.
    _write(follower / "node_modules" / "pkg" / "lib.py", "OldClass\n")
    _write(follower / ".venv" / "lib.py", "OldClass\n")
    # A legit src hit should land.
    _write(follower / "src" / "real.py", "OldClass\n")

    report = sc.sync_check(
        primary=primary,
        followers=[follower],
        removed_symbols=["OldClass"],
    )

    assert len(report.findings) == 1
    assert report.findings[0].path == "src/real.py"


def test_suffix_filter_skips_unsupported_extensions(tmp_path: Path):
    primary = tmp_path / "primary"
    follower = tmp_path / "follower"
    _write(primary / "x.py", "# nothing\n")
    _write(follower / "a.py", "OldClass\n")
    _write(follower / "b.bin", "OldClass\n")  # excluded — not in suffix list

    report = sc.sync_check(
        primary=primary,
        followers=[follower],
        removed_symbols=["OldClass"],
    )

    paths = {f.path for f in report.findings}
    assert paths == {"a.py"}


def test_custom_suffixes_honored(tmp_path: Path):
    primary = tmp_path / "primary"
    follower = tmp_path / "follower"
    _write(primary / "x.py", "# nothing\n")
    _write(follower / "a.custom", "OldClass here\n")

    report = sc.sync_check(
        primary=primary,
        followers=[follower],
        removed_symbols=["OldClass"],
        suffixes=[".custom"],
    )

    assert len(report.findings) == 1
    assert report.findings[0].path == "a.custom"


def test_noise_idents_filtered_from_removed_list(tmp_path: Path):
    primary = tmp_path / "primary"
    follower = tmp_path / "follower"
    _write(primary / "x.py", "# nothing\n")
    _write(follower / "y.py", "return self.foo\n")

    report = sc.sync_check(
        primary=primary,
        followers=[follower],
        removed_symbols=["self", "return", "OldClass"],
    )

    # self / return are noise, OldClass isn't in follower → no findings.
    assert "self" not in report.removed_symbols
    assert "return" not in report.removed_symbols
    assert "OldClass" in report.removed_symbols
    assert report.findings == []


def test_skipped_follower_recorded(tmp_path: Path):
    primary = tmp_path / "primary"
    _write(primary / "x.py", "# nothing\n")
    missing = tmp_path / "does_not_exist"

    report = sc.sync_check(
        primary=primary,
        followers=[missing],
        removed_symbols=["OldClass"],
    )

    assert str(missing) in report.skipped_followers
    assert report.followers_scanned == []


def test_primary_not_a_directory(tmp_path: Path):
    bogus = tmp_path / "not_a_dir"
    report = sc.sync_check(
        primary=bogus,
        followers=[],
        removed_symbols=["OldClass"],
    )
    assert report.error is not None
    assert "primary not a directory" in report.error


# ---------------------------------------------------------------------------
# Git-driven path: build a real tiny git repo and use HEAD~1 as the base ref
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    """Run ``git`` against ``repo`` with a hermetic identity."""
    env_args = [
        "-c", "user.email=test@example.com",
        "-c", "user.name=test",
        "-c", "commit.gpgsign=false",
    ]
    subprocess.run(
        ["git", "-C", str(repo), *env_args, *args],
        check=True,
        capture_output=True,
    )


def _have_git() -> bool:
    from shutil import which
    return which("git") is not None


@pytest.mark.skipif(not _have_git(), reason="git not installed")
def test_git_diff_drives_detection(tmp_path: Path):
    primary = tmp_path / "primary"
    primary.mkdir()
    _git(primary, "init", "-q", "-b", "main")

    # First commit: defines OldClass.
    _write(primary / "lib.py", "class OldClass:\n    pass\n")
    _git(primary, "add", "lib.py")
    _git(primary, "commit", "-q", "-m", "initial")

    # Second commit: rename to NewClass.
    _write(primary / "lib.py", "class NewClass:\n    pass\n")
    _git(primary, "add", "lib.py")
    _git(primary, "commit", "-q", "-m", "rename")

    # Follower still uses OldClass.
    follower = tmp_path / "follower"
    _write(follower / "use.py", "from primary import OldClass\n")

    report = sc.sync_check(
        primary=primary,
        followers=[follower],
        base_ref="HEAD~1",
    )

    assert report.error is None
    assert "OldClass" in report.removed_symbols
    # NewClass exists in primary → must NOT be in removed_symbols.
    assert "NewClass" not in report.removed_symbols
    assert len(report.findings) == 1
    assert report.findings[0].symbol == "OldClass"


@pytest.mark.skipif(not _have_git(), reason="git not installed")
def test_git_diff_no_false_positive_for_moved_symbol(tmp_path: Path):
    """A symbol moved between files in primary must NOT appear stale.

    This is the acceptance criterion's second bullet:
    'No false positives for symbols that exist in both.'
    """
    primary = tmp_path / "primary"
    primary.mkdir()
    _git(primary, "init", "-q", "-b", "main")

    # Initial: KeptClass lives in old_home.py.
    _write(primary / "old_home.py", "class KeptClass:\n    pass\n")
    _git(primary, "add", "old_home.py")
    _git(primary, "commit", "-q", "-m", "initial")

    # Next commit: KeptClass moves to new_home.py. Same name — should be
    # filtered out because it still exists in primary.
    (primary / "old_home.py").unlink()
    _write(primary / "new_home.py", "class KeptClass:\n    pass\n")
    _git(primary, "add", "-A")
    _git(primary, "commit", "-q", "-m", "move")

    follower = tmp_path / "follower"
    _write(follower / "use.py", "from primary import KeptClass\n")

    report = sc.sync_check(
        primary=primary,
        followers=[follower],
        base_ref="HEAD~1",
    )

    assert report.error is None
    assert "KeptClass" not in report.removed_symbols
    assert report.findings == []


@pytest.mark.skipif(not _have_git(), reason="git not installed")
def test_git_diff_bad_ref_surfaces_error(tmp_path: Path):
    primary = tmp_path / "primary"
    primary.mkdir()
    _git(primary, "init", "-q", "-b", "main")
    _write(primary / "x.py", "x = 1\n")
    _git(primary, "add", "x.py")
    _git(primary, "commit", "-q", "-m", "initial")

    report = sc.sync_check(
        primary=primary,
        followers=[],
        base_ref="not_a_real_ref",
    )

    assert report.error is not None
    assert report.findings == []


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------


def test_format_report_summary_when_no_findings(tmp_path: Path):
    primary = tmp_path / "primary"
    _write(primary / "x.py", "# nothing\n")
    report = sc.sync_check(
        primary=primary,
        followers=[],
        removed_symbols=["OldClass"],
    )
    out = sc.format_report(report)
    assert "0 stale refs" in out


def test_format_report_lists_findings(tmp_path: Path):
    primary = tmp_path / "primary"
    follower = tmp_path / "follower"
    _write(primary / "x.py", "# nothing\n")
    _write(follower / "y.py", "import OldClass\n")
    report = sc.sync_check(
        primary=primary,
        followers=[follower],
        removed_symbols=["OldClass"],
    )
    out = sc.format_report(report)
    assert "stale ref" in out
    assert "OldClass" in out


def test_format_report_surfaces_error():
    report = sc.SyncReport(
        primary="/x",
        base_ref="HEAD~1",
        error="bad ref",
    )
    out = sc.format_report(report)
    assert "error" in out.lower()
    assert "bad ref" in out


def test_to_dict_round_trip(tmp_path: Path):
    primary = tmp_path / "primary"
    follower = tmp_path / "follower"
    _write(primary / "x.py", "# nothing\n")
    _write(follower / "y.py", "OldClass\n")
    report = sc.sync_check(
        primary=primary,
        followers=[follower],
        removed_symbols=["OldClass"],
    )
    d = report.to_dict()
    assert d["removed_symbols"] == ["OldClass"]
    assert len(d["findings"]) == 1
    assert d["findings"][0]["symbol"] == "OldClass"
    assert d["findings"][0]["line_no"] == 1
