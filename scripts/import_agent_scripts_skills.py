"""Import steipete/agent-scripts skills into user-local Skill Hub skills.

Usage:
    python scripts/import_agent_scripts_skills.py --source PATH --dry-run
    python scripts/import_agent_scripts_skills.py --source PATH --apply

The importer reads an already-cloned repository from disk. It does not clone,
install, import, or execute anything from agent-scripts. Each upstream
``skills/<name>/`` directory is copied as a direct user skill directory named
``agent-scripts__<name>`` so existing local skills are not clobbered and the
Skill Hub indexer can derive one id per imported skill.
"""

from __future__ import annotations

import argparse
import filecmp
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


DEFAULT_DEST = Path.home() / ".claude" / "skills"
IMPORT_MARKER = "<!-- imported-from: steipete/agent-scripts -->"
DEST_PREFIX = "agent-scripts__"

_SECURITY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "credential workflow reference",
        re.compile(r"\b(1Password|op |api[_-]?key|password|secret|token)\b", re.I),
    ),
    ("git push workflow", re.compile(r"\bgit\s+push\b", re.I)),
    (
        "destructive git reset reference",
        re.compile(r"\bgit\s+reset\s+--hard\b", re.I),
    ),
    ("recursive removal reference", re.compile(r"\brm\s+-rf\b", re.I)),
    (
        "shell execution helper",
        re.compile(r"\b(os\.system|subprocess|child_process|spawn\()\b", re.I),
    ),
    (
        "browser or UI automation",
        re.compile(r"\b(Chrome|browser|screenshot|click|type)\b", re.I),
    ),
)


@dataclass
class ImportResult:
    written: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)
    overwrote: list[Path] = field(default_factory=list)
    skipped_foreign: list[Path] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    security_notes: dict[str, list[str]] = field(default_factory=dict)


def discover_skill_dirs(source_root: Path) -> list[Path]:
    """Return direct ``skills/<name>`` directories that contain ``SKILL.md``."""
    skills_root = source_root / "skills"
    if not skills_root.is_dir():
        return []
    return sorted(
        path
        for path in skills_root.iterdir()
        if path.is_dir() and (path / "SKILL.md").is_file()
    )


def security_notes_for_skill(skill_dir: Path) -> list[str]:
    """Classify sensitive guidance in the skill directory without executing it."""
    seen: list[str] = []
    for path in sorted(skill_dir.rglob("*")):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for label, pattern in _SECURITY_PATTERNS:
            if label not in seen and pattern.search(text):
                seen.append(label)
    return seen


def run(source_root: Path, dest_root: Path, *, dry_run: bool = False) -> ImportResult:
    result = ImportResult()
    source_root = source_root.expanduser()
    dest_root = dest_root.expanduser()

    if not source_root.exists():
        result.errors.append(f"source not found: {source_root}")
        return result
    if not (source_root / "skills").is_dir():
        result.errors.append(f"source has no skills directory: {source_root}")
        return result

    for skill_dir in discover_skill_dirs(source_root):
        slug = skill_dir.name
        target_dir = dest_root / f"{DEST_PREFIX}{slug}"
        target_skill = target_dir / "SKILL.md"
        notes = security_notes_for_skill(skill_dir)
        if notes:
            result.security_notes[slug] = notes

        if target_skill.exists() and not _is_owned(target_skill):
            result.skipped_foreign.append(target_skill)
            continue

        if target_dir.exists() and _dirs_match(skill_dir, target_dir):
            result.skipped.append(target_skill)
            continue

        if dry_run:
            if target_dir.exists():
                result.overwrote.append(target_skill)
            else:
                result.written.append(target_skill)
            continue

        if target_dir.exists():
            shutil.rmtree(target_dir)
            _copy_skill_dir(skill_dir, target_dir)
            result.overwrote.append(target_skill)
        else:
            _copy_skill_dir(skill_dir, target_dir)
            result.written.append(target_skill)

    return result


def _copy_skill_dir(source: Path, target: Path) -> None:
    shutil.copytree(source, target)
    skill_file = target / "SKILL.md"
    text = skill_file.read_text(encoding="utf-8", errors="replace")
    if IMPORT_MARKER not in text:
        skill_file.write_text(_insert_marker(text), encoding="utf-8")


def _insert_marker(text: str) -> str:
    frontmatter = re.match(r"^---\s*\n.*?\n---\s*\n?", text, re.DOTALL)
    marker_block = f"\n{IMPORT_MARKER}\n"
    if frontmatter:
        return (
            text[: frontmatter.end()]
            + marker_block
            + text[frontmatter.end():].lstrip("\n")
        )
    return f"{IMPORT_MARKER}\n\n{text}"


def _is_owned(skill_file: Path) -> bool:
    try:
        return IMPORT_MARKER in skill_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False


def _dirs_match(source: Path, target: Path) -> bool:
    if not target.exists():
        return False
    comparison = filecmp.dircmp(source, target, ignore=[])
    return _dircmp_equal(comparison)


def _dircmp_equal(comparison: filecmp.dircmp[str]) -> bool:
    if comparison.left_only or comparison.right_only or comparison.funny_files:
        return False
    for name in comparison.common_files:
        left = Path(comparison.left) / name
        right = Path(comparison.right) / name
        if name == "SKILL.md":
            left_text = _insert_marker(left.read_text(encoding="utf-8", errors="replace"))
            right_text = right.read_text(encoding="utf-8", errors="replace")
            if left_text != right_text:
                return False
            continue
        if not filecmp.cmp(left, right, shallow=False):
            return False
    return all(_dircmp_equal(sub) for sub in comparison.subdirs.values())


def _fmt(paths: list[Path]) -> str:
    return "\n  ".join(str(path) for path in paths) if paths else "(none)"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="import-agent-scripts-skills",
        description="Import steipete/agent-scripts skills into user-local skills.",
    )
    parser.add_argument(
        "--source", type=Path, required=True, help="Path to an agent-scripts checkout"
    )
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST, help="Destination root")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    parser.add_argument("--apply", action="store_true", help="Write files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    dry_run = args.dry_run or not args.apply
    result = run(args.source, args.dest, dry_run=dry_run)
    label = "[dry-run] " if dry_run else ""

    print(f"{label}written:\n  {_fmt(result.written)}")
    print(f"{label}skipped (unchanged):\n  {_fmt(result.skipped)}")
    print(f"{label}overwrote (updated):\n  {_fmt(result.overwrote)}")
    if result.skipped_foreign:
        print(f"skipped foreign (no import marker):\n  {_fmt(result.skipped_foreign)}")
    if result.security_notes:
        print("security notes:")
        for slug, notes in sorted(result.security_notes.items()):
            print(f"  {slug}: {', '.join(notes)}")
    if result.errors:
        print("errors:", file=sys.stderr)
        for error in result.errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
