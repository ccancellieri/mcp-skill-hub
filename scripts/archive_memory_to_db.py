"""archive_memory_to_db.py — archive DONE/SHIPPED/DEFERRED/COMPLETE/CLOSED memory
entries from a MEMORY.md index file into the skill-hub SQLite DB as closed tasks.

Usage:
    python scripts/archive_memory_to_db.py [--no-dry-run] [--memory-path PATH]

Default: dry-run mode (prints what would happen, makes no changes).
--no-dry-run: actually perform the archival.
--memory-path PATH: override default MEMORY.md path.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Make the skill_hub package importable when running this script directly.
# Tests monkey-patch SkillStore / embed / embed_available at module level,
# so these must be top-level names (not buried inside run()).
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    from skill_hub.store import SkillStore
    from skill_hub.embeddings import embed, embed_available
    _SKILL_HUB_AVAILABLE = True
except ImportError:
    SkillStore = None  # type: ignore[assignment,misc]
    embed = None  # type: ignore[assignment]
    embed_available = None  # type: ignore[assignment]
    _SKILL_HUB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ARCHIVE_STATUS_RE = re.compile(
    r"\b(DONE|SHIPPED|DEFERRED|COMPLETE|CLOSED)\b", re.IGNORECASE
)

# Fix 3: anchor description match to require status word at start of description
_ARCHIVE_STATUS_LEADING_RE = re.compile(
    r'^(DONE|SHIPPED|DEFERRED|COMPLETE|ARCHIVED|CLOSED)\b', re.IGNORECASE
)

_ENTRY_RE = re.compile(
    r"^- \[([^\]]+)\]\(([^\)]+)\)(?:\s+[—–\-]+\s+(.+))?$"
)

_DEFAULT_MEMORY_PATH = (
    Path.home()
    / ".claude"
    / "projects"
    / "-Users-ccancellieri-work-code"
    / "memory"
    / "MEMORY.md"
)

_ARCHIVE_SUBDIR = "_archive"

_TAGS = "archived,auto-migration"
_SESSION_ID = "archive-migration-script"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_archiveable_by_description(description: str) -> bool:
    """Return True if the inline description starts with a status marker (Fix 3: anchored)."""
    return bool(_ARCHIVE_STATUS_LEADING_RE.match(description.strip()))


def _is_archiveable_by_body(body: str) -> bool:
    """Return True if the body text of the .md file contains a status marker.

    The marker must appear either:
    - at the very start of a line (optionally preceded by whitespace), or
    - in the first heading of the file (a line starting with # or ##).
    """
    for line in body.splitlines():
        stripped = line.strip()
        if _ARCHIVE_STATUS_RE.match(stripped):
            return True
        if stripped.startswith("#") and _ARCHIVE_STATUS_RE.search(stripped):
            return True
    return False


def _parse_entries(memory_text: str) -> list[dict]:
    """Parse MEMORY.md bullet entries.

    Returns a list of dicts:
        {
            "line_index": int,        # 0-based index in lines list
            "raw_line": str,          # original full line
            "title": str,             # link label
            "filename": str,          # link href (relative .md path)
            "description": str,       # text after the dash separator (may be "")
        }
    """
    entries: list[dict] = []
    for i, line in enumerate(memory_text.splitlines()):
        m = _ENTRY_RE.match(line.rstrip())
        if m:
            entries.append(
                {
                    "line_index": i,
                    "raw_line": line,
                    "title": m.group(1),
                    "filename": m.group(2),
                    "description": (m.group(3) or "").strip(),
                }
            )
    return entries


def _should_archive(entry: dict, memory_dir: Path) -> tuple[bool, str | None]:
    """Decide whether to archive an entry.

    Returns (should_archive, body_or_None).
    body is the file content when we managed to read it; None on skip.
    """
    # Fast path: description already has the marker
    if _is_archiveable_by_description(entry["description"]):
        md_path = memory_dir / entry["filename"]
        # Fix 4: if source file is already gone, skip to avoid duplicate DB entries
        if not md_path.exists():
            return False, None
        # Still read body for richer task summary
        body = md_path.read_text(encoding="utf-8")
        return True, body

    # Slower path: read the .md file
    md_path = memory_dir / entry["filename"]
    if not md_path.exists():
        return False, None

    body = md_path.read_text(encoding="utf-8")
    if _is_archiveable_by_body(body):
        return True, body

    return False, None


# ---------------------------------------------------------------------------
# Core archival logic
# ---------------------------------------------------------------------------


def run(
    memory_path: Path,
    *,
    dry_run: bool = True,
) -> int:
    """Archive entries from *memory_path*.

    Returns the count of entries archived (or that would be archived in dry-run).
    """
    if not memory_path.exists():
        print(f"ERROR: MEMORY.md not found at {memory_path}", file=sys.stderr)
        sys.exit(1)

    memory_dir = memory_path.parent
    memory_text = memory_path.read_text(encoding="utf-8")
    original_lines = memory_text.splitlines(keepends=True)
    original_row_count = len(
        [l for l in original_lines if _ENTRY_RE.match(l.rstrip())]
    )

    entries = _parse_entries(memory_text)

    # Determine which entries are archiveable
    to_archive: list[tuple[dict, str]] = []
    for entry in entries:
        should, body = _should_archive(entry, memory_dir)
        if should:
            to_archive.append((entry, body or ""))
        elif body is None and (memory_dir / entry["filename"]).exists() is False:
            # .md file is completely missing
            print(
                f"  WARN: {entry['filename']} not found on disk — skipping",
                file=sys.stderr,
            )

    if not to_archive:
        print("Nothing to archive.")
        return 0

    # Print plan
    print(f"Found {len(to_archive)} archiveable entr{'y' if len(to_archive)==1 else 'ies'}:")
    for entry, _ in to_archive:
        tag = ""
        if _is_archiveable_by_description(entry["description"]):
            tag = "(description match)"
        else:
            tag = "(body match)"
        print(f"  - {entry['filename']} {tag}")

    if dry_run:
        print("\n[DRY-RUN] No changes made. Pass --no-dry-run to apply.")
        return len(to_archive)

    # --- Live mode ---
    if not _SKILL_HUB_AVAILABLE or SkillStore is None:
        print(
            "ERROR: skill_hub package not importable. "
            "Run from the project root with uv run or after installing the package.",
            file=sys.stderr,
        )
        sys.exit(1)

    store = SkillStore()
    archive_dir = memory_dir / _ARCHIVE_SUBDIR
    archive_dir.mkdir(exist_ok=True)

    # Build the set of line indices to remove
    remove_line_indices: set[int] = set()

    archived_count = 0
    for entry, body in to_archive:
        # Build task title and summary
        title = entry["title"]
        description = entry["description"]
        summary = body if body else description

        # Embed
        vec: list[float] = []
        if embed_available():
            try:
                vec = embed(summary[:4000])
            except Exception as exc:
                print(f"  WARN: embedding failed for {title!r}: {exc} — saving without vector")

        # Save as open task, then close it immediately
        task_id = store.save_task(
            title=title,
            summary=summary,
            vector=vec,
            tags=_TAGS,
            session_id=_SESSION_ID,
        )
        store.close_task(
            task_id=task_id,
            compact=description or title,
            compact_vector=vec if vec else None,
        )
        print(f"  Archived task #{task_id}: {title}")

        # Fix 2: create parent directories before moving (handles subdirectory entries)
        # Fix 1: collect successful moves; only then update MEMORY.md
        md_path = memory_dir / entry["filename"]
        if md_path.exists():
            dest = archive_dir / entry["filename"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(md_path), str(dest))
            print(f"    Moved {entry['filename']} → {_ARCHIVE_SUBDIR}/{entry['filename']}")

        remove_line_indices.add(entry["line_index"])
        archived_count += 1

    # Fix 1: atomic MEMORY.md rewrite using temp file + os.replace()
    new_lines = [
        line
        for i, line in enumerate(original_lines)
        if i not in remove_line_indices
    ]
    new_content = "".join(new_lines)
    tmp = memory_path.with_suffix(".tmp")
    tmp.write_text(new_content, encoding="utf-8")
    os.replace(str(tmp), str(memory_path))

    final_entry_count = len(
        [l for l in new_lines if _ENTRY_RE.match(l.rstrip())]
    )
    print(
        f"\nArchived {archived_count} entr{'y' if archived_count == 1 else 'ies'}; "
        f"MEMORY.md shrank from {original_row_count} → {final_entry_count} rows."
    )
    return archived_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Archive DONE/SHIPPED/DEFERRED/COMPLETE/CLOSED memory entries "
                    "from MEMORY.md to the skill-hub SQLite DB."
    )
    p.add_argument(
        "--no-dry-run",
        dest="no_dry_run",
        action="store_true",
        default=False,
        help="Actually perform the archival (default: dry-run only).",
    )
    p.add_argument(
        "--memory-path",
        dest="memory_path",
        type=Path,
        default=_DEFAULT_MEMORY_PATH,
        help=f"Path to MEMORY.md (default: {_DEFAULT_MEMORY_PATH}).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    dry_run = not args.no_dry_run
    if dry_run:
        print("[DRY-RUN MODE] No changes will be made.\n")
    run(memory_path=args.memory_path, dry_run=dry_run)


if __name__ == "__main__":
    main()
