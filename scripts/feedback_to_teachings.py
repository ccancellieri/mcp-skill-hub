"""feedback_to_teachings.py — seed feedback_*.md files from the auto-memory
directory into the skill-hub SQLite DB as persistent teachings.

Usage:
    uv run python scripts/feedback_to_teachings.py [--no-dry-run] \
        [--memory-dir PATH] [--db-path PATH]

Default: dry-run mode (prints what would be stored, makes no changes).
--no-dry-run: actually store teachings to DB.
--memory-dir PATH: override default memory directory path.
--db-path PATH: override default DB path.

The actual feedback_*.md parsing (frontmatter, rule, why, how-to-apply) is
not reimplemented here — it delegates to skill_hub.feedback_teachings, the
same primitives the continuous-teaching cron job uses, so this CLI and the
cron path can never drift apart on the file format.
"""

from __future__ import annotations

import argparse
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
    from skill_hub.feedback_teachings import build_action, parse_feedback_text
    _SKILL_HUB_AVAILABLE = True
except ImportError:
    SkillStore = None  # type: ignore[assignment,misc]
    embed = None  # type: ignore[assignment]
    embed_available = None  # type: ignore[assignment]
    parse_feedback_text = None  # type: ignore[assignment]
    build_action = None  # type: ignore[assignment]
    _SKILL_HUB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MEMORY_PATH = (
    Path.home()
    / ".claude"
    / "projects"
    / "-Users-ccancellieri-work-code"
    / "memory"
)

_TARGET_TYPE = "global"
_TARGET_ID = "global"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_feedback_file(path: Path) -> dict[str, str] | None:
    """Parse a feedback_*.md file and return a dict with keys:
    rule, why, how_to_apply, description, name, filename.

    Returns None if the file cannot be parsed or yields no usable content.
    The actual parse is skill_hub.feedback_teachings.parse_feedback_text;
    this wrapper only adds file I/O and the CLI's WARN reporting.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"  WARN: cannot read {path.name}: {exc}", file=sys.stderr)
        return None

    if not text.strip():
        print(f"  WARN: {path.name} is empty — skipping", file=sys.stderr)
        return None

    item = parse_feedback_text(text, name_fallback=path.stem)
    if item is None:
        print(f"  WARN: {path.name} has no usable rule content — skipping", file=sys.stderr)
        return None

    item["filename"] = path.name
    return item


def _build_action(item: dict[str, str]) -> str:
    """Build the action/context string from why + how_to_apply fields."""
    return build_action(item)


def _is_duplicate(existing_rules: list[str], candidate_rule: str) -> bool:
    """Return True if candidate_rule matches an existing teaching exactly."""
    return candidate_rule in existing_rules


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def run(
    memory_dir: Path,
    *,
    dry_run: bool = True,
    db_path: str | None = None,
) -> int:
    """Seed teachings from feedback_*.md files in *memory_dir*.

    Returns the count of teachings added (or that would be added in dry-run).
    """
    if not memory_dir.exists():
        print(f"ERROR: memory directory not found at {memory_dir}", file=sys.stderr)
        sys.exit(1)

    feedback_files = sorted(memory_dir.glob("feedback_*.md"))

    print("Feedback → Teachings Migration")
    print("================================")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print()
    print(f"Memory dir: {memory_dir}")

    if not feedback_files:
        print(f"No feedback_*.md files found in {memory_dir}")
        return 0

    print(f"Found {len(feedback_files)} feedback_*.md files")
    print()

    parsed: list[dict[str, str]] = []
    for path in feedback_files:
        result = _parse_feedback_file(path)
        if result is not None:
            parsed.append(result)

    if not parsed:
        print("No parseable feedback files found.")
        return 0

    # --- Duplicate detection ---
    # Fetch existing teachings unconditionally — read-only, safe in dry-run too.
    existing_rules: list[str] = []
    store: object | None = None

    if _SKILL_HUB_AVAILABLE and SkillStore is not None:
        try:
            _store_kwargs: dict = {}
            if db_path:
                _store_kwargs["db_path"] = Path(db_path)
            _probe_store = SkillStore(**_store_kwargs)  # type: ignore[call-arg]
            existing_rules = [row["rule"] for row in _probe_store.list_teachings()]  # type: ignore[union-attr]
            if dry_run:
                _probe_store.close()  # type: ignore[union-attr]
            else:
                store = _probe_store  # reuse in live mode
        except Exception as exc:
            print(f"  WARN: could not fetch existing teachings: {exc}", file=sys.stderr)

    if not dry_run:
        if not _SKILL_HUB_AVAILABLE or SkillStore is None:
            print(
                "ERROR: skill_hub package not importable. "
                "Run from the project root with uv run or after installing the package.",
                file=sys.stderr,
            )
            sys.exit(1)
        if store is None:
            # _probe_store failed; open a fresh store for writes
            store_kwargs: dict = {}
            if db_path:
                store_kwargs["db_path"] = Path(db_path)
            store = SkillStore(**store_kwargs)  # type: ignore[call-arg]

    to_insert: list[dict[str, str]] = []
    already_exists: list[dict[str, str]] = []
    errors: list[tuple[dict[str, str], str]] = []

    for item in parsed:
        if _is_duplicate(existing_rules, item["rule"]):
            already_exists.append(item)
        else:
            to_insert.append(item)

    # --- Print per-file status ---
    for item in parsed:
        filename = item["filename"]
        if item in already_exists:
            print(f"  ~ {filename:<60} [already exists, skip]")
        else:
            print(f"  ✓ {filename:<60} [new]")

    print()

    if dry_run:
        print(
            f"To insert: {len(to_insert)}  |  "
            f"Already exists: {len(already_exists)}  |  "
            f"Errors: 0"
        )
        print()
        print("(Pass --no-dry-run to apply.)")
        return len(to_insert)

    # --- Live mode: insert new teachings ---
    added = 0
    for item in to_insert:
        rule = item["rule"]
        action = _build_action(item)

        rule_vector: list[float] = []
        if embed_available():
            try:
                rule_vector = embed(rule)  # type: ignore[operator]
            except Exception as exc:
                print(f"  WARN: embedding failed for {item['name']!r}: {exc} — saving without vector")
        # Do NOT fall back to [0.0]*VEC_DIM — an empty list signals _mirror_teaching_vec
        # to skip the vec0 write, avoiding zero-vector poisoning of semantic search.

        try:
            teaching_id = store.add_teaching(  # type: ignore[union-attr]
                rule=rule,
                rule_vector=rule_vector,
                action=action,
                target_type=_TARGET_TYPE,
                target_id=_TARGET_ID,
            )
            added += 1
        except Exception as exc:
            errors.append((item, str(exc)))
            print(f"  ERROR: {item['filename']}: {exc}", file=sys.stderr)

    print(
        f"To insert: {len(to_insert)}  |  "
        f"Already exists: {len(already_exists)}  |  "
        f"Errors: {len(errors)}"
    )
    return added


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Seed feedback_*.md files as persistent teachings in the skill-hub DB."
    )
    p.add_argument(
        "--no-dry-run",
        dest="no_dry_run",
        action="store_true",
        default=False,
        help="Actually store teachings (default: dry-run only).",
    )
    p.add_argument(
        "--memory-dir",
        dest="memory_dir",
        type=Path,
        default=_DEFAULT_MEMORY_PATH,
        help=f"Path to memory directory containing feedback_*.md files "
             f"(default: {_DEFAULT_MEMORY_PATH}).",
    )
    p.add_argument(
        "--db-path",
        dest="db_path",
        default=None,
        help="Path to the skill-hub SQLite database (default: auto-detected from config).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    dry_run = not args.no_dry_run
    run(memory_dir=args.memory_dir, dry_run=dry_run, db_path=args.db_path)


if __name__ == "__main__":
    main()
