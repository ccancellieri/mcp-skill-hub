"""feedback_to_teachings.py — seed feedback_*.md files from the auto-memory
directory into the skill-hub SQLite DB as persistent teachings.

Usage:
    python scripts/feedback_to_teachings.py [--no-dry-run] [--memory-path PATH]

Default: dry-run mode (prints what would be stored, makes no changes).
--no-dry-run: actually store teachings to DB.
--memory-path PATH: override default memory directory path.
"""

from __future__ import annotations

import argparse
import re
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

_DEFAULT_MEMORY_PATH = (
    Path.home()
    / ".claude"
    / "projects"
    / "-Users-ccancellieri-work-code"
    / "memory"
)

_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n?(.*)$", re.DOTALL)
_WHY_RE = re.compile(r"\*\*Why:\*\*\s*(.+?)(?=\n\n|\n\*\*|\Z)", re.DOTALL)

_TARGET_TYPE = "global"
_TARGET_ID = "global"


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_frontmatter(text: str) -> dict[str, str]:
    """Parse simple YAML-like frontmatter into a dict (string values only)."""
    result: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            result[key.strip()] = value.strip()
    return result


def _parse_feedback_file(path: Path) -> dict[str, str] | None:
    """Parse a feedback_*.md file and return a dict with keys:
    rule, why, description, name.

    Returns None if the file cannot be parsed or yields no usable content.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"  WARN: cannot read {path.name}: {exc}", file=sys.stderr)
        return None

    text = text.strip()
    if not text:
        print(f"  WARN: {path.name} is empty — skipping", file=sys.stderr)
        return None

    fm_match = _FRONTMATTER_RE.match(text)
    if fm_match:
        fm_raw = fm_match.group(1)
        body = fm_match.group(2).strip()
        fm = _parse_frontmatter(fm_raw)
    else:
        fm = {}
        body = text

    name = fm.get("name") or path.stem
    description = fm.get("description") or ""

    # Rule = full body content, capped at 500 chars
    rule = body[:500] if body else description[:500]
    if not rule:
        print(f"  WARN: {path.name} has no usable rule content — skipping", file=sys.stderr)
        return None

    # Why = text after **Why:** section, fallback to frontmatter description
    why = ""
    why_match = _WHY_RE.search(body)
    if why_match:
        why = why_match.group(1).strip()
    if not why:
        why = description

    return {
        "name": name,
        "rule": rule,
        "why": why,
        "description": description,
    }


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def run(
    memory_dir: Path,
    *,
    dry_run: bool = True,
) -> int:
    """Seed teachings from feedback_*.md files in *memory_dir*.

    Returns the count of teachings added (or that would be added in dry-run).
    """
    if not memory_dir.exists():
        print(f"ERROR: memory directory not found at {memory_dir}", file=sys.stderr)
        sys.exit(1)

    feedback_files = sorted(memory_dir.glob("feedback_*.md"))
    if not feedback_files:
        print(f"No feedback_*.md files found in {memory_dir}")
        return 0

    parsed: list[dict[str, str]] = []
    for path in feedback_files:
        result = _parse_feedback_file(path)
        if result is not None:
            parsed.append(result)

    if not parsed:
        print("No parseable feedback files found.")
        return 0

    print(f"Found {len(parsed)} feedback file(s) to convert:")
    for item in parsed:
        print(f"  - {item['name']}: {item['rule'][:60]!r}...")

    if dry_run:
        print(
            f"\n[DRY-RUN] Would store {len(parsed)} teaching(s). "
            "Pass --no-dry-run to apply."
        )
        return len(parsed)

    # --- Live mode ---
    if not _SKILL_HUB_AVAILABLE or SkillStore is None:
        print(
            "ERROR: skill_hub package not importable. "
            "Run from the project root with uv run or after installing the package.",
            file=sys.stderr,
        )
        sys.exit(1)

    store = SkillStore()

    added = 0
    for item in parsed:
        rule = item["rule"]
        why = item["why"]
        name = item["name"]

        # Build the action text from why context
        action = why if why else f"apply rule: {name}"

        # Embed the rule text
        rule_vector: list[float] = []
        if embed_available():
            try:
                rule_vector = embed(rule)
            except Exception as exc:
                print(f"  WARN: embedding failed for {name!r}: {exc} — saving without vector")

        if not rule_vector:
            # add_teaching requires rule_vector; use a zero vector as fallback
            from skill_hub.store import VEC_DIM  # type: ignore[import]
            rule_vector = [0.0] * VEC_DIM

        teaching_id = store.add_teaching(
            rule=rule,
            rule_vector=rule_vector,
            action=action,
            target_type=_TARGET_TYPE,
            target_id=_TARGET_ID,
        )
        print(f"  Teaching #{teaching_id}: {name!r}")
        added += 1

    print(f"\nConverted {added} feedback file(s) to teachings.")
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
        "--memory-path",
        dest="memory_path",
        type=Path,
        default=_DEFAULT_MEMORY_PATH,
        help=f"Path to memory directory containing feedback_*.md files "
             f"(default: {_DEFAULT_MEMORY_PATH}).",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    dry_run = not args.no_dry_run
    if dry_run:
        print("[DRY-RUN MODE] No changes will be made.\n")
    run(memory_dir=args.memory_path, dry_run=dry_run)


if __name__ == "__main__":
    main()
