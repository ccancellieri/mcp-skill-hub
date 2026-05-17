"""import_ruflo_skills.py — one-shot importer that converts ruflo skill manifests
into skill-hub native SKILL.md files.

Usage:
    python scripts/import_ruflo_skills.py [--ruflo-root PATH] [--dest PATH] [--dry-run]

Constraints (issue m4):
    - Read-only filesystem access against the ruflo install.
    - No runtime import of ruflo / claude-flow code.
    - Lives under scripts/ so it is NOT loaded by the MCP server.
    - Idempotent: re-running with no changes is a no-op (no writes, no churn).

The script auto-detects the ruflo install in this order:
    1. --ruflo-root argument
    2. $RUFLO_ROOT env var
    3. ~/.claude-flow/
    4. ~/.claude/plugins/cache/ruflo/  (where ruflo plugin caches actually land)

Output layout:
    <dest>/imported_ruflo/<plugin>__<skill>/SKILL.md
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_DEST = Path.home() / ".skill_hub" / "skills"

CANDIDATE_RUFLO_ROOTS: tuple[Path, ...] = (
    Path.home() / ".claude-flow",
    Path.home() / ".claude" / "plugins" / "cache" / "ruflo",
)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
# Tolerant field parser: handles "key: value" lines, ignores indented continuations.
_FM_FIELD_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$")

# Marker line inserted into imported SKILL.md so future imports can detect
# files we own and avoid clobbering hand-authored skills with the same name.
IMPORT_MARKER = "<!-- imported-from: ruflo -->"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


@dataclass
class RufloSkill:
    plugin: str          # e.g. "ruflo-swarm"
    skill_name: str      # e.g. "swarm-init" (slug, from directory)
    source_path: Path    # absolute path to source SKILL.md
    frontmatter: dict[str, str]
    body: str            # everything after frontmatter

    @property
    def dest_dirname(self) -> str:
        return f"{self.plugin}__{self.skill_name}"


@dataclass
class ImportResult:
    written: list[Path]
    skipped: list[Path]    # destinations whose content already matched
    overwrote: list[Path]  # destinations that existed but had different content
    skipped_foreign: list[Path]  # destinations existing without IMPORT_MARKER
    errors: list[str]


# ---------------------------------------------------------------------------
# Detection + parsing
# ---------------------------------------------------------------------------


def detect_ruflo_root(
    explicit: Path | None = None,
    *,
    env: dict[str, str] | None = None,
    candidates: tuple[Path, ...] = CANDIDATE_RUFLO_ROOTS,
) -> Path | None:
    """Resolve which ruflo install to read from. Returns None if not found.

    Precedence: explicit arg > $RUFLO_ROOT > first existing candidate.
    """
    env = env if env is not None else dict(os.environ)
    if explicit is not None:
        return explicit if explicit.is_dir() else None
    env_root = env.get("RUFLO_ROOT")
    if env_root:
        p = Path(env_root)
        return p if p.is_dir() else None
    for c in candidates:
        if c.is_dir():
            return c
    return None


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse a SKILL.md style YAML frontmatter block.

    Returns (fields_dict, body_text). If no frontmatter, returns ({}, text).
    Only top-level scalar `key: value` lines are captured — that matches
    what ruflo emits and matches the indexer's own parser in
    src/skill_hub/indexer.py.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    block = m.group(1)
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        # Skip indented continuation lines (block scalars / nested keys).
        if line[:1] in (" ", "\t"):
            continue
        fm = _FM_FIELD_RE.match(line)
        if not fm:
            continue
        key, raw = fm.group(1), fm.group(2).strip()
        if not raw:
            # Empty value almost always means a block scalar / nested key with
            # the actual content on continuation lines. Skip — it would just
            # pollute the dict with a useless empty string.
            continue
        # Strip surrounding quotes if present.
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in ("'", '"'):
            raw = raw[1:-1]
        fields[key] = raw
    return fields, text[m.end():]


def _infer_plugin(path: Path) -> str:
    """Derive a plugin id from the source path.

    Heuristic that works for both layouts we know about:
      .../ruflo-core/0.2.1/skills/<skill>/SKILL.md  -> "ruflo-core"
      .../plugins/ruflo-swarm/skills/<skill>/SKILL.md -> "ruflo-swarm"
    Falls back to "ruflo" if no plugin segment is found.
    """
    parts = path.parts
    try:
        idx = parts.index("skills")
    except ValueError:
        return "ruflo"
    # Walk back over version-looking and generic segments to find the plugin name.
    for seg in reversed(parts[:idx]):
        if re.fullmatch(r"v?\d+(\.\d+){1,3}", seg):  # version like 0.2.1
            continue
        if seg in ("plugins", "cache", "marketplaces", ".claude", "ruflo"):
            continue
        return seg
    return "ruflo"


def discover_skill_files(root: Path) -> list[Path]:
    """Return every SKILL.md under a `skills/<name>/` directory below root."""
    found: list[Path] = []
    for p in root.rglob("SKILL.md"):
        # Require the parent-of-parent to be named "skills" — that filters
        # out e.g. agent definitions or top-level READMEs that happen to be
        # named SKILL.md.
        if p.parent.parent.name == "skills":
            found.append(p)
    return sorted(found)


def load_skill(path: Path) -> RufloSkill | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    fm, body = parse_frontmatter(text)
    # Note: we deliberately do NOT synthesize `name` from the directory when
    # frontmatter is missing — `run()` uses an empty `name`/`description` as
    # the signal that the source file has no manifest worth importing.
    slug = path.parent.name
    plugin = _infer_plugin(path)
    return RufloSkill(
        plugin=plugin,
        skill_name=slug,
        source_path=path,
        frontmatter=dict(fm),
        body=body,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


# Frontmatter fields we preserve verbatim. Anything else is dropped to keep the
# imported manifest lean — the indexer only consumes `name` + `description`.
_PRESERVED_FIELDS = ("name", "description", "argument-hint", "allowed-tools")


def render_skill(skill: RufloSkill) -> str:
    """Render a skill-hub native SKILL.md for an imported ruflo skill.

    The output is canonical (stable field order, single trailing newline) so
    re-running the importer with no source changes produces byte-identical
    output — that's what makes the importer idempotent.
    """
    lines: list[str] = ["---"]
    for k in _PRESERVED_FIELDS:
        v = skill.frontmatter.get(k)
        if v:
            lines.append(f"{k}: {v}")
    # Provenance — kept inside frontmatter so it survives downstream parsers
    # but uses a custom key the indexer will simply ignore.
    lines.append(f"source: ruflo:{skill.plugin}:{skill.skill_name}")
    lines.append("---")
    lines.append("")
    lines.append(IMPORT_MARKER)
    lines.append("")
    body = skill.body.lstrip("\n").rstrip() + "\n"
    lines.append(body)
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------------------
# Idempotent write
# ---------------------------------------------------------------------------


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _is_skill_hub_owned(existing: str) -> bool:
    """True if the existing file looks like a previous import of ours.

    We only overwrite files that carry IMPORT_MARKER — that way a hand-authored
    skill under the same dest path is preserved (and reported as a foreign
    conflict).
    """
    return IMPORT_MARKER in existing


def write_skill(
    skill: RufloSkill,
    dest_root: Path,
    *,
    dry_run: bool = False,
    result: ImportResult,
) -> None:
    target_dir = dest_root / "imported_ruflo" / skill.dest_dirname
    target = target_dir / "SKILL.md"
    new_content = render_skill(skill)

    if target.exists():
        try:
            old = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            result.errors.append(f"read failed: {target}: {e}")
            return
        if _sha256(old) == _sha256(new_content):
            result.skipped.append(target)
            return
        if not _is_skill_hub_owned(old):
            result.skipped_foreign.append(target)
            return
        if dry_run:
            result.overwrote.append(target)
            return
        target.write_text(new_content, encoding="utf-8")
        result.overwrote.append(target)
        return

    if dry_run:
        result.written.append(target)
        return
    target_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(new_content, encoding="utf-8")
    result.written.append(target)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run(
    ruflo_root: Path | None,
    dest_root: Path,
    *,
    dry_run: bool = False,
) -> ImportResult:
    result = ImportResult(written=[], skipped=[], overwrote=[],
                          skipped_foreign=[], errors=[])
    root = detect_ruflo_root(ruflo_root)
    if root is None:
        result.errors.append(
            "no ruflo install found "
            f"(checked {[str(c) for c in CANDIDATE_RUFLO_ROOTS]})"
        )
        return result

    for src in discover_skill_files(root):
        skill = load_skill(src)
        if skill is None:
            result.errors.append(f"parse failed: {src}")
            continue
        if not skill.frontmatter.get("name") and not skill.frontmatter.get("description"):
            # No metadata at all — surface as an error but don't write
            # something useless.
            result.errors.append(f"no frontmatter: {src}")
            continue
        write_skill(skill, dest_root, dry_run=dry_run, result=result)
    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="import-ruflo-skills",
        description="Import ruflo skill manifests into skill-hub native format.",
    )
    p.add_argument(
        "--ruflo-root",
        type=Path,
        default=None,
        help="Path to a ruflo install (default: auto-detect)",
    )
    p.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help="Output root (default: ~/.skill_hub/skills)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without writing.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = run(args.ruflo_root, args.dest, dry_run=args.dry_run)

    def _fmt(paths: list[Path]) -> str:
        return "\n  ".join(str(p) for p in paths) if paths else "(none)"

    label = "[dry-run] " if args.dry_run else ""
    print(f"{label}written:\n  {_fmt(result.written)}")
    print(f"{label}skipped (unchanged):\n  {_fmt(result.skipped)}")
    print(f"{label}overwrote (updated):\n  {_fmt(result.overwrote)}")
    if result.skipped_foreign:
        print(f"skipped foreign (no import marker, left untouched):\n  "
              f"{_fmt(result.skipped_foreign)}")
    if result.errors:
        print("errors:", file=sys.stderr)
        for e in result.errors:
            print(f"  {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
