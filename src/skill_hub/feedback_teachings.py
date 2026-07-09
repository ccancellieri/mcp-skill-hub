"""Promote feedback_*.md memory files into persistent teachings.

feedback_*.md files are confirmed rules written to Claude's auto-memory:
YAML frontmatter carrying ``name``/``description``, with a body holding the
rule text and optional ``**Why:**`` / ``**How to apply:**`` sections. This
module scans them and inserts each as a row in the ``teachings`` table via
the same embed() + ``store.add_teaching()`` flow the ``teach`` MCP tool uses
(see ``server.teach``), so a converted rule is matched semantically the same
way an explicitly taught one is.

Called from ``cron._feedback_to_teachings_handler`` when
``continuous_teaching_enabled`` is set (default off).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

from .embeddings import embed, embed_available
from .store import SkillStore

_log = logging.getLogger(__name__)

_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)
_WHY_RE = re.compile(r"\*\*Why:\*\*\s*(.+?)(?=\n\n|\n\*\*|\Z)", re.DOTALL)
_HOW_RE = re.compile(r"\*\*How to apply:\*\*\s*(.+?)(?=\n\n|\n\*\*|\Z)", re.DOTALL)

# Converted feedback rules are not tied to a specific skill/plugin.
_TARGET_TYPE = "global"
_TARGET_ID = "global"

_RULE_MAX_LEN = 500
_ACTION_MAX_LEN = 300


def _parse_frontmatter(raw: str) -> dict[str, str]:
    fm: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            fm[key.strip()] = value.strip()
    return fm


def _parse_feedback_file(path: Path) -> dict[str, str] | None:
    """Extract a teaching-ready rule/action from one feedback_*.md file.

    Returns ``None`` if the file is missing, empty, or has no usable content.
    """
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        _log.warning("feedback_to_teachings: cannot read %s: %s", path, exc)
        return None
    if not text:
        return None

    match = _FRONTMATTER_RE.match(text)
    if match:
        fm = _parse_frontmatter(match.group(1))
        body = match.group(2).strip()
    else:
        fm = {}
        body = text

    description = fm.get("description", "")
    rule = (body or description)[:_RULE_MAX_LEN]
    if not rule:
        return None

    why_match = _WHY_RE.search(body)
    why = (why_match.group(1).strip() if why_match else description)[:_ACTION_MAX_LEN]

    how_match = _HOW_RE.search(body)
    how_to_apply = how_match.group(1).strip()[:_ACTION_MAX_LEN] if how_match else ""

    return {
        "name": fm.get("name") or path.stem,
        "rule": rule,
        "why": why,
        "how_to_apply": how_to_apply,
    }


def _build_action(item: dict[str, str]) -> str:
    parts = []
    if item["why"]:
        parts.append(f"Why: {item['why']}")
    if item["how_to_apply"]:
        parts.append(f"How to apply: {item['how_to_apply']}")
    return "\n".join(parts) if parts else f"apply rule: {item['name']}"


def _normalize_rule(text: str) -> str:
    """Collapse whitespace/case so near-identical rule text still dedups."""
    return " ".join(text.split()).lower()


def convert(store: SkillStore, memory_root: Path) -> dict[str, int]:
    """Scan feedback_*.md files under *memory_root* and promote new ones.

    De-duplicates against existing teaching rule text (normalized), so
    re-running the job never inserts the same rule twice. When embedding is
    unavailable or fails for a file, the teaching is still inserted with an
    empty vector — ``store.add_teaching`` / ``_mirror_teaching_vec`` already
    treat that as a supported degraded insert (skips the vec0 mirror only).

    Returns counts: ``found``, ``converted``, ``duplicates``, ``unparsed``,
    ``errors``.
    """
    files = sorted(memory_root.rglob("feedback_*.md")) if memory_root.exists() else []
    existing = {_normalize_rule(row["rule"]) for row in store.list_teachings()}

    converted = 0
    duplicates = 0
    unparsed = 0
    errors = 0

    for path in files:
        item = _parse_feedback_file(path)
        if item is None:
            unparsed += 1
            continue

        norm_rule = _normalize_rule(item["rule"])
        if norm_rule in existing:
            duplicates += 1
            continue

        rule_vector: list[float] = []
        if embed_available():
            try:
                rule_vector = embed(item["rule"])
            except Exception as exc:  # noqa: BLE001 — degrade to vectorless insert
                _log.warning(
                    "feedback_to_teachings: embedding failed for %s: %s",
                    path.name, exc,
                )

        try:
            store.add_teaching(
                rule=item["rule"],
                rule_vector=rule_vector,
                action=_build_action(item),
                target_type=_TARGET_TYPE,
                target_id=_TARGET_ID,
            )
        except Exception as exc:  # noqa: BLE001 — one bad file must not abort the run
            errors += 1
            _log.warning("feedback_to_teachings: insert failed for %s: %s", path.name, exc)
            continue

        existing.add(norm_rule)
        converted += 1

    return {
        "found": len(files),
        "converted": converted,
        "duplicates": duplicates,
        "unparsed": unparsed,
        "errors": errors,
    }
