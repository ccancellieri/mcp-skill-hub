"""Detect what to export: projects, tables, and PII guards.

All functions are pure / stdlib-only and accept paths so tests can pass
fixtures without touching ``~/.claude/``.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Tables in skill_hub.db that hold ephemeral / per-host runtime state and
# must NOT be exported (they're regenerated automatically and would just
# bloat the snapshot).
EPHEMERAL_TABLES: frozenset[str] = frozenset({
    "session_log",
    "triage_log",
    "interceptions",
    "context_injections",
    "response_cache",
    "error_cache",
    "conversation_state",
    "session_context",
    "skill_injections",
    "plugin_migrations",
})

# Tokens that indicate PII / private content. Any markdown file containing
# any of these (case-insensitive substring) is flagged. The list mirrors the
# heuristic used in the legacy ``export_geoid_context.sh`` and can be
# extended via ``scan_for_pii(..., extra_tokens=...)``.
DEFAULT_PII_TOKENS: tuple[str, ...] = (
    "glicemia",
    "sovereign",
    "ollama",
    "ssh_key",
    "ssh-key",
    "medical_llm",
    "telegram",
)

DEFAULT_CLAUDE_PROJECTS_ROOT = Path.home() / ".claude" / "projects"


@dataclass(frozen=True)
class ProjectInfo:
    key: str
    path: Path
    top_level_md_count: int
    private_md_count: int
    has_memory_index: bool

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "path": str(self.path),
            "top_level_md_count": self.top_level_md_count,
            "private_md_count": self.private_md_count,
            "has_memory_index": self.has_memory_index,
        }


def list_projects(root: Path | None = None) -> list[ProjectInfo]:
    """List every ``<root>/<project-key>/memory/`` directory.

    ``root`` defaults to ``~/.claude/projects/``. Each entry counts top-level
    ``*.md`` files (export candidates) and any markdown under ``private/``
    (always excluded from exports — counted only for the UI to display
    "Has private/: yes/no" hints).
    """
    base = root or DEFAULT_CLAUDE_PROJECTS_ROOT
    if not base.exists():
        return []
    out: list[ProjectInfo] = []
    for project_dir in sorted(base.iterdir()):
        if not project_dir.is_dir():
            continue
        memory_dir = project_dir / "memory"
        if not memory_dir.is_dir():
            continue
        top_level = [p for p in memory_dir.glob("*.md") if p.is_file()]
        private_dir = memory_dir / "private"
        private_files = (
            list(private_dir.rglob("*.md")) if private_dir.is_dir() else []
        )
        out.append(
            ProjectInfo(
                key=project_dir.name,
                path=memory_dir,
                top_level_md_count=len(top_level),
                private_md_count=len(private_files),
                has_memory_index=(memory_dir / "MEMORY.md").exists(),
            )
        )
    return out


def list_exportable_tables(conn: sqlite3.Connection) -> list[str]:
    """Return user-table names from ``conn`` minus ephemeral & virtual tables.

    Virtual tables (e.g. ``CREATE VIRTUAL TABLE … USING vec0(…)``) require
    their backing extension to be loaded, can't be SELECTed otherwise, and
    are typically rebuilt from the underlying source data — so we always
    skip them. Tables whose creator (``sqlite_master.sql``) starts with
    ``CREATE VIRTUAL TABLE`` are excluded. Shadow tables (suffixes used
    internally by virtual-table modules: ``_chunks``, ``_rowids``,
    ``_vector_chunks*``) are also excluded.
    """
    cur = conn.execute(
        "SELECT name, COALESCE(sql, '') FROM sqlite_master "
        "WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    )
    virtual_names: set[str] = set()
    rows = cur.fetchall()
    for name, sql in rows:
        if "CREATE VIRTUAL TABLE" in (sql or "").upper():
            virtual_names.add(name)

    out: list[str] = []
    for name, _sql in rows:
        if name in EPHEMERAL_TABLES:
            continue
        if name in virtual_names:
            continue
        # Shadow tables created by virtual-table modules — usually named
        # ``<vt>_<suffix>`` where ``<vt>`` is a virtual table.
        if any(name.startswith(f"{vt}_") for vt in virtual_names):
            continue
        out.append(name)
    return out


# Matches a markdown link/list entry that points at a private/* path.
# Examples it catches:
#   - [foo](private/foo.md) - one-line desc
#   - [bar](./private/bar.md)
#   - * private/baz.md - whatever
_PRIVATE_LINK_RE = re.compile(
    r"^.*?(?:\(|\b)\.?/?private/[^)\s]+\.md.*$",
    flags=re.IGNORECASE,
)


def filter_memory_index(text: str) -> str:
    """Drop lines from a ``MEMORY.md`` body that point to ``private/`` files.

    Header-only sections that become empty after filtering are also dropped.
    Idempotent: re-running on the result returns the same string.
    """
    kept: list[str] = []
    for line in text.splitlines():
        if _PRIVATE_LINK_RE.match(line):
            continue
        # Strip standalone "Private Projects" section headers.
        if re.match(r"^\s*#+\s*private\s+projects\s*$", line, re.IGNORECASE):
            continue
        kept.append(line)
    return "\n".join(kept).rstrip() + ("\n" if text.endswith("\n") else "")


def scan_for_pii(
    paths: Iterable[Path],
    extra_tokens: Iterable[str] = (),
) -> list[Path]:
    """Return the subset of ``paths`` whose contents contain any PII token.

    Tokens are matched case-insensitively as plain substrings (not regex).
    ``paths`` should already exclude ``private/`` files — this is a guardrail
    for the *exportable* set.
    """
    tokens = tuple(t.lower() for t in (*DEFAULT_PII_TOKENS, *extra_tokens))
    offenders: list[Path] = []
    for p in paths:
        try:
            text = p.read_text(encoding="utf-8", errors="ignore").lower()
        except OSError:
            continue
        if any(tok in text for tok in tokens):
            offenders.append(p)
    return offenders


def exportable_md_files(memory_dir: Path) -> list[Path]:
    """Top-level ``*.md`` files in ``memory_dir`` (private/ excluded)."""
    if not memory_dir.is_dir():
        return []
    return sorted(p for p in memory_dir.glob("*.md") if p.is_file())
