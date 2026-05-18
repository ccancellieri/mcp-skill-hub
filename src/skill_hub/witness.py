"""witness-log: append-only fix manifest per repo (M1 / issue #10).

Captures structured "what fix shipped where" records to a JSONL file so the
dashboard (and humans) can see a real fix history instead of relying on memory
files. Append-only — records are never edited or removed by this module.

Storage
-------
Shares the existing ``witness_log.jsonl`` file used by ``lint_canary`` under
``~/.claude/mcp-skill-hub/state/``. The two record kinds coexist by the ``kind``
field: ``"fix"`` for entries written here, ``"lint_canary"`` for the canary
cadence.

Record shape
------------
Each fix record contains::

    {
      "kind":        "fix",
      "at":          1715990400,        # epoch seconds, integer
      "issue":       "#10",             # caller-supplied issue ref
      "pr":          "#42",             # caller-supplied PR ref
      "sha":         "abc1234",         # commit sha (short or full)
      "repo":        "ccancellieri/mcp-skill-hub",
      "fix_kind":    "feat",            # caller-supplied kind label
      "fix_summary": "free-form summary"
    }

(``kind`` is the JSONL record discriminator — what *type* of witness entry this
is. ``fix_kind`` is the caller's conventional-commit / change-type label.)

This module is intentionally dependency-free (only stdlib). The JSONL output is
parseable by ``json.loads`` per line — no custom encoding.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .lint_canary import witness_log_path

WITNESS_KIND = "fix"


@dataclass(frozen=True)
class WitnessRecord:
    """One fix-manifest entry."""

    issue: str
    pr: str
    sha: str
    repo: str
    fix_kind: str
    fix_summary: str
    at: int

    def to_record(self) -> dict:
        return {
            "kind": WITNESS_KIND,
            "at": self.at,
            "issue": self.issue,
            "pr": self.pr,
            "sha": self.sha,
            "repo": self.repo,
            "fix_kind": self.fix_kind,
            "fix_summary": self.fix_summary,
        }


class AppendOnlyError(RuntimeError):
    """Raised when caller attempts to mutate a witness record in-place.

    Append-only is the load-bearing contract; we surface attempts to edit a
    record as a hard failure so callers don't silently corrupt history.
    """


def record_witness(
    issue: str,
    pr: str,
    sha: str,
    repo: str,
    fix_kind: str = "fix",
    fix_summary: str = "",
    witness_file: Path | None = None,
    now: int | None = None,
) -> WitnessRecord:
    """Append a fix-manifest entry to the witness log.

    Parameters
    ----------
    issue:       Issue reference (e.g. ``"#10"`` or ``"GH-10"``).
    pr:          PR reference (e.g. ``"#42"``).
    sha:         Commit sha (short or full).
    repo:        Repo identifier — ``"owner/name"`` is conventional.
    fix_kind:    Caller-supplied kind label (feat / fix / refactor / ...).
    fix_summary: Free-form one-line description.
    witness_file: Override target path for hermetic tests.
    now:         Override timestamp (epoch seconds) for hermetic tests.

    Returns the persisted ``WitnessRecord``. Raises ``OSError`` if the path
    cannot be written (the caller decides whether to swallow it).
    """
    if not repo:
        raise ValueError("repo is required")
    record = WitnessRecord(
        issue=str(issue).strip(),
        pr=str(pr).strip(),
        sha=str(sha).strip(),
        repo=str(repo).strip(),
        fix_kind=str(fix_kind).strip() or "fix",
        fix_summary=str(fix_summary).strip(),
        at=int(now if now is not None else time.time()),
    )
    wp = witness_file or witness_log_path()
    wp.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record.to_record()) + "\n"
    # Append-only: never open with "w" / "r+" / truncate. ``"a"`` plus
    # O_APPEND semantics give us atomic append on POSIX for line-sized writes.
    with wp.open("a", encoding="utf-8") as fh:
        fh.write(line)
        try:
            fh.flush()
            os.fsync(fh.fileno())
        except (OSError, AttributeError):
            pass
    return record


def edit_witness(*_args, **_kwargs):
    """Editing existing witness records is forbidden.

    This entry point exists so the append-only contract is discoverable: any
    caller reaching for "edit" sees an explicit failure rather than finding no
    such function and writing their own mutator.
    """
    raise AppendOnlyError(
        "witness log is append-only — existing records cannot be edited. "
        "Append a new record via record_witness() instead."
    )


def _iter_records(witness_file: Path) -> Iterator[dict]:
    """Yield every parseable JSON object in the witness log.

    Malformed lines and non-dict objects are skipped silently — the log is
    append-only and may have trailing partial writes after a crash; we don't
    want one bad line to hide the rest.
    """
    if not witness_file.exists():
        return
    try:
        text = witness_file.read_text(encoding="utf-8")
    except OSError:
        return
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            yield obj


def list_witness(
    repo: str | None = None,
    since: int | None = None,
    limit: int | None = None,
    witness_file: Path | None = None,
) -> list[dict]:
    """Return fix-manifest records, newest-first.

    Parameters
    ----------
    repo:  Optional ``"owner/name"`` filter. Exact match against the record's
           ``repo`` field. ``None`` returns all repos.
    since: Optional minimum epoch-seconds (inclusive). Records with ``at < since``
           are skipped. ``None`` returns all timestamps.
    limit: Optional max records to return (after filtering + sort).
    witness_file: Override target path for hermetic tests.

    Only records with ``kind == "fix"`` are returned — other kinds (e.g.
    ``"lint_canary"``) coexist in the same JSONL but are not fix-manifest
    entries and are filtered out here.
    """
    wp = witness_file or witness_log_path()
    matches: list[dict] = []
    for rec in _iter_records(wp):
        if rec.get("kind") != WITNESS_KIND:
            continue
        if repo is not None and rec.get("repo") != repo:
            continue
        if since is not None:
            try:
                if int(rec.get("at", 0)) < int(since):
                    continue
            except (TypeError, ValueError):
                continue
        matches.append(rec)
    # Newest-first ordering.
    matches.sort(key=lambda r: int(r.get("at", 0)), reverse=True)
    if limit is not None and limit > 0:
        matches = matches[:limit]
    return matches


def format_witness_list(records: Iterable[dict]) -> str:
    """Human-readable rendering for the MCP tool return value."""
    rows = list(records)
    if not rows:
        return "witness: no records match."
    lines = [f"witness: {len(rows)} record(s)"]
    for r in rows:
        head = (
            f"  [{r.get('at')}] {r.get('repo','?')} "
            f"{r.get('fix_kind','?')} {r.get('issue','?')} "
            f"pr={r.get('pr','?')} sha={r.get('sha','?')}"
        )
        summary = (r.get("fix_summary") or "").strip()
        if summary:
            head += f" — {summary}"
        lines.append(head)
    return "\n".join(lines)
