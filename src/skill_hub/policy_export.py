"""Per-repo POLICY.md export from auto-memory feedback_* files.

The maintainer's `feedback_*.md` memory rules live outside any repo (under
``~/.claude/projects/<slug>/memory/``). They drive behavior but are invisible to
anyone reading the repo. ``export_policies`` paraphrases those rules into a
durable ``<project>/.skill-hub/POLICY.md`` so the policy is discoverable to
humans and tooling without leaking AI-context paths into tracked content.

Hard rules enforced here (mirrors the no-AI-paths-in-tracked-content directive):
  - The rendered POLICY.md MUST NOT contain any occurrence of ``~/.claude/`` or
    ``.claude/`` or absolute paths under user homes.
  - Memory filename slugs (``feedback_<slug>.md``) become rule titles, never
    file path citations.
  - YAML frontmatter is stripped (its keys are indexer metadata, not policy).

Atomic file IO mirrors ``master_state._atomic_write``: tempfile + os.replace +
parent-dir fsync. Backups under ``.skill-hub/.backups/`` retain the last 10.

Idempotency: if POLICY.md mtime >= newest feedback file mtime, returns
``status="noop"`` without rewriting.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from .master_state import _project_to_memory_dir, _strip_frontmatter

_log = logging.getLogger(__name__)

_OUTPUT_REL = ".skill-hub/POLICY.md"
_BACKUP_SUBDIR = ".backups"
_BACKUP_RETENTION = 10

# Regex that strips any leakable AI-context reference from a line of paraphrased
# rule text. Mirrors the grep list in the global "No AI Paths In Tracked Files"
# rule. Order matters: longest patterns first so partial matches don't shadow.
_SCRUB_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"~/\.claude/[\w\-./]*"),
    re.compile(r"(?<!\w)\.claude/[\w\-./]*"),
    re.compile(r"/Users/[\w\-]+/\.claude/[\w\-./]*"),
    re.compile(r"~/\.claude\b"),
)

# YAML frontmatter parsing — same minimal parser as the indexer uses elsewhere.
# We extract ``name`` and ``description`` (the human-meaningful keys) and drop
# everything else.
_FRONTMATTER_RE = re.compile(r"^---\s*\n(?P<body>.*?)\n---\s*\n", re.DOTALL)


def _scrub_ai_paths(text: str) -> str:
    """Remove every ~/.claude / .claude path reference from `text`.

    Replacements collapse the path to ``[memory]`` to keep the surrounding
    sentence readable. Idempotent.
    """
    out = text
    for pat in _SCRUB_PATTERNS:
        out = pat.sub("[memory]", out)
    return out


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Return (frontmatter_dict, body) for a markdown string.

    The "dict" is a tolerant key:value scan — values are stripped, no nested
    YAML. Body is the text after the closing ``---``. If no frontmatter, the
    dict is empty and body == text.
    """
    m = _FRONTMATTER_RE.match(text)
    if not m:
        return {}, text
    body = text[m.end():]
    fm: dict[str, str] = {}
    for line in m.group("body").splitlines():
        if ":" not in line:
            continue
        k, _, v = line.partition(":")
        fm[k.strip()] = v.strip().strip('"').strip("'")
    return fm, body


def _atomic_write(path: Path, content: str) -> None:
    """Atomic write — same recipe as ``master_state._atomic_write``.

    Tempfile in same dir → fsync → os.replace → parent-dir fsync.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass
        os.replace(tmp_name, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except (OSError, AttributeError):
            pass
    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _prune_backups(backup_dir: Path, stem: str, retain: int = _BACKUP_RETENTION) -> int:
    if not backup_dir.exists():
        return 0
    candidates = sorted(
        backup_dir.glob(f"{stem}-*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    deleted = 0
    for old in candidates[retain:]:
        try:
            old.unlink()
            deleted += 1
        except OSError as exc:
            _log.warning("backup prune failed for %s: %s", old, exc)
    return deleted


def _list_feedback_files(memory_dir: Path) -> list[Path]:
    """Return all ``feedback_*.md`` files in `memory_dir`, sorted by name."""
    if not memory_dir.exists():
        return []
    return sorted(memory_dir.glob("feedback_*.md"), key=lambda p: p.name)


def _slug_to_title(stem: str) -> str:
    """Convert ``feedback_no_intermodule_deps`` -> ``No Intermodule Deps``.

    The leading ``feedback_`` prefix is dropped. Remaining underscores become
    spaces and each word is title-cased.
    """
    name = stem
    if name.startswith("feedback_"):
        name = name[len("feedback_"):]
    parts = [p for p in name.replace("-", "_").split("_") if p]
    return " ".join(p.capitalize() for p in parts) if parts else stem


def _render_rule_block(path: Path) -> str:
    """Render one feedback file as a paraphrased POLICY rule block.

    Output shape::

        ### <Title from frontmatter `name` or slug>

        <description from frontmatter, if present>

        <body, frontmatter stripped, AI-paths scrubbed>

    Returns "" if the file can't be read.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return ""

    fm, body = _parse_frontmatter(raw)
    title = fm.get("name") or _slug_to_title(path.stem)
    description = fm.get("description", "")

    # Scrub AI-paths from every textual surface we render.
    title = _scrub_ai_paths(title).strip()
    description = _scrub_ai_paths(description).strip()
    body = _scrub_ai_paths(body).strip()

    lines = [f"### {title}", ""]
    if description:
        lines.append(f"_{description}_")
        lines.append("")
    if body:
        lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_policy_md(project_name: str, dated: str, rule_blocks: list[str]) -> str:
    """Render the full POLICY.md document.

    The header explains that this file is a paraphrase of memory rules,
    so a reader who finds it in a repo understands why it exists.
    """
    header = [
        f"# POLICY — {project_name}",
        "",
        f"_Generated {dated} by `export_policies()` from local memory feedback rules._",
        "",
        "This file paraphrases maintainer feedback rules into a durable, in-repo policy",
        "document. The source rules live outside the repo; this is the discoverable",
        "form. Edits made here will be overwritten on the next export — change the",
        "source rule instead.",
        "",
        "---",
        "",
    ]
    if not rule_blocks:
        header.append("_No feedback rules found in the project's memory directory._")
        header.append("")
        return "\n".join(header).rstrip() + "\n"

    body = "\n".join(rule_blocks)
    out = "\n".join(header) + body
    # Final scrub pass — defensive: catches any path the per-block scrub missed
    # (e.g. paths that span a title/body boundary).
    out = _scrub_ai_paths(out)
    return out.rstrip() + "\n"


def export_policies(
    project_root: str,
    output_file: str = _OUTPUT_REL,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Render feedback_*.md memory files as a per-repo POLICY.md.

    Args:
        project_root: absolute or ``~``-expandable repo root.
        output_file: relative path inside ``project_root`` (default
            ``.skill-hub/POLICY.md``).
        dry_run: render but do not write; return rendered text in result.
        force: rewrite even if POLICY.md is fresher than every feedback file.

    Returns a dict with ``status`` in {written, noop, skipped, dry_run, empty}
    plus diagnostic fields. Never raises for missing memory dir — returns
    ``status="skipped"`` so callers can wire this into hooks safely.
    """
    root = Path(os.path.expanduser(project_root)).resolve()
    if not root.is_dir():
        return {"status": "skipped", "reason": f"project_root not a directory: {root}"}

    memory_dir = _project_to_memory_dir(root)
    if memory_dir is None:
        return {"status": "skipped", "reason": f"no auto-memory dir for {root}"}

    feedback_files = _list_feedback_files(memory_dir)
    output_path = root / output_file

    if not feedback_files:
        # Render a placeholder doc so a reader knows the mechanism exists and
        # is wired up — they just have no rules yet.
        rendered = _render_policy_md(root.name, _dt.date.today().isoformat(), [])
        if dry_run:
            return {
                "status": "empty",
                "rendered": rendered,
                "feedback_files_considered": [],
            }
        write_result = _upsert_policy(output_path, rendered)
        return {
            "status": "empty",
            "feedback_files_considered": [],
            **write_result,
        }

    # mtime-based no-op: if POLICY.md is newer than every feedback file, skip.
    if not force and not dry_run and output_path.exists():
        try:
            out_mtime = output_path.stat().st_mtime
            newest_fb_mtime = max(p.stat().st_mtime for p in feedback_files)
            if out_mtime >= newest_fb_mtime:
                return {
                    "status": "noop",
                    "reason": (
                        f"POLICY.md fresher than newest feedback file "
                        f"({_dt.datetime.fromtimestamp(out_mtime).isoformat()} >= "
                        f"{_dt.datetime.fromtimestamp(newest_fb_mtime).isoformat()})"
                    ),
                    "feedback_files_considered": [p.name for p in feedback_files],
                }
        except OSError:
            pass

    rule_blocks = [b for b in (_render_rule_block(p) for p in feedback_files) if b]
    rendered = _render_policy_md(root.name, _dt.date.today().isoformat(), rule_blocks)

    # Acceptance criterion: zero ~/.claude / .claude references in output.
    # If the scrubbers somehow left one through, fail loudly rather than
    # write a contract-violating file.
    if _has_ai_path_leak(rendered):
        return {
            "status": "skipped",
            "reason": "internal: scrubber left a ~/.claude reference in output (refusing to write)",
            "feedback_files_considered": [p.name for p in feedback_files],
        }

    if dry_run:
        return {
            "status": "dry_run",
            "rendered": rendered,
            "rendered_chars": len(rendered),
            "feedback_files_considered": [p.name for p in feedback_files],
        }

    write_result = _upsert_policy(output_path, rendered)
    return {
        "status": "written",
        "feedback_files_considered": [p.name for p in feedback_files],
        **write_result,
    }


def _has_ai_path_leak(text: str) -> bool:
    """Hard check: any of the patterns the global rule forbids in tracked content."""
    for pat in _SCRUB_PATTERNS:
        if pat.search(text):
            return True
    return False


def _upsert_policy(output_path: Path, rendered: str) -> dict[str, Any]:
    """Atomic write with backup + bounded retention. Returns diag dict."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing = output_path.read_text(encoding="utf-8") if output_path.exists() else ""

    backup_path: Path | None = None
    pruned = 0
    if existing:
        backup_dir = output_path.parent / _BACKUP_SUBDIR
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        backup_path = backup_dir / f"{output_path.stem}-{ts}{output_path.suffix}"
        _atomic_write(backup_path, existing)
        pruned = _prune_backups(backup_dir, output_path.stem)

    _atomic_write(output_path, rendered)
    return {
        "wrote": str(output_path),
        "backup": str(backup_path) if backup_path else None,
        "backups_pruned": pruned,
        "new_chars": len(rendered),
        "delta_chars": len(rendered) - len(existing),
    }
