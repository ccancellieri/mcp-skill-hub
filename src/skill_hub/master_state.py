"""Master Project State compaction — file IO, rendering, and upsert.

Companion to `embeddings.compact_master_state` (the LLM-driven JSON producer).
This module handles:
  - Resolving the auto-memory directory for a project root.
  - Listing recent project_*.md / feedback_*.md files.
  - Reading the existing snapshot section.
  - Rendering the JSON output to Markdown.
  - Upserting under a `## <title>` heading with a backup on overwrite.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

# How many days of auto-memory to fold in by default.
_DEFAULT_WINDOW_DAYS = 7
# Where backups land, relative to the output file's parent.
_BACKUP_SUBDIR = ".backups"
# Maximum number of backups to retain per file (oldest are pruned).
_BACKUP_RETENTION = 10


def _atomic_write(path: Path, content: str) -> None:
    """Write `content` to `path` atomically: temp file in same dir + os.replace.

    Same-directory tempfile guarantees the rename is atomic on POSIX (a single
    inode swap, no cross-filesystem fallback). Crash mid-write leaves either
    the old file intact or the new file complete — never a truncated one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp_name, path)
    except Exception:
        # Best-effort cleanup of orphaned tempfile on failure.
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def _prune_backups(backup_dir: Path, stem: str, retain: int = _BACKUP_RETENTION) -> int:
    """Keep only the `retain` most recent backups for a given stem; delete the rest.

    Returns the count of files deleted.
    """
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


def _project_to_memory_dir(project_root: Path) -> Path | None:
    """Map a project root to its auto-memory directory under ~/.claude/projects/.

    The auto-memory layout in this codebase encodes the cwd path as a slug:
      ~/work/code              -> -Users-<user>-work-code
      ~/work/code/geoid        -> -Users-<user>-work-code-geoid (if dedicated)
    We look for the longest matching prefix folder under ~/.claude/projects/.
    """
    home = Path.home()
    base = home / ".claude" / "projects"
    if not base.exists():
        return None

    target = project_root.resolve()
    # Convert /Users/foo/work/code/geoid -> -Users-foo-work-code-geoid
    slug = "-" + str(target).strip("/").replace("/", "-")

    # Try exact match first, then walk up to find a parent slug that exists.
    candidates = [slug]
    parent = target.parent
    while parent != parent.parent:
        candidates.append("-" + str(parent).strip("/").replace("/", "-"))
        parent = parent.parent

    for slug_try in candidates:
        candidate = base / slug_try / "memory"
        if candidate.exists():
            return candidate
    return None


def _list_recent_memory(memory_dir: Path, since: _dt.datetime) -> list[Path]:
    """List project_*.md and feedback_*.md modified since `since`."""
    out: list[Path] = []
    for pattern in ("project_*.md", "feedback_*.md"):
        for path in memory_dir.glob(pattern):
            try:
                mtime = _dt.datetime.fromtimestamp(path.stat().st_mtime)
                if mtime >= since:
                    out.append(path)
            except OSError:
                continue
    out.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return out


def _read_existing_section(file_path: Path, section_title: str) -> str:
    """Read the body of a `## <section_title>` heading. Empty string if not found."""
    if not file_path.exists():
        return ""
    text = file_path.read_text(encoding="utf-8")
    pattern = rf"(?ms)^##\s+{re.escape(section_title)}\b.*?(?=^##\s|\Z)"
    m = re.search(pattern, text)
    return m.group(0).strip() if m else ""


def _summarize_memory_entries(paths: list[Path], char_budget: int = 6000) -> str:
    """Concatenate first ~10 lines of each recent memory file, capped at budget."""
    chunks: list[str] = []
    used = 0
    for p in paths:
        try:
            head = "\n".join(p.read_text(encoding="utf-8").splitlines()[:12])
            entry = f"### {p.name}\n{head}\n"
        except OSError:
            continue
        if used + len(entry) > char_budget:
            break
        chunks.append(entry)
        used += len(entry)
    return "\n".join(chunks)


def _render_markdown(payload: dict[str, Any], section_title: str, dated: str) -> str:
    """Render the LLM JSON payload as a Markdown section."""
    lines: list[str] = [f"## {section_title} — {dated}", ""]
    arch = payload.get("architecture") or ""
    if arch.strip():
        lines.append("### Current Architecture (Finalized)")
        lines.append("")
        lines.append(arch.strip())
        lines.append("")

    invs = payload.get("invariants") or []
    if invs:
        lines.append("### Global Invariants (Always / Never)")
        lines.append("")
        for i, inv in enumerate(invs, start=1):
            text = str(inv).strip()
            if not re.match(r"^\d+\.", text):
                text = f"{i}. {text}"
            lines.append(text)
        lines.append("")

    mods = payload.get("active_modules") or []
    if mods:
        lines.append("### Active Working Set")
        lines.append("")
        for i, mod in enumerate(mods, start=1):
            name = mod.get("name", f"module-{i}")
            para = mod.get("paragraph", "").strip()
            lines.append(f"{i}. **{name}** — {para}")
        lines.append("")

    pivots = payload.get("recent_pivots") or []
    if pivots:
        lines.append("### Recent Pivot Log")
        lines.append("")
        for i, p in enumerate(pivots, start=1):
            date = p.get("date", "")
            title = p.get("title", "")
            trig = p.get("trigger", "")
            dec = p.get("decision", "")
            why = p.get("why", "")
            lines.append(f"{i}. **{date} — {title}**")
            if trig:
                lines.append(f"   - Trigger: {trig}")
            if dec:
                lines.append(f"   - Decision: {dec}")
            if why:
                lines.append(f"   - Why: {why}")
        lines.append("")

    assumptions = payload.get("assumptions") or []
    if assumptions:
        lines.append("### Assumptions to Verify")
        lines.append("")
        lines.append("> These are claims the compactor inferred but did not directly observe.")
        lines.append("> Verify each before acting on it; demote to invariants when confirmed,")
        lines.append("> drop when refuted.")
        lines.append("")
        for i, a in enumerate(assumptions, start=1):
            claim = a.get("claim", "").strip() if isinstance(a, dict) else str(a).strip()
            verify = a.get("verify_by", "").strip() if isinstance(a, dict) else ""
            lines.append(f"{i}. {claim}")
            if verify:
                lines.append(f"   - Verify by: `{verify}`")
        lines.append("")

    if payload.get("_fallback"):
        lines.append("> _LLM compaction returned an empty payload; existing snapshot preserved if present._")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _append_assumptions_to_inbox(project_root: Path, assumptions: list[Any]) -> Path | None:
    """Append assumptions to `<project>/.memory/inbox.md` so they don't get lost.

    Inbox is the canonical place for "inferred-not-confirmed" items in this codebase
    (per `feedback_inbox_pattern`). Each entry is dated and marked with the source.
    """
    if not assumptions:
        return None
    inbox = project_root / ".memory" / "inbox.md"
    inbox.parent.mkdir(parents=True, exist_ok=True)
    today = _dt.date.today().isoformat()
    block = [f"\n## {today} — Master State compaction assumptions\n"]
    for a in assumptions:
        if isinstance(a, dict):
            claim = a.get("claim", "").strip()
            verify = a.get("verify_by", "").strip()
            block.append(f"- **Claim**: {claim}")
            if verify:
                block.append(f"  - **Verify by**: `{verify}`")
        else:
            block.append(f"- {str(a).strip()}")
    block.append("")
    existing = inbox.read_text(encoding="utf-8") if inbox.exists() else "# Inbox — Inferred Items Awaiting Confirmation\n\nItems the agent inferred but has not yet confirmed with the user.\nReview periodically: confirm (move to decisions.md / patterns.md) or reject (delete).\n\n---\n"
    inbox.write_text(existing + "\n".join(block), encoding="utf-8")
    return inbox


def _upsert_section(
    file_path: Path,
    section_title: str,
    new_section: str,
    retain_backups: int = _BACKUP_RETENTION,
) -> dict[str, Any]:
    """Replace `## <section_title>` block (if present) or prepend to file.

    Atomic write (temp file + os.replace), backup on overwrite, bounded
    backup retention (oldest deleted past `retain_backups`).
    """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    existing = file_path.read_text(encoding="utf-8") if file_path.exists() else ""

    # Backup before overwrite.
    backup_path: Path | None = None
    pruned = 0
    if existing:
        backup_dir = file_path.parent / _BACKUP_SUBDIR
        backup_dir.mkdir(parents=True, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")
        backup_path = backup_dir / f"{file_path.stem}-{ts}{file_path.suffix}"
        _atomic_write(backup_path, existing)
        pruned = _prune_backups(backup_dir, file_path.stem, retain=retain_backups)

    pattern = rf"(?ms)^##\s+{re.escape(section_title)}\b.*?(?=^##\s|\Z)"
    if re.search(pattern, existing):
        new_text = re.sub(pattern, new_section, existing, count=1)
    else:
        # Prepend after a leading H1 if present, else at the very top.
        h1_match = re.match(r"(?m)^#\s.*?\n", existing)
        if h1_match:
            insert_at = h1_match.end()
            new_text = existing[:insert_at] + "\n" + new_section + "\n" + existing[insert_at:]
        else:
            new_text = new_section + "\n" + existing

    _atomic_write(file_path, new_text)
    return {
        "wrote": str(file_path),
        "backup": str(backup_path) if backup_path else None,
        "backups_pruned": pruned,
        "new_chars": len(new_text),
        "delta_chars": len(new_text) - len(existing),
    }


def compact_to_master_state(
    project_root: str,
    output_file: str = ".memory/decisions.md",
    section_title: str = "Master Project State",
    since_iso: str | None = None,
    always_keep_path: str = ".skillhub/master_state.yaml",
    dry_run: bool = False,
    window_days: int = _DEFAULT_WINDOW_DAYS,
) -> dict[str, Any]:
    """End-to-end: read recent memory, call LLM, render, upsert.

    Returns a dict describing what was done. Never raises for missing memory dir;
    returns `{"status": "skipped", "reason": "..."}` instead.
    """
    from .embeddings import compact_master_state as _llm_compact

    root = Path(os.path.expanduser(project_root)).resolve()
    if not root.is_dir():
        return {"status": "skipped", "reason": f"project_root not a directory: {root}"}

    memory_dir = _project_to_memory_dir(root)
    if memory_dir is None:
        return {"status": "skipped", "reason": f"no auto-memory dir for {root}"}

    if since_iso:
        try:
            since = _dt.datetime.fromisoformat(since_iso)
        except ValueError:
            since = _dt.datetime.now() - _dt.timedelta(days=window_days)
    else:
        since = _dt.datetime.now() - _dt.timedelta(days=window_days)

    recent = _list_recent_memory(memory_dir, since)
    if not recent:
        return {"status": "noop", "reason": "no recent auto-memory in window"}

    output_path = root / output_file

    # Cost-saver: if the existing snapshot is fresher than the newest memory
    # entry, the LLM has nothing new to fold in. Skip without paying for the
    # call. `dry_run=True` overrides — the user explicitly asked for a preview.
    if not dry_run and output_path.exists():
        try:
            snap_mtime = output_path.stat().st_mtime
            newest_mem_mtime = max(p.stat().st_mtime for p in recent)
            if snap_mtime >= newest_mem_mtime:
                return {
                    "status": "noop",
                    "reason": (
                        f"existing snapshot is fresher than newest memory entry "
                        f"(snapshot {_dt.datetime.fromtimestamp(snap_mtime).isoformat()}, "
                        f"newest mem {_dt.datetime.fromtimestamp(newest_mem_mtime).isoformat()})"
                    ),
                }
        except OSError:
            pass  # If stat fails, fall through and let the LLM run.

    existing_snapshot = _read_existing_section(output_path, section_title)
    memory_summary = _summarize_memory_entries(recent)

    always_keep_text = ""
    always_keep_full = root / always_keep_path
    if always_keep_full.exists():
        try:
            always_keep_text = always_keep_full.read_text(encoding="utf-8")
        except OSError:
            pass

    payload = _llm_compact(
        existing_snapshot=existing_snapshot,
        task_summaries="",  # MCP tool layer can wire task summaries when called from close_task
        memory_entries=memory_summary,
        always_keep=always_keep_text,
        window_days=window_days,
    )

    dated = _dt.date.today().isoformat()
    rendered = _render_markdown(payload, section_title, dated)

    if dry_run:
        return {
            "status": "dry_run",
            "rendered_chars": len(rendered),
            "rendered": rendered,
            "memory_files_considered": [str(p) for p in recent[:10]],
            "had_existing": bool(existing_snapshot),
        }

    write_result = _upsert_section(output_path, section_title, rendered)
    inbox_path = _append_assumptions_to_inbox(root, payload.get("assumptions") or [])
    return {
        "status": "written",
        "memory_files_considered": [str(p.name) for p in recent[:10]],
        "had_existing": bool(existing_snapshot),
        "assumptions_count": len(payload.get("assumptions") or []),
        "inbox": str(inbox_path) if inbox_path else None,
        **write_result,
    }
