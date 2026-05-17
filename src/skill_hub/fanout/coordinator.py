"""Fan-out coordinator — turns N issues into N worktree-bound skill-hub tasks.

Reuses existing primitives:
    worktree.ensure_worktree(project, name)  — creates worktree, no session launch
    store.save_task(..., worktree=spec.to_json(), tags="fanout:<gid> src:<source>")
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from .directive import DispatchSpec, render_directive
from .prompt_synth import draft_prompt
from .sources import Issue, get_source


@dataclass
class FanoutResult:
    group_id: str
    task_ids: list[int] = field(default_factory=list)
    worktree_paths: list[str] = field(default_factory=list)
    directive: str = ""
    prompt_qualities: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)


def _slug_for_issue(issue: Issue, *, max_len: int = 40) -> str:
    """Worktree/branch-friendly slug. Includes issue number when present."""
    title = (issue.title or "issue").lower()
    slug = re.sub(r"[^a-z0-9._\-]+", "-", title)
    slug = re.sub(r"-{2,}", "-", slug).strip("-.")
    slug = slug[:max_len] or "issue"
    # Prefix gh:NNN issues with the number for human recognition.
    if issue.source in ("gh", "github") and ":" in issue.id:
        num = issue.id.split(":", 1)[1]
        return f"issue-{num}-{slug}"[:max_len + 12]
    return slug


def _default_limit() -> int:
    try:
        from .. import config as _cfg
        n = (_cfg.load_config().get("fanout") or {}).get("default_limit", 3)
    except Exception:  # noqa: BLE001
        n = 3
    return int(n or 3)


def fanout(
    source: str,
    *,
    filter: str = "",
    limit: int | None = None,
    project: str = "",
    repo: str = "",
    dry_run: bool = False,
    use_llm: bool = True,
    store: object | None = None,
) -> FanoutResult:
    """Prep N parallel worktrees + tasks for one source's issues.

    Parameters
    ----------
    source: "gh" | "text" | configured custom name
    filter: source-specific filter (gh search query, raw bullet text, ...)
    limit:  max issues to fan out; defaults to fanout.default_limit (3)
    project: skill-hub project name (resolved under worktree.repo_roots)
    repo:   passed to sources that support it (gh: "owner/name")
    dry_run: don't create worktrees / tasks — return what *would* happen
    use_llm: when False, every prompt uses the deterministic fallback
    store:   optional SkillStore instance (for testing); else a fresh one
    """
    from .. import worktree as _wt

    if not project:
        raise ValueError("fanout requires a `project` (skill-hub project name).")
    if limit is None:
        limit = _default_limit()

    src = get_source(source)
    issues = src.fetch(filter=filter, limit=limit, repo=repo)
    if not issues:
        gid = uuid.uuid4().hex[:8]
        return FanoutResult(group_id=gid, directive=render_directive([], gid))

    # Resolve project to a real repo path so prompt synth can read .github/...
    repo_path = _wt.resolve_project(project)

    group_id = uuid.uuid4().hex[:8]
    if store is None:
        from ..store import SkillStore
        store = SkillStore()

    specs: list[DispatchSpec] = []
    result = FanoutResult(group_id=group_id)

    for issue in issues:
        try:
            prompt_text, quality = draft_prompt(
                issue, repo_path,
                store_conn=getattr(store, "_conn", None),
                use_llm=use_llm,
            )
        except Exception as e:  # noqa: BLE001 — fall back to title+body
            prompt_text = f"{issue.title}\n\n{issue.body}".strip() or issue.title
            quality = f"error:{e.__class__.__name__}"

        slug = _slug_for_issue(issue)
        tags = " ".join([
            f"fanout:{group_id}",
            f"src:{issue.source}",
            f"issue:{issue.id}",
        ])
        title = f"{project}: {issue.title}"[:120] or f"{project}/{slug}"

        if dry_run:
            specs.append(DispatchSpec(
                description=f"Issue {issue.id}: {issue.title[:60]}",
                prompt=prompt_text,
                worktree_path=str(Path(repo_path) / ".claude" / "worktrees" / slug),
                branch=f"cc/{slug}",
                task_id=0,
            ))
            result.task_ids.append(0)
            result.worktree_paths.append(specs[-1].worktree_path)
            result.prompt_qualities.append(quality)
            continue

        try:
            spec = _wt.ensure_worktree(project, slug)
        except _wt.WorktreeError as e:
            result.skipped.append({"issue": issue.id, "reason": str(e)})
            continue

        try:
            from ..embeddings import embed
            try:
                vector = embed(f"{title}: {issue.body}")
            except RuntimeError:
                vector = []
        except Exception:  # noqa: BLE001
            vector = []

        task_id = store.save_task(  # type: ignore[attr-defined]
            title=title,
            summary=issue.body or issue.title,
            vector=vector,
            context=prompt_text,
            tags=tags,
            session_id="fanout",
            cwd=spec.worktree_path,
            branch=spec.branch,
            worktree=spec.to_json(),
            color="cyan",
        )
        result.task_ids.append(task_id)
        result.worktree_paths.append(spec.worktree_path)
        result.prompt_qualities.append(quality)
        specs.append(DispatchSpec(
            description=f"Issue {issue.id}: {issue.title[:60]}",
            prompt=prompt_text,
            worktree_path=spec.worktree_path,
            branch=spec.branch,
            task_id=task_id,
        ))

    result.directive = render_directive(specs, group_id)
    return result


__all__ = ["FanoutResult", "fanout"]
