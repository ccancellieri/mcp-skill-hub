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

    # Resolve project to a real repo path up-front so sources that shell out
    # (e.g. GitHubSource → `gh issue list`) can chdir into the checkout
    # instead of inheriting an unrelated cwd that isn't a git repo.
    repo_path = _wt.resolve_project(project)

    src = get_source(source)
    try:
        issues = src.fetch(filter=filter, limit=limit, repo=repo, cwd=str(repo_path))
    except TypeError:
        # Backwards-compat: a custom-registered source may not accept `cwd`.
        issues = src.fetch(filter=filter, limit=limit, repo=repo)
    if not issues:
        gid = uuid.uuid4().hex[:8]
        return FanoutResult(group_id=gid, directive=render_directive([], gid))

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
            repo=spec.project,
        )

        # Typed task↔issue link (issue #37): extract numeric issue number from
        # the scoped id (e.g. "gh:123" → 123, "text:0001" → 1). Skip if the id
        # does not carry a parseable integer — non-GitHub sources may not have
        # numeric GitHub issue numbers.
        if issue.source in ("gh", "github") and ":" in issue.id:
            _num_str = issue.id.split(":", 1)[1]
            try:
                _issue_num = int(_num_str)
                store.link_task_issue(  # type: ignore[attr-defined]
                    task_id,
                    _issue_num,
                    repo=repo or "",
                    url=issue.url or None,
                )
            except (ValueError, Exception):  # noqa: BLE001
                pass  # non-fatal: the freeform tag still exists

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


@dataclass
class CleanupResult:
    group_id: str
    closed_task_ids: list[int] = field(default_factory=list)
    removed_worktrees: list[str] = field(default_factory=list)
    deleted_branches: list[str] = field(default_factory=list)
    skipped: list[dict] = field(default_factory=list)


def fanout_cleanup(
    group_id: str,
    *,
    close_open_tasks: bool = True,
    remove_worktrees: bool = True,
    delete_branches: bool = True,
    summary: str = "",
    store: object | None = None,
) -> CleanupResult:
    """Tear down every task + worktree + branch in a fanout group.

    The bulk inverse of :func:`fanout`. Idempotent: safe to call after a
    partial cleanup. Closed tasks are still inspected so their worktrees /
    branches can be removed.
    """
    from .. import worktree as _wt

    if store is None:
        from ..store import SkillStore
        store = SkillStore()

    rows = store.list_tasks(status="all", tag=f"fanout:{group_id}")  # type: ignore[attr-defined]
    result = CleanupResult(group_id=group_id)
    if not rows:
        return result

    digest = (summary or f"fanout group {group_id} cleanup").strip()

    for r in rows:
        row = dict(r)
        task_id = row.get("id")

        if close_open_tasks and row.get("status") == "open":
            try:
                if store.close_task(task_id, compact=digest):  # type: ignore[attr-defined]
                    result.closed_task_ids.append(task_id)
            except Exception as e:  # noqa: BLE001
                result.skipped.append({"task_id": task_id, "stage": "close", "reason": str(e)})

        full = store.get_task(task_id)  # type: ignore[attr-defined]
        spec_blob = dict(full).get("worktree") if full else None
        if not spec_blob or not (remove_worktrees or delete_branches):
            continue

        try:
            spec = _wt.WorktreeSpec.from_json(spec_blob)
        except Exception as e:  # noqa: BLE001
            result.skipped.append({"task_id": task_id, "stage": "parse_spec", "reason": str(e)})
            continue

        worktree_existed = Path(spec.worktree_path).exists()
        try:
            _wt.teardown_worktree(spec, delete_branch=delete_branches)
            if remove_worktrees and worktree_existed:
                result.removed_worktrees.append(spec.worktree_path)
            if delete_branches:
                result.deleted_branches.append(spec.branch)
        except Exception as e:  # noqa: BLE001
            result.skipped.append({"task_id": task_id, "stage": "teardown", "reason": str(e)})

    return result


__all__ = ["FanoutResult", "CleanupResult", "fanout", "fanout_cleanup"]
