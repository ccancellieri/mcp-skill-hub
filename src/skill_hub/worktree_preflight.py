"""Worktree pre-flight collision check.

Encodes the worktree-naming-collision rule (M3-1) as a callable check rather
than a memory-rule the maintainer re-reads at session start.

Given an issue number and a target project, report whether starting a fresh
worktree-bound task on that issue is safe:

- existing worktrees under ``<repo>/.claude/worktrees/`` matching
  ``issue-<num>-*``
- existing local branches matching ``cc/issue-<num>-*``
- open GitHub PRs whose head branch starts with ``cc/issue-<num>-`` (best
  effort via ``gh``; silently skipped if ``gh`` is missing / unauthenticated /
  the repo can't be resolved)

The check is intended to be sub-second:

- local checks ``git worktree list --porcelain`` and ``git branch --list``
  (typically tens of ms);
- two best-effort ``gh`` calls (``issue view`` and ``pr list``) with a small
  hard timeout (default 1.5 s each). Missing/slow ``gh`` does not turn a
  clean-state into a collision — it surfaces a note in ``warnings`` and the
  rest of the report stands.

Returns a :class:`PreflightResult` plus a human-readable formatter so callers
(MCP tool, CLI) get a one-shot string.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from . import worktree as _wt

# Hard cap on each `gh` invocation. Keeps total cost well under a second
# even if the network is flaky; if `gh` exceeds it we emit a warning and
# treat that signal as unknown rather than failing the call.
_GH_TIMEOUT_S = 1.5


@dataclass
class PreflightResult:
    """Structured outcome of :func:`preflight`."""
    issue_number: int
    project: str
    repo_path: str
    issue_prefix: str            # "issue-<num>-"
    branch_prefix: str           # "cc/issue-<num>-"
    worktrees: list[str] = field(default_factory=list)   # absolute paths
    branches: list[str] = field(default_factory=list)    # local branch names
    pull_requests: list[dict] = field(default_factory=list)
    issue_title: str = ""
    issue_state: str = ""        # "OPEN" / "CLOSED" / "" if unknown
    warnings: list[str] = field(default_factory=list)

    @property
    def safe(self) -> bool:
        """No collisions on any of the three axes."""
        return not (self.worktrees or self.branches or self.pull_requests)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["safe"] = self.safe
        return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _normalize_issue_number(issue_number: int | str) -> int:
    """Coerce ``issue_number`` to a positive int; raise on garbage."""
    if isinstance(issue_number, str):
        s = issue_number.strip().lstrip("#")
        try:
            issue_number = int(s)
        except ValueError as e:
            raise ValueError(
                f"issue_number must be a positive integer, got {issue_number!r}"
            ) from e
    if not isinstance(issue_number, int) or issue_number <= 0:
        raise ValueError(
            f"issue_number must be a positive integer, got {issue_number!r}"
        )
    return issue_number


def _list_matching_worktrees(repo: Path, prefix: str) -> list[str]:
    """Return absolute paths of worktrees whose final segment starts with ``prefix``."""
    registered = _wt._registered_worktrees(repo)
    hits: list[str] = []
    for path in registered:
        name = Path(path).name
        if name.startswith(prefix):
            hits.append(path)
    return sorted(hits)


def _list_matching_branches(repo: Path, prefix: str) -> list[str]:
    """Return local branches matching ``<prefix>*`` via ``git branch --list``."""
    res = subprocess.run(
        ["git", "-C", str(repo), "branch", "--list", f"{prefix}*"],
        capture_output=True, text=True, check=False,
    )
    out: list[str] = []
    for line in (res.stdout or "").splitlines():
        # `git branch --list` prefixes the current branch with "* " and a
        # branch that's checked out in another worktree with "+ ".
        name = line.lstrip(" *+").strip()
        if name and name.startswith(prefix):
            out.append(name)
    return sorted(out)


def _gh_available() -> bool:
    return shutil.which("gh") is not None


def _run_gh(args: list[str]) -> tuple[int, str, str]:
    """Run a `gh` command with a hard timeout. Returns (rc, stdout, stderr)."""
    try:
        res = subprocess.run(
            args, capture_output=True, text=True, check=False,
            timeout=_GH_TIMEOUT_S,
        )
        return res.returncode, res.stdout or "", res.stderr or ""
    except subprocess.TimeoutExpired:
        return 124, "", f"gh timed out after {_GH_TIMEOUT_S}s"
    except FileNotFoundError:
        return 127, "", "gh not found"


def _fetch_issue_meta(issue_number: int, repo: str) -> tuple[str, str, list[str]]:
    """Return (title, state, warnings) for `gh issue view`. Best effort."""
    args = ["gh", "issue", "view", str(issue_number),
            "--json", "title,state"]
    if repo:
        args += ["--repo", repo]
    rc, out, err = _run_gh(args)
    warnings: list[str] = []
    if rc == 0:
        try:
            data = json.loads(out or "{}")
            return (
                str(data.get("title") or ""),
                str(data.get("state") or ""),
                warnings,
            )
        except json.JSONDecodeError as e:
            warnings.append(f"gh issue view returned invalid JSON: {e}")
            return "", "", warnings
    stderr = (err or "").strip()
    low = stderr.lower()
    if "not authenticated" in low or "authentication" in low:
        warnings.append("gh not authenticated — issue / PR collision check skipped.")
    elif "could not resolve" in low or "no default remote" in low \
            or "not a git repository" in low:
        warnings.append("gh could not resolve a repository — pass repo='owner/name'.")
    elif rc == 124:
        warnings.append(stderr or "gh issue view timed out")
    else:
        warnings.append(f"gh issue view failed (rc={rc}): {stderr[:160]}")
    return "", "", warnings


def _fetch_open_prs(branch_prefix: str, repo: str) -> tuple[list[dict], list[str]]:
    """Return (open PRs whose head branch starts with ``branch_prefix``, warnings)."""
    # `gh pr list --search "head:<prefix>"` matches branches that *begin* with
    # the prefix. We narrow to open state explicitly.
    args = [
        "gh", "pr", "list",
        "--state", "open",
        "--json", "number,title,url,headRefName",
        "--search", f"head:{branch_prefix}",
        "--limit", "20",
    ]
    if repo:
        args += ["--repo", repo]
    rc, out, err = _run_gh(args)
    warnings: list[str] = []
    if rc != 0:
        stderr = (err or "").strip()
        low = stderr.lower()
        if "not authenticated" in low or "authentication" in low:
            warnings.append("gh not authenticated — PR collision check skipped.")
        elif rc == 124:
            warnings.append(stderr or "gh pr list timed out")
        else:
            warnings.append(f"gh pr list failed (rc={rc}): {stderr[:160]}")
        return [], warnings
    try:
        rows = json.loads(out or "[]")
    except json.JSONDecodeError as e:
        warnings.append(f"gh pr list returned invalid JSON: {e}")
        return [], warnings
    out_rows: list[dict] = []
    for r in rows:
        head = str(r.get("headRefName") or "")
        # gh's search is fuzzy on `head:` — double-check the prefix locally.
        if not head.startswith(branch_prefix):
            continue
        out_rows.append({
            "number": int(r.get("number") or 0),
            "title": str(r.get("title") or "").strip(),
            "url": str(r.get("url") or ""),
            "head": head,
        })
    return out_rows, warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preflight(
    issue_number: int | str,
    *,
    project: str,
    repo: str = "",
    repo_roots: Optional[Iterable[Path]] = None,
) -> PreflightResult:
    """Run the three-axis collision check and return a structured result.

    Parameters
    ----------
    issue_number:
        Numeric GitHub issue id. Coerced from str (``"15"`` / ``"#15"``).
    project:
        skill-hub project name (resolved under ``worktree.repo_roots``) — the
        *local* repository whose worktrees / branches we inspect.
    repo:
        Optional ``owner/name`` passed to ``gh`` for the issue + PR lookups.
        When empty, ``gh`` falls back to the cwd's remote — which fails inside
        the daemon, so always pass this for non-cwd checks.

    Raises
    ------
    ValueError
        On invalid ``issue_number``.
    skill_hub.worktree.WorktreeError
        Propagated from :func:`skill_hub.worktree.resolve_project` when the
        local project can't be located.
    """
    num = _normalize_issue_number(issue_number)
    repo_path = _wt.resolve_project(project, repo_roots=repo_roots)
    issue_prefix = f"issue-{num}-"
    branch_prefix = f"cc/issue-{num}-"

    wts = _list_matching_worktrees(repo_path, issue_prefix)
    branches = _list_matching_branches(repo_path, branch_prefix)

    warnings: list[str] = []
    title, state = "", ""
    prs: list[dict] = []
    if _gh_available():
        title, state, w = _fetch_issue_meta(num, repo)
        warnings.extend(w)
        prs, w = _fetch_open_prs(branch_prefix, repo)
        warnings.extend(w)
    else:
        warnings.append("gh CLI not installed — issue / PR collision check skipped.")

    return PreflightResult(
        issue_number=num,
        project=project,
        repo_path=str(repo_path),
        issue_prefix=issue_prefix,
        branch_prefix=branch_prefix,
        worktrees=wts,
        branches=branches,
        pull_requests=prs,
        issue_title=title,
        issue_state=state,
        warnings=warnings,
    )


def format_result(res: PreflightResult) -> str:
    """Human-readable summary for the MCP tool / CLI."""
    head = (
        f"worktree_preflight #{res.issue_number} ({res.project})"
    )
    if res.issue_title:
        head += f" — {res.issue_title}"
    if res.issue_state:
        head += f" [{res.issue_state}]"
    lines: list[str] = [head]
    if res.safe:
        lines.append("status: safe to start")
    else:
        lines.append("status: collision detected")
    if res.worktrees:
        lines.append(f"  worktrees ({len(res.worktrees)}):")
        for p in res.worktrees:
            lines.append(f"    - {p}")
    if res.branches:
        lines.append(f"  branches ({len(res.branches)}):")
        for b in res.branches:
            lines.append(f"    - {b}")
    if res.pull_requests:
        lines.append(f"  open PRs ({len(res.pull_requests)}):")
        for pr in res.pull_requests:
            lines.append(
                f"    - #{pr['number']} {pr['title']} "
                f"(head {pr['head']}) {pr['url']}"
            )
    if res.warnings:
        lines.append("  warnings:")
        for w in res.warnings:
            lines.append(f"    - {w}")
    return "\n".join(lines)
