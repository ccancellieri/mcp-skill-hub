"""Issue #37 — bidirectional task↔GitHub issue sync.

Design
------
GitHub is the source of truth (issue wins).  When a linked issue is closed,
the local task is closed automatically.  When a local task is closed but the
linked issue is still open, the sync can optionally write back a comment or
close the issue (configurable; default 'off' = no writes).

All GitHub I/O is channelled through two narrow helpers (_gh_view / _gh_comment
/ _gh_close) so tests can monkeypatch them without touching subprocess internals.
"""
from __future__ import annotations

import json
import logging
import subprocess
from typing import Callable

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Read-only gh helper (mock THIS in tests)
# ---------------------------------------------------------------------------

def _gh_view(number: int, repo: str = "") -> dict | None:
    """Fetch ``state``, ``title``, ``url`` for a GitHub issue via ``gh``.

    Returns a dict on success, None on any error (not-found, auth failure,
    network, ``gh`` not installed, …).
    """
    args = [
        "gh", "issue", "view", str(number),
        "--json", "state,title,url",
    ]
    if repo:
        args += ["--repo", repo]
    try:
        res = subprocess.run(
            args, capture_output=True, text=True, check=False, timeout=15
        )
        if res.returncode != 0:
            _log.debug("gh issue view %s failed (rc=%d): %s",
                       number, res.returncode, res.stderr.strip())
            return None
        return json.loads(res.stdout)
    except Exception as exc:  # noqa: BLE001
        _log.debug("gh issue view %s error: %s", number, exc)
        return None


# ---------------------------------------------------------------------------
# Write-back gh helpers (mock THESE in tests)
# ---------------------------------------------------------------------------

def _gh_comment(number: int, body: str, repo: str = "") -> bool:
    """Post a comment on a GitHub issue.  Returns True on success."""
    args = ["gh", "issue", "comment", str(number), "--body", body]
    if repo:
        args += ["--repo", repo]
    try:
        res = subprocess.run(
            args, capture_output=True, text=True, check=False, timeout=15
        )
        if res.returncode != 0:
            _log.warning("gh issue comment %s failed: %s", number, res.stderr.strip())
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("gh issue comment %s error: %s", number, exc)
        return False


def _gh_close(number: int, repo: str = "") -> bool:
    """Close a GitHub issue via the CLI.  Returns True on success."""
    args = ["gh", "issue", "close", str(number)]
    if repo:
        args += ["--repo", repo]
    try:
        res = subprocess.run(
            args, capture_output=True, text=True, check=False, timeout=15
        )
        if res.returncode != 0:
            _log.warning("gh issue close %s failed: %s", number, res.stderr.strip())
            return False
        return True
    except Exception as exc:  # noqa: BLE001
        _log.warning("gh issue close %s error: %s", number, exc)
        return False


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------

def reconcile(
    store: object,
    *,
    repo: str = "",
    dry_run: bool = False,
    writeback: str = "off",
    emit: Callable | None = None,
) -> dict:
    """Reconcile all task↔issue links.

    Parameters
    ----------
    store:
        A ``SkillStore`` instance (or any object exposing
        ``list_all_issue_links``, ``get_task``, ``close_task``,
        ``update_link_state``).
    repo:
        Optional filter — only process links whose ``repo`` matches.
    dry_run:
        When True, compute what *would* happen but make no DB writes and no
        gh calls.
    writeback:
        "off"     — never write to GitHub (default; safe).
        "comment" — post a completion comment on the linked issue when a task
                    is closed locally; idempotent via ``writeback_done`` flag.
        "close"   — comment + close the linked issue.
    emit:
        Optional ``(kind, tool_name, payload) -> ...`` callback.  The server
        passes ``_store.append_event`` bound to the current session.

    Returns
    -------
    dict with keys: checked, tasks_closed, issues_commented, issues_closed,
                    drift, dry_run, writeback.
    """
    links: list[dict] = store.list_all_issue_links(repo=repo)  # type: ignore[attr-defined]

    report: dict = {
        "checked": len(links),
        "tasks_closed": 0,
        "issues_commented": 0,
        "issues_closed": 0,
        "drift": [],
        "dry_run": dry_run,
        "writeback": writeback,
    }

    for link in links:
        link_id = link["id"]
        task_id = link["task_id"]
        issue_number = link["issue_number"]
        link_repo = link["repo"] or ""

        task = store.get_task(task_id)  # type: ignore[attr-defined]
        if task is None:
            _log.debug("link %d references missing task %d — skipping", link_id, task_id)
            continue

        task_status = task["status"]

        gh_data = _gh_view(issue_number, repo=link_repo)
        if gh_data is None:
            _log.debug("could not fetch gh issue %s#%d", link_repo, issue_number)
            continue

        issue_state = (gh_data.get("state") or "").lower()  # "open" | "closed"

        # Persist the fetched state regardless of what action we take.
        if not dry_run:
            store.update_link_state(link_id, issue_state)  # type: ignore[attr-defined]

        # ── Branch 1: issue closed, task open → issue wins ─────────────────
        if issue_state == "closed" and task_status == "open":
            drift_entry = {
                "link_id": link_id,
                "task_id": task_id,
                "issue_number": issue_number,
                "repo": link_repo,
                "direction": "issue→task",
                "action": "close_task" if not dry_run else "would_close_task",
            }
            report["drift"].append(drift_entry)

            if not dry_run:
                compact_msg = f"auto-closed: linked issue #{issue_number} closed"
                store.close_task(task_id, compact=compact_msg)  # type: ignore[attr-defined]
                report["tasks_closed"] += 1
                if emit is not None:
                    try:
                        emit("task.closed", None, {
                            "task_id": task_id,
                            "reason": "issue_closed",
                            "issue_number": issue_number,
                            "repo": link_repo,
                        })
                    except Exception:  # noqa: BLE001
                        pass

        # ── Branch 2: task closed, issue open → potential writeback ────────
        elif task_status == "closed" and issue_state == "open":
            drift_entry = {
                "link_id": link_id,
                "task_id": task_id,
                "issue_number": issue_number,
                "repo": link_repo,
                "direction": "task→issue",
                "writeback_mode": writeback,
                "writeback_done": bool(link.get("writeback_done")),
            }
            report["drift"].append(drift_entry)

            if writeback == "off" or dry_run:
                # Record intent only — no gh write.
                continue

            if link.get("writeback_done"):
                # Idempotent: already wrote back.
                continue

            comment_body = f"Completed via skill-hub task #{task_id}."

            if writeback in ("comment", "close"):
                commented = _gh_comment(issue_number, comment_body, repo=link_repo)
                if commented:
                    report["issues_commented"] += 1
                    if emit is not None:
                        try:
                            emit("issue.commented", None, {
                                "issue_number": issue_number,
                                "repo": link_repo,
                                "task_id": task_id,
                            })
                        except Exception:  # noqa: BLE001
                            pass

            if writeback == "close":
                closed = _gh_close(issue_number, repo=link_repo)
                if closed:
                    report["issues_closed"] += 1
                    if emit is not None:
                        try:
                            emit("issue.closed", None, {
                                "issue_number": issue_number,
                                "repo": link_repo,
                                "task_id": task_id,
                            })
                        except Exception:  # noqa: BLE001
                            pass

            # Mark writeback done so re-runs are no-ops.
            if writeback in ("comment", "close"):
                store.update_link_state(  # type: ignore[attr-defined]
                    link_id, issue_state, writeback_done=1
                )

    return report
