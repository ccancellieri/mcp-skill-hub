"""Index GitHub Discussions (+ comments) into vector memory (issue #41).

Design
------
Read-only: fetches discussions via the GitHub GraphQL API and upserts them
into the shared ``vectors`` table under namespace ``discussions``.  A bad
item is skipped, never fatal.  All GitHub I/O goes through two narrow
helpers (``_resolve_repo`` / ``_gh_graphql``) so tests can monkeypatch
them without touching subprocess internals.
"""
from __future__ import annotations

import json
import logging
import subprocess
from typing import Callable

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GraphQL query (mock _gh_graphql in tests, not this string)
# ---------------------------------------------------------------------------

_DISCUSSIONS_QUERY = """
query($owner: String!, $name: String!, $first: Int!) {
  repository(owner: $owner, name: $name) {
    discussions(first: $first, orderBy: {field: UPDATED_AT, direction: DESC}) {
      nodes {
        number
        title
        url
        body
        updatedAt
        category {
          name
          isAnswerable
        }
        author {
          login
        }
        answerChosenAt
        comments(first: 20) {
          nodes {
            id
            body
            url
            updatedAt
            author {
              login
            }
          }
        }
      }
    }
  }
}
"""


# ---------------------------------------------------------------------------
# Narrow mockable helpers
# ---------------------------------------------------------------------------

def _resolve_repo(repo: str) -> tuple[str, str] | None:
    """Return (owner, name) for *repo*.

    If *repo* is ``"owner/name"`` it is split directly.  Otherwise ``gh
    repo view`` is called to resolve the current directory's repo.  Returns
    None on any failure.
    """
    if repo and "/" in repo:
        parts = repo.split("/", 1)
        if len(parts) == 2 and parts[0] and parts[1]:
            return parts[0], parts[1]

    args = ["gh", "repo", "view", "--json", "owner,name"]
    if repo:
        args += ["--repo", repo]
    try:
        res = subprocess.run(
            args, capture_output=True, text=True, check=False, timeout=15
        )
        if res.returncode != 0:
            _log.debug("gh repo view failed (rc=%d): %s", res.returncode, res.stderr.strip())
            return None
        data = json.loads(res.stdout)
        owner = (data.get("owner") or {}).get("login") or data.get("owner", "")
        name = data.get("name", "")
        if owner and name:
            return str(owner), str(name)
        _log.debug("gh repo view returned incomplete data: %s", data)
        return None
    except Exception as exc:  # noqa: BLE001
        _log.debug("gh repo view error: %s", exc)
        return None


def _gh_graphql(query: str, variables: dict, *, timeout: float = 30.0) -> dict | None:
    """Run a GraphQL query via ``gh api graphql``.

    Variables are passed with ``-F`` so integer values stay integers on the
    wire.  Returns the parsed JSON body on rc=0, else None (logged at debug).
    """
    args = ["gh", "api", "graphql", "-f", f"query={query}"]
    for key, value in variables.items():
        args += ["-F", f"{key}={value}"]
    try:
        res = subprocess.run(
            args, capture_output=True, text=True, check=False, timeout=timeout
        )
        if res.returncode != 0:
            _log.debug("gh api graphql failed (rc=%d): %s", res.returncode, res.stderr.strip())
            return None
        return json.loads(res.stdout)
    except Exception as exc:  # noqa: BLE001
        _log.debug("gh api graphql error: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Public sync entry point
# ---------------------------------------------------------------------------

def sync_discussions(
    store: object,
    repo: str = "",
    *,
    dry_run: bool = False,
    first: int = 50,
    emit: Callable | None = None,
) -> dict:
    """Fetch GitHub Discussions and upsert them into vector memory.

    Parameters
    ----------
    store:
        A ``SkillStore`` instance exposing ``upsert_vector``.
    repo:
        ``"owner/name"`` or empty (resolve from current dir).
    dry_run:
        When True, compute counts but make no DB writes.
    first:
        Maximum number of discussions to fetch (GitHub cap: 100).
    emit:
        Optional ``(kind, tool_name, payload) -> ...`` callback.

    Returns
    -------
    dict with keys: checked, indexed, discussions, comments, skipped,
                    dry_run.  On error: adds an ``error`` key.
    """
    resolved = _resolve_repo(repo)
    if resolved is None:
        return {"error": "could not resolve repo", "indexed": 0,
                "checked": 0, "discussions": 0, "comments": 0,
                "skipped": 0, "dry_run": dry_run}

    owner, name = resolved
    result = _gh_graphql(_DISCUSSIONS_QUERY, {"owner": owner, "name": name, "first": first})
    if result is None:
        return {"error": "graphql fetch failed", "indexed": 0,
                "checked": 0, "discussions": 0, "comments": 0,
                "skipped": 0, "dry_run": dry_run}

    nodes = (
        (result.get("data") or {})
        .get("repository", {})
        .get("discussions", {})
        .get("nodes") or []
    )

    report: dict = {
        "checked": len(nodes),
        "indexed": 0,
        "discussions": 0,
        "comments": 0,
        "skipped": 0,
        "dry_run": dry_run,
    }

    for disc in nodes:
        try:
            number = disc.get("number")
            title = disc.get("title") or ""
            url = disc.get("url") or ""
            body = disc.get("body") or ""
            updated_at = disc.get("updatedAt") or ""
            category = disc.get("category") or {}
            cat_name = category.get("name") or ""
            author = disc.get("author") or {}
            author_login = author.get("login") or ""
            answer_chosen_at = disc.get("answerChosenAt")
            answered = bool(answer_chosen_at)

            # Body document
            body_text = f"{title}\n\n{body}".strip()
            if body_text:
                doc_id = f"discussion:{number}"
                meta = {
                    "kind": "discussion",
                    "number": number,
                    "title": title,
                    "url": url,
                    "category": cat_name,
                    "author": author_login,
                    "updated_at": updated_at,
                    "answered": answered,
                    "path": title or url,
                }
                if not dry_run:
                    try:
                        store.upsert_vector(  # type: ignore[attr-defined]
                            namespace="discussions",
                            doc_id=doc_id,
                            text=body_text,
                            source="discussion",
                            metadata=meta,
                        )
                        report["indexed"] += 1
                    except Exception as exc:  # noqa: BLE001
                        _log.debug("upsert_vector failed for %s: %s", doc_id, exc)
                        report["skipped"] += 1
                else:
                    report["indexed"] += 1
                report["discussions"] += 1

            # Comment documents
            comments = (disc.get("comments") or {}).get("nodes") or []
            for comment in comments:
                try:
                    comment_id = comment.get("id") or ""
                    comment_body = comment.get("body") or ""
                    if not comment_body:
                        continue
                    comment_url = comment.get("url") or ""
                    comment_updated = comment.get("updatedAt") or ""
                    comment_author = (comment.get("author") or {}).get("login") or ""

                    c_doc_id = f"discussion:{number}:comment:{comment_id}"
                    c_text = f"Re: {title}\n\n{comment_body}".strip()
                    c_meta = {
                        "kind": "comment",
                        "number": number,
                        "title": title,
                        "url": comment_url,
                        "category": cat_name,
                        "author": comment_author,
                        "updated_at": comment_updated,
                        "path": f"{title} (comment)",
                    }
                    if not dry_run:
                        try:
                            store.upsert_vector(  # type: ignore[attr-defined]
                                namespace="discussions",
                                doc_id=c_doc_id,
                                text=c_text,
                                source="discussion",
                                metadata=c_meta,
                            )
                            report["indexed"] += 1
                        except Exception as exc:  # noqa: BLE001
                            _log.debug("upsert_vector failed for %s: %s", c_doc_id, exc)
                            report["skipped"] += 1
                    else:
                        report["indexed"] += 1
                    report["comments"] += 1
                except Exception as exc:  # noqa: BLE001
                    _log.debug("comment processing error: %s", exc)
                    report["skipped"] += 1

        except Exception as exc:  # noqa: BLE001
            _log.debug("discussion processing error: %s", exc)
            report["skipped"] += 1
            continue

        if emit is not None:
            try:
                emit("discussion.indexed", None, {
                    "number": disc.get("number"),
                    "title": disc.get("title") or "",
                    "url": disc.get("url") or "",
                })
            except Exception:  # noqa: BLE001
                pass

    return report
