"""GitHub Discussions sync — read path (issue #41) and write path (issue #87).

Read path
---------
Fetches discussions via the GraphQL API and lands each one (with its comments
folded in) as a mechanical wiki ``source`` page via ``wiki.write_source_page``
— no LLM. The scan→approve→ingest loop distills them later. This supersedes
the old raw ``discussions`` vector namespace. A bad item is skipped, never
fatal.

Write path
----------
Promotes a long-form retrospective / design note INTO a GitHub Discussion via
the GraphQL ``createDiscussion`` mutation. Gated behind config key
``discussions_write_enabled`` (default False) — the function is a no-op when
the flag is off. The category is resolved by name via the GraphQL
``repository { discussionCategories }`` query; the target name is configurable
via ``discussions_category`` (default "General").

All GitHub I/O goes through two narrow helpers (``_resolve_repo`` /
``_gh_graphql``) so tests can monkeypatch them without touching subprocess
internals.
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

    # Wave 2: discussions land as mechanical wiki ``source`` pages (no LLM). The
    # scan→approve→ingest loop distills them later. This retires the raw
    # ``discussions`` vector namespace — reindex embeds the source pages into
    # the ``wiki`` namespace instead.
    from . import wiki as _wiki
    from . import config as _cfg
    from pathlib import Path as _Path
    wiki_root = _Path(_cfg.get("wiki_root") or
                      _Path.home() / ".claude" / "mcp-skill-hub" / "wiki")

    for disc in nodes:
        try:
            number = disc.get("number")
            title = disc.get("title") or ""
            url = disc.get("url") or ""
            body = disc.get("body") or ""
            cat_name = (disc.get("category") or {}).get("name") or ""
            author_login = (disc.get("author") or {}).get("login") or ""
            answered = bool(disc.get("answerChosenAt"))

            # Fold the discussion + its comments into one source page body.
            header = (f"_discussion #{number} · {cat_name} · by {author_login}"
                      + (" · answered_" if answered else "_"))
            parts = [f"# {title}".strip(), header]
            if body:
                parts.append(body)
            for comment in (disc.get("comments") or {}).get("nodes") or []:
                cbody = comment.get("body") or ""
                if not cbody:
                    continue
                cauthor = (comment.get("author") or {}).get("login") or ""
                parts.append(f"## Comment by {cauthor}\n\n{cbody}")
                report["comments"] += 1
            page_body = "\n\n".join(p for p in parts if p).strip()
            if not page_body:
                continue

            report["discussions"] += 1
            if dry_run:
                report["indexed"] += 1
            else:
                try:
                    slug = _wiki.write_source_page(
                        store, wiki_root,
                        source_id=f"discussion-{number}",
                        title=title or f"Discussion {number}",
                        body=page_body, url=url,
                        scope="public", project=name,
                    )
                    if slug is not None:
                        report["indexed"] += 1
                    else:
                        report["skipped"] += 1  # unchanged source_hash
                except Exception as exc:  # noqa: BLE001
                    _log.debug("write_source_page failed for #%s: %s", number, exc)
                    report["skipped"] += 1

            if emit is not None:
                try:
                    emit("discussion.indexed", None,
                         {"number": number, "title": title, "url": url})
                except Exception:  # noqa: BLE001
                    pass

        except Exception as exc:  # noqa: BLE001
            _log.debug("discussion processing error: %s", exc)
            report["skipped"] += 1
            continue

    return report


# ---------------------------------------------------------------------------
# Write path — promote a note into a GitHub Discussion (issue #87)
# ---------------------------------------------------------------------------

_CATEGORY_QUERY = """
query($owner: String!, $name: String!) {
  repository(owner: $owner, name: $name) {
    id
    discussionCategories(first: 25) {
      nodes {
        id
        name
      }
    }
  }
}
"""

_CREATE_DISCUSSION_MUTATION = """
mutation($repositoryId: ID!, $categoryId: ID!, $title: String!, $body: String!) {
  createDiscussion(input: {
    repositoryId: $repositoryId,
    categoryId: $categoryId,
    title: $title,
    body: $body
  }) {
    discussion {
      number
      url
      title
    }
  }
}
"""

_ADD_COMMENT_MUTATION = """
mutation($discussionId: ID!, $body: String!) {
  addDiscussionComment(input: {
    discussionId: $discussionId,
    body: $body
  }) {
    comment {
      id
      url
    }
  }
}
"""


def create_discussion(
    repo: str = "",
    title: str = "",
    body: str = "",
    *,
    emit: Callable | None = None,
) -> dict:
    """Promote a note into a GitHub Discussion.

    Gated by config key ``discussions_write_enabled`` (default False).  When
    the flag is off, returns a ``{"status": "disabled"}`` dict immediately —
    no GitHub calls are made.

    The category is resolved by name from config key ``discussions_category``
    (default "General").  If the named category does not exist in the repo,
    returns an error dict rather than guessing.

    Parameters
    ----------
    repo:
        ``"owner/name"`` or empty (resolve from current dir).
    title:
        Discussion title (required).
    body:
        Discussion body markdown (required).
    emit:
        Optional ``(kind, tool_name, payload) -> ...`` callback for event log.

    Returns
    -------
    dict with keys: status, url, number, title.
    On disabled: {"status": "disabled"}.
    On error: {"status": "error", "error": "<message>"}.
    """
    from . import config as _cfg

    write_enabled = bool(_cfg.get("discussions_write_enabled"))
    if not write_enabled:
        return {"status": "disabled"}

    if not title or not title.strip():
        return {"status": "error", "error": "title is required"}
    if not body or not body.strip():
        return {"status": "error", "error": "body is required"}

    category_name: str = str(_cfg.get("discussions_category") or "General")

    resolved = _resolve_repo(repo)
    if resolved is None:
        return {"status": "error", "error": "could not resolve repo"}
    owner, name = resolved

    # Resolve repository node id + category id.
    cat_result = _gh_graphql(_CATEGORY_QUERY, {"owner": owner, "name": name})
    if cat_result is None:
        return {"status": "error", "error": "category query failed"}

    repo_data = (cat_result.get("data") or {}).get("repository") or {}
    repo_id = repo_data.get("id")
    if not repo_id:
        return {"status": "error", "error": "could not resolve repository id"}

    categories = (repo_data.get("discussionCategories") or {}).get("nodes") or []
    category_id: str | None = None
    for cat in categories:
        if (cat.get("name") or "").strip().lower() == category_name.strip().lower():
            category_id = cat.get("id")
            break

    if not category_id:
        available = ", ".join(c.get("name", "") for c in categories if c.get("name"))
        return {
            "status": "error",
            "error": (
                f"category {category_name!r} not found in repo {owner}/{name}. "
                f"Available: {available or '(none)'}"
            ),
        }

    create_result = _gh_graphql(
        _CREATE_DISCUSSION_MUTATION,
        {
            "repositoryId": repo_id,
            "categoryId": category_id,
            "title": title.strip(),
            "body": body.strip(),
        },
    )
    if create_result is None:
        return {"status": "error", "error": "createDiscussion mutation failed"}

    errors = create_result.get("errors")
    if errors:
        msg = "; ".join(e.get("message", str(e)) for e in errors)
        return {"status": "error", "error": f"GraphQL errors: {msg}"}

    disc = (
        (create_result.get("data") or {})
        .get("createDiscussion", {})
        .get("discussion") or {}
    )
    disc_url = disc.get("url", "")
    disc_number = disc.get("number")
    disc_title = disc.get("title", title)

    _log.info(
        "create_discussion: created #%s %r in %s/%s category=%s",
        disc_number, disc_title, owner, name, category_name,
    )

    if emit is not None:
        try:
            emit(
                "discussion.created",
                None,
                {
                    "number": disc_number,
                    "title": disc_title,
                    "url": disc_url,
                    "repo": f"{owner}/{name}",
                    "category": category_name,
                },
            )
        except Exception:  # noqa: BLE001
            pass

    return {
        "status": "ok",
        "number": disc_number,
        "title": disc_title,
        "url": disc_url,
    }
