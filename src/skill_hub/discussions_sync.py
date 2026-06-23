"""Index GitHub Discussions (+ comments) into vector memory (issue #41).

Design
------
Read-only against GitHub: fetches discussions via the GraphQL API and lands
each one (with its comments folded in) as a mechanical wiki ``source`` page
via ``wiki.write_source_page`` — no LLM. The scan→approve→ingest loop distills
them later. This supersedes the old raw ``discussions`` vector namespace. A bad
item is skipped, never fatal. All GitHub I/O goes through two narrow helpers
(``_resolve_repo`` / ``_gh_graphql``) so tests can monkeypatch them without
touching subprocess internals.
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
