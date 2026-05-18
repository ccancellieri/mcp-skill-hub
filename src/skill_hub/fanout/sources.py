"""Issue sources — pluggable adapters that yield normalized Issue records.

Defaults:
    GitHubSource — shells out to `gh issue list` / `gh issue view`
    TextSource   — parses a bullet / numbered list from a free-text blob

Custom sources can be registered via config:
    fanout.sources = ["skill_hub.fanout.sources:GitHubSource",
                      "my_pkg.adapters:LinearSource"]
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from importlib import import_module
from typing import Iterable, Protocol, runtime_checkable


@dataclass
class Issue:
    """Normalized issue record returned by every source."""
    id: str               # source-scoped id ("gh:123", "text:0001")
    title: str
    body: str = ""
    labels: list[str] = field(default_factory=list)
    url: str = ""
    source: str = ""      # "gh", "text", custom name
    raw: dict = field(default_factory=dict)


@runtime_checkable
class IssueSource(Protocol):
    """Pluggable adapter — implement `name` and `fetch`."""
    name: str

    def fetch(self, filter: str = "", limit: int | None = None,
              *, repo: str = "", cwd: str = "", **kwargs) -> list[Issue]: ...


# ---------------------------------------------------------------------------
# GitHubSource
# ---------------------------------------------------------------------------

class GitHubSource:
    """Fetch issues via the `gh` CLI.

    `filter` is passed as a free-form query to `gh issue list --search` so
    callers can use the full GitHub search syntax (`label:bug is:open`,
    `assignee:@me`, etc.). `repo` (``owner/name``) or `cwd` (a path inside
    a git checkout) are how `gh` finds the target repo when the caller's
    own cwd isn't a git repository — at least one must resolve, otherwise
    `gh` errors with ``not a git repository``.
    """
    name = "gh"

    def fetch(self, filter: str = "", limit: int | None = None,
              *, repo: str = "", cwd: str = "", **kwargs) -> list[Issue]:
        if not shutil.which("gh"):
            raise RuntimeError(
                "fanout source=gh requires the `gh` CLI; install it and run `gh auth login`."
            )
        args = [
            "gh", "issue", "list",
            "--json", "number,title,body,labels,url,state",
            "--limit", str(limit or 30),
        ]
        if filter:
            args += ["--search", filter]
        if repo:
            args += ["--repo", repo]
        run_cwd = cwd or None
        res = subprocess.run(args, capture_output=True, text=True, check=False,
                             cwd=run_cwd)
        if res.returncode != 0:
            stderr = (res.stderr or "").strip()
            if "authentication" in stderr.lower() or "not authenticated" in stderr.lower():
                raise RuntimeError("gh not authenticated — run `gh auth login` and retry.")
            raise RuntimeError(f"gh issue list failed: {stderr or res.stdout.strip()}")
        try:
            rows = json.loads(res.stdout or "[]")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"gh issue list returned invalid JSON: {e}") from e
        issues: list[Issue] = []
        for r in rows:
            labels = [lab.get("name", "") for lab in (r.get("labels") or []) if lab.get("name")]
            issues.append(Issue(
                id=f"gh:{r.get('number')}",
                title=str(r.get("title") or "").strip(),
                body=str(r.get("body") or "").strip(),
                labels=labels,
                url=str(r.get("url") or ""),
                source=self.name,
                raw=r,
            ))
        return issues


# ---------------------------------------------------------------------------
# TextSource
# ---------------------------------------------------------------------------

# Match "- foo", "* foo", "1. foo", "1) foo" (single-line items).
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+(.+?)\s*$")


class TextSource:
    """Parse a bullet / numbered list into Issue records.

    `filter` is the raw text. Each line that starts with `-`, `*`, `+`, or
    `N.` / `N)` becomes one Issue. Empty lines and non-bullet lines are
    treated as continuation of the previous item's body (if any).
    """
    name = "text"

    def fetch(self, filter: str = "", limit: int | None = None,
              *, repo: str = "", cwd: str = "", **kwargs) -> list[Issue]:
        if not filter:
            return []
        issues: list[Issue] = []
        current: Issue | None = None
        body_buf: list[str] = []
        for line in filter.splitlines():
            m = _BULLET_RE.match(line)
            if m:
                if current is not None:
                    current.body = "\n".join(body_buf).strip()
                    issues.append(current)
                    body_buf = []
                current = Issue(
                    id=f"text:{len(issues)+1:04d}",
                    title=m.group(1).strip(),
                    source=self.name,
                )
            elif current is not None and line.strip():
                body_buf.append(line.strip())
        if current is not None:
            current.body = "\n".join(body_buf).strip()
            issues.append(current)
        if limit:
            issues = issues[: int(limit)]
        return issues


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BUILTIN: dict[str, type] = {
    "gh": GitHubSource,
    "github": GitHubSource,
    "text": TextSource,
}


def _configured_sources() -> Iterable[str]:
    try:
        from .. import config as _cfg
        out = (_cfg.load_config().get("fanout") or {}).get("sources")
    except Exception:  # noqa: BLE001
        out = None
    return list(out) if out else []


def get_source(name: str) -> IssueSource:
    """Resolve a source name to an instance.

    Order: built-ins → dotted-paths from config (`module:Class`).
    """
    name = (name or "").strip().lower()
    if name in _BUILTIN:
        return _BUILTIN[name]()
    for dotted in _configured_sources():
        mod_name, _, cls_name = dotted.partition(":")
        if not mod_name or not cls_name:
            continue
        try:
            cls = getattr(import_module(mod_name), cls_name)
        except (ImportError, AttributeError):
            continue
        try:
            instance = cls()
        except TypeError:
            continue
        if getattr(instance, "name", "").lower() == name:
            return instance
    raise ValueError(
        f"unknown fanout source: {name!r}. "
        f"built-ins: {sorted(_BUILTIN)}; check fanout.sources config for custom adapters."
    )
