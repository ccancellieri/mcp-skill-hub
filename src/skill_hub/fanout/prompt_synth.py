"""Per-issue Agent-prompt synthesis.

Reads the target repo's `.github/ISSUE_TEMPLATE/*.yml`, `.github/labels.yml`,
and any `scripts/seed_*.sh` files to give the local LLM enough context to
draft a focused Agent prompt: scope, acceptance criteria, files-of-interest
hint, constraints. Falls back to a deterministic template when the LLM is
unavailable.

Results are cached in SQLite (`fanout_prompt_cache`, keyed by issue URL or
content hash) so repeated `fanout_issues` calls don't re-spend LLM budget.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from pathlib import Path

from .sources import Issue

# Limit how much of each template / seed script we ship to the LLM — these
# files can be long but only their structure is load-bearing.
_TEMPLATE_BUDGET = 2_000
_SEED_BUDGET = 1_500
_BODY_BUDGET = 4_000

# Marker prefix identifying a prompt that already carries the standing
# tooling directive (see build_standing_preamble) — lets other call sites
# (e.g. directive.py) avoid stacking a second copy on top.
PREAMBLE_MARKER = "STANDING DIRECTIVE:"


def _is_codegraph_indexed(repo_path: str | Path) -> bool:
    """True if `repo_path` has a `.codegraph/` index directory."""
    return os.path.isdir(Path(repo_path) / ".codegraph")


def build_standing_preamble(repo_path: str | Path) -> str:
    """Few-line tooling directive prepended to every generated agent prompt.

    States explicitly whether the target repo is codegraph-indexed (checked
    at prompt-build time) so the dispatched agent doesn't default to grep,
    and nudges it toward compressed fetch/search tools for web pages and
    large payloads.
    """
    if _is_codegraph_indexed(repo_path):
        codegraph_line = (
            "This repo IS codegraph-indexed (.codegraph/ found) — prefer "
            "codegraph_search / codegraph_callers / codegraph_callees / "
            "codegraph_impact over grep for symbol lookups and change impact."
        )
    else:
        codegraph_line = (
            "This repo IS NOT codegraph-indexed (no .codegraph/) — grep/glob "
            "are the right tool for code search here."
        )
    return (
        f"{PREAMBLE_MARKER}\n"
        f"- {codegraph_line}\n"
        "- Prefer fetch_compressed / compressed search over raw fetches for web "
        "pages and large payloads; skip compression only when exact bytes matter "
        "(code layout, JSON structure checks).\n"
    )


def _safe_read(path: Path, budget: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[:budget]


def gather_repo_context(repo_path: str | Path) -> dict[str, str]:
    """Read the four GH-template / labels / seed files (best-effort).

    Returns a dict with keys: feature_template, bug_template, config_template,
    labels, seed_labels, seed_issues. Missing files map to empty strings.
    """
    repo = Path(repo_path)
    gh = repo / ".github"
    tpl = gh / "ISSUE_TEMPLATE"
    scripts = repo / "scripts"
    return {
        "feature_template": _safe_read(tpl / "feature.yml", _TEMPLATE_BUDGET),
        "bug_template": _safe_read(tpl / "bug.yml", _TEMPLATE_BUDGET),
        "config_template": _safe_read(tpl / "config.yml", _TEMPLATE_BUDGET),
        "labels": _safe_read(gh / "labels.yml", _TEMPLATE_BUDGET),
        "seed_labels": _safe_read(scripts / "seed_labels.sh", _SEED_BUDGET),
        "seed_issues": _safe_read(scripts / "seed_issues.sh", _SEED_BUDGET),
    }


_FALLBACK_TEMPLATE = """\
You are working on issue: {title}

ISSUE BODY:
{body}

LABELS: {labels}
URL: {url}

INSTRUCTIONS:
1. Read the issue body carefully; identify the smallest scope that satisfies it.
2. Locate the relevant files via grep/glob before editing.
3. Make focused changes — do NOT refactor unrelated code.
4. Write/extend tests for the change.
5. Run the project's test suite and report a green run.
6. When done, summarize the diff in 3-5 bullets.

CONSTRAINTS:
- Work only inside this worktree; do not touch sibling worktrees.
- Follow existing project conventions visible in nearby code.
- Do not add new top-level dependencies without flagging it.
"""


def _fallback_prompt(issue: Issue, repo_path: str | Path) -> str:
    return build_standing_preamble(repo_path) + "\n" + _FALLBACK_TEMPLATE.format(
        title=issue.title or "(no title)",
        body=(issue.body or "(no body)")[:_BODY_BUDGET],
        labels=", ".join(issue.labels) or "(none)",
        url=issue.url or "(none)",
    )


def _llm_prompt(issue: Issue, repo_ctx: dict[str, str], repo_path: str | Path) -> str:
    indexed = _is_codegraph_indexed(repo_path)
    parts = [
        "You are drafting an instruction prompt for a separate Claude Code agent that will work on ONE GitHub issue in an isolated git worktree.",
        "Your output is the prompt itself — no preamble, no JSON, no code fences. Plain text only.",
        "",
        f"This repo {'IS' if indexed else 'IS NOT'} codegraph-indexed "
        f"(.codegraph/ {'found' if indexed else 'not found'}).",
        "",
        "Issue title:",
        issue.title or "(no title)",
        "",
        "Issue body:",
        (issue.body or "(no body)")[:_BODY_BUDGET],
        "",
        f"Labels: {', '.join(issue.labels) or '(none)'}",
        f"URL: {issue.url or '(none)'}",
        "",
    ]
    sections = [
        ("Repo issue template — feature.yml", repo_ctx.get("feature_template")),
        ("Repo issue template — bug.yml", repo_ctx.get("bug_template")),
        ("Repo labels.yml", repo_ctx.get("labels")),
        ("Repo scripts/seed_labels.sh", repo_ctx.get("seed_labels")),
        ("Repo scripts/seed_issues.sh", repo_ctx.get("seed_issues")),
    ]
    for header, body in sections:
        if body:
            parts += [f"--- {header} ---", body, ""]
    parts += [
        "Draft a focused agent prompt with these sections (in this order):",
        "1. Scope: 1-2 sentences naming the bounded change.",
        "2. Acceptance criteria: 3-5 bullets, testable.",
        "3. Files of interest: best-guess paths the agent should grep/read first.",
        "4. Constraints: what NOT to touch; only-this-worktree reminder; if the repo "
        "is codegraph-indexed (stated above) prefer codegraph_search/callers/callees/"
        "impact over grep, else use grep/glob; prefer fetch_compressed/compressed "
        "search over raw fetches for web pages and large payloads unless exact bytes "
        "matter.",
        "5. Done-when: how the agent should report completion.",
        "Keep the prompt under 400 words. Address the agent directly in imperative voice.",
    ]
    return "\n".join(parts)


def _cache_key(issue: Issue) -> str:
    if issue.url:
        return f"url:{issue.url}"
    h = hashlib.sha256()
    h.update(issue.title.encode("utf-8"))
    h.update(b"\x00")
    h.update(issue.body.encode("utf-8"))
    return f"hash:{h.hexdigest()[:32]}"


def _ensure_cache_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        "CREATE TABLE IF NOT EXISTS fanout_prompt_cache ("
        "  key TEXT PRIMARY KEY, "
        "  prompt TEXT NOT NULL, "
        "  source TEXT, "
        "  quality TEXT, "
        "  created_at TEXT DEFAULT (datetime('now'))"
        ")"
    )


def _get_cached(conn: sqlite3.Connection, key: str) -> tuple[str, str] | None:
    _ensure_cache_table(conn)
    row = conn.execute(
        "SELECT prompt, quality FROM fanout_prompt_cache WHERE key = ?",
        (key,),
    ).fetchone()
    if not row:
        return None
    return (row[0], row[1] or "")


def _put_cached(conn: sqlite3.Connection, key: str, prompt: str,
                source: str, quality: str) -> None:
    _ensure_cache_table(conn)
    conn.execute(
        "INSERT OR REPLACE INTO fanout_prompt_cache (key, prompt, source, quality) "
        "VALUES (?, ?, ?, ?)",
        (key, prompt, source, quality),
    )
    conn.commit()


def draft_prompt(
    issue: Issue,
    repo_path: str | Path,
    *,
    store_conn: sqlite3.Connection | None = None,
    use_cache: bool = True,
    use_llm: bool = True,
) -> tuple[str, str]:
    """Draft an Agent prompt for one issue.

    Returns (prompt_text, quality) where quality ∈ {"llm", "fallback"}.
    Caches LLM-quality results keyed by issue URL or content hash.
    """
    key = _cache_key(issue)
    if use_cache and store_conn is not None:
        cached = _get_cached(store_conn, key)
        if cached:
            return cached

    repo_ctx = gather_repo_context(repo_path)
    quality = "fallback"
    text = _fallback_prompt(issue, repo_path)

    if use_llm:
        try:
            from ..llm import LLMError, get_provider
            from .. import config as _cfg
            cfg = (_cfg.load_config().get("fanout") or {})
            model = cfg.get("prompt_model")
            timeout = int(cfg.get("prompt_timeout", 60))
            kwargs: dict = {"max_tokens": 800, "temperature": 0.2, "timeout": timeout}
            if model:
                kwargs["model"] = model
            raw = get_provider().complete(_llm_prompt(issue, repo_ctx, repo_path), **kwargs)
            raw = (raw or "").strip()
            # Strip any accidental <think>...</think> block.
            if "<think>" in raw:
                import re as _re
                raw = _re.sub(r"<think>.*?</think>", "", raw, flags=_re.DOTALL).strip()
            if raw and len(raw) > 60:
                # Prepend deterministically — don't rely on the LLM having
                # followed the standing-tooling instruction in _llm_prompt.
                text = build_standing_preamble(repo_path) + "\n" + raw
                quality = "llm"
        except (ImportError, RuntimeError, ValueError):
            pass
        except Exception:  # noqa: BLE001 — LLM is best-effort
            pass

    if use_cache and store_conn is not None and quality == "llm":
        _put_cached(store_conn, key, text, issue.source, quality)
    return text, quality


__all__ = ["draft_prompt", "gather_repo_context", "build_standing_preamble", "PREAMBLE_MARKER"]
