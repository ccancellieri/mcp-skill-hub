"""Fanout — turn a list of issues into N parallel worktree-bound tasks.

One MCP call (`fanout_issues`) takes a source ("gh" / "text" / pluggable),
fetches N issues, drafts an `Agent` prompt per issue from the repo's GH
templates and labels, creates a worktree + skill-hub task per issue, and
returns a directive the active Claude pastes back to dispatch all `Agent`
calls in a single message.

Layout:
    sources.py        — Issue dataclass + IssueSource protocol + adapters
    prompt_synth.py   — Per-issue prompt drafting (LLM + template fallback)
    coordinator.py    — fanout(...) orchestrator; reuses ensure_worktree + save_task
    directive.py      — Renders the parallel-Agent dispatch directive
"""
from __future__ import annotations

from .coordinator import FanoutResult, fanout
from .directive import render_directive
from .sources import Issue, IssueSource, get_source

__all__ = [
    "FanoutResult",
    "Issue",
    "IssueSource",
    "fanout",
    "get_source",
    "render_directive",
]
