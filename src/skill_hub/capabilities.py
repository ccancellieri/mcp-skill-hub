"""Capability matrix — what every MCP tool needs and whether it works now.

Issue #13 (M1 — Useful Without LLM): a stdlib-only, single-source-of-truth
view of "given the current backend state, which tools work?" Each tool
declares which backends it depends on; this module probes the backends
once per request and renders a green / yellow / red verdict.

Issue #7 extends this with a coarser ``tier`` axis (``none`` / ``embedding``
/ ``llm``) so ``status`` / ``list_skills`` / ``--help`` can answer the
simpler question "does this tool work without an LLM at all?" without
re-deriving from the full backend matrix on every call. The tier is
mechanically derived from each ``ToolSpec``'s hard deps and recorded in
``TIER_REGISTRY`` at import time so no tool can end up ``unknown``. The
``@requires_capability`` decorator lets server.py declare a tier inline
on a tool def — useful as an explicit override or as a self-documenting
annotation visible at the call site.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Callable, Literal, TypeVar

# Backend identifiers used across the matrix. Keep small and stable.
BACKEND_MCP = "mcp"
BACKEND_DB = "db"
BACKEND_GIT = "git"
BACKEND_GH = "gh"
BACKEND_CLAUDE_CLI = "claude_cli"
BACKEND_OLLAMA = "ollama"             # daemon reachable
BACKEND_EMBED = "embed"               # any embedding backend usable
BACKEND_REASON_LLM = "reason_llm"     # configured reason_model installed in Ollama
BACKEND_SEARXNG = "searxng"           # service enabled + reachable
BACKEND_VOYAGE = "voyage"             # VOYAGE_API_KEY present


@dataclass(frozen=True)
class Backend:
    """One backend skill-hub may or may not have right now."""
    id: str
    label: str
    setup: str  # one-line setup instructions
    check: Callable[[], bool]


def _check_mcp() -> bool:
    # If we are rendering this page, the MCP webapp is up. Always green.
    return True


def _check_db() -> bool:
    # SkillStore is always available — sqlite is stdlib.
    return True


def _check_git() -> bool:
    return shutil.which("git") is not None


def _check_gh() -> bool:
    return shutil.which("gh") is not None


def _check_claude_cli() -> bool:
    return shutil.which("claude") is not None


def _no_llm_mode() -> bool:
    """True when the user has opted out of every LLM-backed feature.

    Centralised so backend probes, ``status``, and the dashboard banner all
    agree on the answer without each importing config independently.
    """
    try:
        from . import config as _cfg
        return bool(_cfg.get("no_llm_mode"))
    except Exception:  # noqa: BLE001
        return False


# Backends the user explicitly disables when no_llm_mode is on. The remaining
# backends (mcp/db/git/gh/claude_cli/searxng) are independent of the local LLM
# stack and continue probing normally.
_NO_LLM_DISABLED_BACKENDS: frozenset[str] = frozenset({
    "ollama", "embed", "reason_llm", "voyage",
})


def _check_ollama() -> bool:
    """Daemon reachable, regardless of which models are installed."""
    if _no_llm_mode():
        return False
    try:
        from .ollama_client import get_ollama_client
        return get_ollama_client().get_api_base(None) is not None
    except Exception:  # noqa: BLE001
        return False


def _check_embed() -> bool:
    if _no_llm_mode():
        return False
    try:
        from .embeddings import embed_available
        return bool(embed_available())
    except Exception:  # noqa: BLE001
        return False


def _check_reason_llm() -> bool:
    if _no_llm_mode():
        return False
    try:
        from . import config as _cfg
        from .embeddings import ollama_available
        reason_model = _cfg.get("reason_model") or "deepseek-r1:1.5b"
        return bool(ollama_available(reason_model))
    except Exception:  # noqa: BLE001
        return False


def _check_searxng() -> bool:
    try:
        from . import config as _cfg
        from .searxng import _resolve_searxng_url
        if not _cfg.is_service_enabled("searxng"):
            return False
        return _resolve_searxng_url(timeout=2.0) is not None
    except Exception:  # noqa: BLE001
        return False


def _check_voyage() -> bool:
    if _no_llm_mode():
        return False
    if os.environ.get("VOYAGE_API_KEY"):
        return True
    try:
        from . import config as _cfg
        return bool(_cfg.get("voyage_api_key"))
    except Exception:  # noqa: BLE001
        return False


BACKENDS: dict[str, Backend] = {
    b.id: b for b in [
        Backend(BACKEND_MCP, "MCP server",
                "Running — this page proves it.", _check_mcp),
        Backend(BACKEND_DB, "Local SQLite",
                "Bundled — no setup needed.", _check_db),
        Backend(BACKEND_GIT, "git CLI",
                "Install git: https://git-scm.com/downloads", _check_git),
        Backend(BACKEND_GH, "GitHub CLI",
                "Install gh: https://cli.github.com — then `gh auth login`.", _check_gh),
        Backend(BACKEND_CLAUDE_CLI, "Claude CLI",
                "Install Claude Code: https://github.com/anthropics/claude-code", _check_claude_cli),
        Backend(BACKEND_OLLAMA, "Ollama daemon",
                "Install Ollama: https://ollama.com — then `ollama serve`.", _check_ollama),
        Backend(BACKEND_EMBED, "Embedding backend",
                "Pick one: `ollama pull nomic-embed-text`, set VOYAGE_API_KEY, "
                "or `pip install sentence-transformers`.", _check_embed),
        Backend(BACKEND_REASON_LLM, "Reasoning model",
                "Pull the configured reason_model with Ollama "
                "(e.g. `ollama pull deepseek-r1:1.5b`).", _check_reason_llm),
        Backend(BACKEND_SEARXNG, "SearXNG (web search)",
                "Enable on /control or run "
                "`docker compose -f docker/docker-compose.searxng.yml up -d`.",
                _check_searxng),
        Backend(BACKEND_VOYAGE, "Voyage embeddings",
                "Set the VOYAGE_API_KEY environment variable.", _check_voyage),
    ]
}


@dataclass(frozen=True)
class ToolSpec:
    """Per-tool declaration: hard deps fail the tool, soft deps degrade it."""
    name: str
    summary: str
    hard: tuple[str, ...] = ()   # all required → red if any missing
    soft: tuple[str, ...] = ()   # nice-to-have → yellow if any missing


# 72 MCP tools (see server.py @mcp.tool decorators). Grouped by category so
# the page renders predictably. Dependencies were derived by inspecting each
# tool body — keep this list in sync when adding/removing @mcp.tool entries.
TOOLS: tuple[ToolSpec, ...] = (
    # --- Search & RAG (need an embedding backend) ---
    ToolSpec("search_skills", "Semantic skill search",
             hard=(BACKEND_DB, BACKEND_EMBED)),
    ToolSpec("suggest_plugins", "Recommend plugins to enable",
             hard=(BACKEND_DB, BACKEND_EMBED)),
    ToolSpec("search_context", "Unified semantic search (skills/tasks/teachings)",
             hard=(BACKEND_DB, BACKEND_EMBED)),
    ToolSpec("search_context_profile", "Profile-scoped semantic search",
             hard=(BACKEND_DB, BACKEND_EMBED)),
    ToolSpec("search_web", "Web search via SearXNG",
             hard=(BACKEND_SEARXNG,)),
    # --- Teachings ---
    ToolSpec("teach", "Add a semantic teaching rule",
             hard=(BACKEND_DB, BACKEND_EMBED)),
    ToolSpec("forget_teaching", "Remove a teaching rule",
             hard=(BACKEND_DB,)),
    ToolSpec("list_teachings", "List all teaching rules",
             hard=(BACKEND_DB,)),
    # --- Skills index ---
    ToolSpec("list_skills", "List indexed skills",
             hard=(BACKEND_DB,)),
    ToolSpec("index_skills", "Rebuild the skill index",
             hard=(BACKEND_DB,), soft=(BACKEND_EMBED,)),
    ToolSpec("index_plugins", "Rebuild the plugin suggestion index",
             hard=(BACKEND_DB, BACKEND_EMBED)),
    ToolSpec("update_marketplace", "Sync the upstream skill marketplace",
             hard=(BACKEND_GIT,), soft=(BACKEND_EMBED,)),
    # --- Feedback / signals ---
    ToolSpec("record_feedback", "Record helpful/unhelpful on a skill",
             hard=(BACKEND_DB,)),
    ToolSpec("log_session", "Record a plugin/tool invocation",
             hard=(BACKEND_DB,)),
    ToolSpec("session_stats", "Most-used plugins from history",
             hard=(BACKEND_DB,)),
    ToolSpec("close_session", "End the current session with a summary",
             hard=(BACKEND_DB,)),
    # --- Tasks ---
    ToolSpec("save_task", "Save current work as a task",
             hard=(BACKEND_DB,), soft=(BACKEND_EMBED,)),
    ToolSpec("close_task", "Close + compact a task",
             hard=(BACKEND_DB,), soft=(BACKEND_REASON_LLM,)),
    ToolSpec("reopen_task", "Reopen a closed task",
             hard=(BACKEND_DB,)),
    ToolSpec("list_tasks", "List tasks (open/closed/all)",
             hard=(BACKEND_DB,)),
    ToolSpec("update_task", "Update a task's summary/context",
             hard=(BACKEND_DB,), soft=(BACKEND_EMBED,)),
    ToolSpec("set_task_auto_approve", "Toggle auto-approve on a task",
             hard=(BACKEND_DB,)),
    ToolSpec("set_task_options", "Set per-task options",
             hard=(BACKEND_DB,)),
    # --- M1 claims layer (no LLM) ---
    ToolSpec("claim_task", "Claim a free task for an agent",
             hard=(BACKEND_DB,)),
    ToolSpec("handoff_task", "Hand off a claimed task to another agent",
             hard=(BACKEND_DB,)),
    ToolSpec("steal_task", "Steal a stale claim (after stealable_at)",
             hard=(BACKEND_DB,)),
    ToolSpec("release_task", "Release the claim on a task",
             hard=(BACKEND_DB,)),
    # --- Fanout ---
    ToolSpec("fanout_issues", "Fan out N GitHub issues into worktree tasks",
             hard=(BACKEND_DB, BACKEND_GIT, BACKEND_GH),
             soft=(BACKEND_CLAUDE_CLI,)),
    ToolSpec("fanout_status", "Status of a fanout group",
             hard=(BACKEND_DB,)),
    ToolSpec("fanout_close", "Close all tasks in a fanout group",
             hard=(BACKEND_DB,)),
    ToolSpec("fanout_cleanup", "Remove worktrees for a fanout group",
             hard=(BACKEND_DB, BACKEND_GIT)),
    # --- Worktree / policy ---
    ToolSpec("worktree_preflight", "Check worktree/branch/PR collision",
             hard=(BACKEND_GIT,), soft=(BACKEND_GH,)),
    ToolSpec("sync_check", "Cross-repo stale-import detector",
             hard=(BACKEND_GIT,)),
    ToolSpec("lint_canary", "Rotate ruff selectors as a canary",
             hard=()),
    ToolSpec("record_witness", "Append a fix-manifest entry to the witness log",
             hard=()),
    ToolSpec("list_witness", "List fix-manifest entries from the witness log",
             hard=()),
    ToolSpec("export_policies", "Export memory rules to per-repo POLICY.md",
             hard=()),
    ToolSpec("federation_view", "Read tasks from a remote skill-hub DB",
             hard=(BACKEND_DB,)),
    # --- Planning / team orchestration ---
    ToolSpec("validate_plan", "Static validation of a plan YAML",
             hard=()),
    ToolSpec("team_plan", "Resolve a specialized /team roster + cost estimate",
             hard=()),
    # --- Events / issue sync ---
    ToolSpec("get_events", "Query the append-only event log",
             hard=(BACKEND_DB,)),
    ToolSpec("events_prune", "Prune old rows from the event log",
             hard=(BACKEND_DB,)),
    ToolSpec("issue_sync", "Sync skill-hub tasks with GitHub issues",
             hard=(BACKEND_DB,), soft=(BACKEND_GH,)),
    ToolSpec("discussions_sync", "Index GitHub Discussions into vector memory",
             hard=(BACKEND_DB,), soft=(BACKEND_GH,)),
    ToolSpec("index_logs", "Embed recent activity-log events into vector memory",
             hard=(BACKEND_DB, BACKEND_EMBED)),
    # --- Profiles / plugins ---
    ToolSpec("list_profiles", "List session profiles",
             hard=(BACKEND_DB,)),
    ToolSpec("create_profile", "Create a session profile",
             hard=(BACKEND_DB,)),
    ToolSpec("switch_profile", "Switch session profile",
             hard=(BACKEND_DB,)),
    ToolSpec("delete_profile", "Delete a session profile",
             hard=(BACKEND_DB,)),
    ToolSpec("toggle_plugin", "Enable or disable a plugin",
             hard=()),
    ToolSpec("auto_curate_plugins", "Suggest stale plugins to disable",
             hard=(BACKEND_DB,)),
    # --- Router (bandit + prompt rewriting) ---
    ToolSpec("route_to_model", "Bandit-pick a model tier for a task",
             hard=(BACKEND_DB,)),
    ToolSpec("record_model_reward", "Record reward for a model trial",
             hard=(BACKEND_DB,)),
    ToolSpec("bandit_stats", "Show bandit per-tier stats",
             hard=(BACKEND_DB,)),
    ToolSpec("improve_prompt", "Run a prompt rewriter pipeline",
             hard=(), soft=(BACKEND_REASON_LLM,)),
    ToolSpec("list_prompt_rewriters", "List available prompt rewriters",
             hard=()),
    ToolSpec("analyze_router_log", "Analyze the router decision log",
             hard=()),
    # --- LLM-driven helpers ---
    ToolSpec("compact_master_state", "Compact MEMORY.md with the local LLM",
             hard=(BACKEND_REASON_LLM,)),
    ToolSpec("optimize_memory", "LLM recommendations to prune/compact memory",
             hard=(BACKEND_REASON_LLM,)),
    ToolSpec("optimize_plugin_memory", "Per-plugin memory optimization",
             hard=(BACKEND_REASON_LLM,)),
    ToolSpec("promote_memory", "Promote inbox notes via the local LLM",
             hard=(BACKEND_REASON_LLM,)),
    ToolSpec("exhaustion_save", "Auto-save when Claude is unreachable",
             hard=(BACKEND_DB,), soft=(BACKEND_REASON_LLM,)),
    ToolSpec("remember_identity", "Persist an identity fact",
             hard=(BACKEND_DB,)),
    ToolSpec("get_session_memory", "Read the current session memory",
             hard=(BACKEND_DB,)),
    ToolSpec("rebuild_session_memory", "Rebuild session memory from logs",
             hard=(BACKEND_DB,)),
    ToolSpec("wake_session", "Stateless recovery — event replay + cache rebuild",
             hard=(BACKEND_DB,)),
    # --- Configuration & status ---
    ToolSpec("configure", "View or update a config key",
             hard=()),
    ToolSpec("status", "Skill Hub health summary",
             hard=()),
    ToolSpec("render_dashboard", "Render the static HTML dashboard",
             hard=(BACKEND_DB,)),
    ToolSpec("token_stats", "Token savings report from hook interceptions",
             hard=(BACKEND_DB,)),
    ToolSpec("list_models", "List installed Ollama models",
             hard=(), soft=(BACKEND_OLLAMA,)),
    ToolSpec("pull_model", "Pull an Ollama model",
             hard=(BACKEND_OLLAMA,)),
    # --- Cron / core tasks ---
    ToolSpec("list_core_tasks", "List built-in scheduled core tasks",
             hard=()),
    ToolSpec("list_plugin_tasks", "List plugin-contributed scheduled tasks",
             hard=()),
    ToolSpec("enable_core_task", "Enable a built-in scheduled task",
             hard=()),
    ToolSpec("disable_core_task", "Disable a built-in scheduled task",
             hard=()),
    ToolSpec("enable_plugin_task", "Enable a plugin scheduled task",
             hard=()),
    ToolSpec("disable_plugin_task", "Disable a plugin scheduled task",
             hard=()),
    # --- Compression & tooling orchestration ---
    ToolSpec("retrieve_compressed", "Rehydrate content behind a reversible-compression marker",
             hard=()),
    ToolSpec("ensure_tooling", "Probe and provision dev tooling (e.g. code index) for a path",
             hard=()),
)


def _verdict(spec: ToolSpec, available: dict[str, bool]) -> str:
    """green = all hard+soft met; yellow = hard met, some soft missing; red = hard missing."""
    if any(not available.get(d, False) for d in spec.hard):
        return "red"
    if any(not available.get(d, False) for d in spec.soft):
        return "yellow"
    return "green"


def probe_backends() -> dict[str, dict]:
    """Run every backend probe once and return a name-keyed dict.

    Issue #6 — when ``no_llm_mode`` is on we *skip the probe* for the LLM-tier
    backends (embed/ollama/reason_llm/voyage) and report them as missing
    unconditionally. The non-LLM backends still run their normal probe so
    "DB available?" / "git installed?" remain truthful.
    """
    no_llm = _no_llm_mode()
    out: dict[str, dict] = {}
    for bid, b in BACKENDS.items():
        if no_llm and bid in _NO_LLM_DISABLED_BACKENDS:
            ok = False
        else:
            try:
                ok = bool(b.check())
            except Exception:  # noqa: BLE001
                ok = False
        out[bid] = {
            "id": b.id,
            "label": b.label,
            "ok": ok,
            "setup": b.setup,
        }
    return out


# ---------------------------------------------------------------------------
# Issue #7 — coarse tier registry ("none" / "embedding" / "llm")
#
# A simpler axis on top of the backend-dep matrix. Answers "what does this
# tool need at minimum?" without enumerating every backend. Used by
# ``status``, ``list_skills``, and ``--help`` to filter tools by the
# current install's capability floor.

Tier = Literal["none", "embedding", "llm"]

_F = TypeVar("_F", bound=Callable[..., object])


def tier_from_spec(spec: ToolSpec) -> Tier:
    """Derive the coarse tier from a ToolSpec's *hard* dependencies.

    Hard deps decide tier; soft deps degrade verdict (yellow) but don't
    change what the tool fundamentally needs. Priority: ``llm`` > ``embedding``
    > ``none`` — if a tool needs both, the stricter requirement wins.
    """
    if BACKEND_REASON_LLM in spec.hard:
        return "llm"
    if BACKEND_EMBED in spec.hard:
        return "embedding"
    return "none"


# Module-level registry consulted by ``status`` and friends. Auto-populated
# from TOOLS at import time so every declared tool has a tier — no tool can
# end up ``unknown``. ``requires_capability`` may override an entry inline.
TIER_REGISTRY: dict[str, Tier] = {
    spec.name: tier_from_spec(spec) for spec in TOOLS
}


def requires_capability(tier: Tier) -> Callable[[_F], _F]:
    """Decorator: declare the capability tier a tool needs at minimum.

    Stamps ``__capability_tier__`` on the wrapped function and records the
    declaration in :data:`TIER_REGISTRY`. The declared tier wins over the
    one derived from the matching :class:`ToolSpec` (if any), letting
    server.py call out exceptions inline without editing the spec table.

    Usage in ``server.py``::

        @mcp.tool()
        @requires_capability("embedding")
        def search_skills(...): ...
    """
    if tier not in ("none", "embedding", "llm"):
        raise ValueError(f"tier must be none/embedding/llm, got {tier!r}")

    def deco(fn: _F) -> _F:
        fn.__capability_tier__ = tier  # type: ignore[attr-defined]
        TIER_REGISTRY[fn.__name__] = tier
        return fn

    return deco


def tier_for(tool_name: str) -> Tier:
    """Return the declared tier for ``tool_name``.

    Falls back to the spec-derived tier if the tool is in :data:`TOOLS`
    but somehow missing from :data:`TIER_REGISTRY` (paranoid path —
    shouldn't happen given the import-time population above). Raises
    ``KeyError`` for unknown tools so callers can't silently swallow
    a missing declaration.
    """
    if tool_name in TIER_REGISTRY:
        return TIER_REGISTRY[tool_name]
    for spec in TOOLS:
        if spec.name == tool_name:
            return tier_from_spec(spec)
    raise KeyError(f"no capability tier declared for tool {tool_name!r}")


def render_matrix() -> dict:
    """Return a dict ready for the Jinja template (and JSON consumers).

    Shape::
        {
          "backends": [{id, label, ok, setup}, ...],
          "tools":    [{name, summary, verdict, hard, soft, missing}, ...],
          "summary":  {"green": N, "yellow": N, "red": N, "total": N},
        }
    """
    backends = probe_backends()
    available = {bid: b["ok"] for bid, b in backends.items()}
    tool_rows: list[dict] = []
    summary = {"green": 0, "yellow": 0, "red": 0, "total": len(TOOLS)}
    for spec in TOOLS:
        verdict = _verdict(spec, available)
        missing = [
            backends[d]["label"]
            for d in (*spec.hard, *spec.soft)
            if d in backends and not available.get(d, False)
        ]
        tool_rows.append({
            "name": spec.name,
            "summary": spec.summary,
            "tier": TIER_REGISTRY.get(spec.name, tier_from_spec(spec)),
            "verdict": verdict,
            "hard": list(spec.hard),
            "soft": list(spec.soft),
            "missing": missing,
            "setup": [
                backends[d]["setup"]
                for d in (*spec.hard, *spec.soft)
                if d in backends and not available.get(d, False)
            ],
        })
        summary[verdict] += 1
    return {
        "backends": list(backends.values()),
        "tools": tool_rows,
        "summary": summary,
        "no_llm_mode": _no_llm_mode(),
    }


def no_llm_mode_active() -> bool:
    """Public predicate — True when ``config.no_llm_mode`` is on."""
    return _no_llm_mode()


def no_llm_summary() -> dict:
    """Counts of tools available vs disabled assuming ``no_llm_mode`` is on.

    Derives the answer from ``TOOLS`` (the single source of truth for
    "what does this tool need?") so the count stays accurate as new
    @mcp.tool entries are added. A tool is considered disabled by no_llm_mode
    iff at least one of its hard deps is in ``_NO_LLM_DISABLED_BACKENDS``.

    Returns::
        {
          "available": int,
          "disabled":  int,
          "total":     int,
          "disabled_tools": ["search_skills", ...],
          "available_tools": ["list_teachings", ...],
        }
    """
    disabled: list[str] = []
    available: list[str] = []
    for spec in TOOLS:
        if any(d in _NO_LLM_DISABLED_BACKENDS for d in spec.hard):
            disabled.append(spec.name)
        else:
            available.append(spec.name)
    return {
        "available": len(available),
        "disabled":  len(disabled),
        "total":     len(TOOLS),
        "disabled_tools":  disabled,
        "available_tools": available,
    }
