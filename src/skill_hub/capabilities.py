"""Capability matrix — what every MCP tool needs and whether it works now.

Issue #13 (M1 — Useful Without LLM): a stdlib-only, single-source-of-truth
view of "given the current backend state, which tools work?" Each tool
declares which backends it depends on; this module probes the backends
once per request and renders a green / yellow / red verdict.

This intentionally does NOT depend on a separate tool-capability-matrix
(issue #7) — the dependency table lives here as plain data so the
dashboard view can ship before #7 lands.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from typing import Callable

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


def _check_ollama() -> bool:
    """Daemon reachable, regardless of which models are installed."""
    try:
        from .ollama_client import get_ollama_client
        return get_ollama_client().get_api_base(None) is not None
    except Exception:  # noqa: BLE001
        return False


def _check_embed() -> bool:
    try:
        from .embeddings import embed_available
        return bool(embed_available())
    except Exception:  # noqa: BLE001
        return False


def _check_reason_llm() -> bool:
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
    # --- Fanout / swarm ---
    ToolSpec("fanout_issues", "Fan out N GitHub issues into worktree tasks",
             hard=(BACKEND_DB, BACKEND_GIT, BACKEND_GH),
             soft=(BACKEND_CLAUDE_CLI,)),
    ToolSpec("fanout_status", "Status of a fanout group",
             hard=(BACKEND_DB,)),
    ToolSpec("fanout_close", "Close all tasks in a fanout group",
             hard=(BACKEND_DB,)),
    ToolSpec("fanout_cleanup", "Remove worktrees for a fanout group",
             hard=(BACKEND_DB, BACKEND_GIT)),
    ToolSpec("swarm_launch", "Launch N Claude subprocesses on claims",
             hard=(BACKEND_DB, BACKEND_CLAUDE_CLI)),
    ToolSpec("swarm_reap", "Poll swarm subprocess handles",
             hard=(BACKEND_DB,)),
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
    # --- Planning ---
    ToolSpec("validate_plan", "Static validation of a plan YAML",
             hard=()),
    ToolSpec("author_plan", "Draft a plan from a goal",
             hard=(BACKEND_REASON_LLM,)),
    ToolSpec("run_plan", "Run/dry-run a plan YAML",
             hard=(), soft=(BACKEND_CLAUDE_CLI,)),
    ToolSpec("execute_plan_step", "Execute one plan step",
             hard=(BACKEND_REASON_LLM,)),
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
    # --- Autopilot ---
    ToolSpec("autopilot_run", "Run one autopilot iteration",
             hard=(BACKEND_DB,), soft=(BACKEND_CLAUDE_CLI,)),
    ToolSpec("autopilot_stop", "Stop the autopilot loop",
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
    """Run every backend probe once and return a name-keyed dict."""
    out: dict[str, dict] = {}
    for bid, b in BACKENDS.items():
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
    }
