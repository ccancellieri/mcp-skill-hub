"""Main router orchestrator — called by the CLI 'route' command.

Flow:
  1. Tier 1 heuristics (always, <5ms)
  2. Tier 2 Ollama (if T1 confidence < 0.85)
  3. Tier 3 Haiku batch (if T2 confidence < 0.7 AND haiku enabled)
  4. Enforcement (write settings.json or suggest)
  5. Preload skills for domain_hints
  6. Compact advisor (estimate context pressure)
  7. Thin-prompt enrichment (if prompt is very short)
  8. Build and return output dict for the hook
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from .. import config as _cfg
from . import heuristics, ollama_client, haiku_client, enforcement, preloader
from .verdict import Verdict, format_system_message, append_audit_log


def _project_override(cfg_from_file: dict[str, Any], cwd: str) -> dict[str, Any]:
    """Read .skill-hub-router.yaml from cwd and apply overrides."""
    try:
        yaml_path = Path(cwd) / ".skill-hub-router.yaml"
        if not yaml_path.exists():
            return {}
        overrides: dict[str, Any] = {}
        for line in yaml_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" in line:
                key, _, val = line.partition(":")
                val = val.strip()
                key = key.strip()
                if val.lower() == "true":
                    overrides[key] = True
                elif val.lower() == "false":
                    overrides[key] = False
                else:
                    overrides[key] = val
        return overrides
    except OSError:
        return {}


def route(
    prompt: str,
    session_id: str = "",
    cwd: str = "",
) -> dict[str, Any]:
    """Classify *prompt* and return a hook-compatible output dict.

    Returns:
      {
        "decision": "allow",
        "systemMessage": "...",   # verdict header + skill hints
        "userMessage": "...",     # only present for thin-prompt enrichment
      }
    or {} if the router is disabled / nothing to say.
    """
    t0 = time.monotonic()

    cfg = _cfg.load_config()

    # Master kill-switch (env var takes priority over config)
    env_enabled = os.environ.get("SKILL_HUB_ROUTER_ENABLED", "")
    if env_enabled == "0" or (not env_enabled and not cfg.get("router_enabled", True)):
        return {}

    # Per-project override
    project_cfg = _project_override(cfg, cwd or os.getcwd())

    # Env var overrides
    services = cfg.setdefault("services", {})
    if os.environ.get("SKILL_HUB_ROUTER_OLLAMA_MODEL"):
        services.setdefault("ollama_router", {})["model"] = os.environ["SKILL_HUB_ROUTER_OLLAMA_MODEL"]
    if os.environ.get("SKILL_HUB_ROUTER_HAIKU") == "1":
        services.setdefault("haiku_router", {})["enabled"] = True
    if os.environ.get("SKILL_HUB_ROUTER_ENABLED") == "0":
        return {}

    # Increment session message counter (for compact advisor)
    msg_count = preloader.increment_message_counter(session_id) if session_id else 0

    # ── Tier 1: heuristics ──────────────────────────────────────────────────
    t1_start = time.monotonic()
    t1_signals = heuristics.classify(prompt)
    tier1_ms = int((time.monotonic() - t1_start) * 1000)

    complexity = t1_signals.complexity
    ambiguity  = t1_signals.ambiguity
    scope      = t1_signals.scope
    domain_hints = t1_signals.domain_hints
    confidence = t1_signals.confidence
    model      = t1_signals.model
    plan_mode  = t1_signals.plan_mode
    tier_used  = 1

    haiku_extras: dict[str, Any] = {}
    tier2_ms = 0
    tier3_ms = 0

    # ── Tier 2: Ollama local LLM ────────────────────────────────────────────
    t2_threshold: float = float(cfg.get("router_tier2_confidence_gate", 0.85))
    if confidence < t2_threshold:
        t2_start = time.monotonic()
        t2 = ollama_client.classify(prompt, cfg, cwd=cwd)
        tier2_ms = int((time.monotonic() - t2_start) * 1000)
        if t2 is not None:
            complexity   = t2.complexity
            ambiguity    = t2.ambiguity
            scope        = t2.scope
            domain_hints = list(set(domain_hints + t2.domain_hints))
            confidence   = t2.confidence
            fake = heuristics.HeuristicSignals(complexity=complexity, ambiguity=ambiguity)
            heuristics._derive_verdict(fake)
            model     = fake.model
            plan_mode = fake.plan_mode
            tier_used = 2

    # ── Tier 3: Haiku batched ────────────────────────────────────────────────
    t3_threshold: float = float(cfg.get("router_haiku_threshold", 0.7))
    if confidence < t3_threshold and haiku_client.is_enabled(cfg):
        t3_start = time.monotonic()
        h = haiku_client.classify(prompt, cfg, msg_count, cwd=cwd)
        tier3_ms = int((time.monotonic() - t3_start) * 1000)
        if h is not None:
            complexity   = h.complexity
            ambiguity    = h.ambiguity
            scope        = h.scope
            domain_hints = list(set(domain_hints + h.domain_hints))
            confidence   = h.confidence
            fake2 = heuristics.HeuristicSignals(complexity=complexity, ambiguity=ambiguity)
            heuristics._derive_verdict(fake2)
            model     = fake2.model
            plan_mode = fake2.plan_mode
            tier_used = 3
            haiku_extras = {
                "settings_opt": h.settings_opt,
                "compact_hint": h.compact_hint,
                "subtasks":     h.subtasks,
            }

    # ── Project-level model override (wins over classifier) ─────────────────
    if "model" in project_cfg:
        model     = str(project_cfg["model"])
        plan_mode = bool(project_cfg.get("plan_mode", plan_mode))
        reasoning = "project override (.skill-hub-router.yaml)"
    else:
        c_label     = "complex" if complexity >= 0.7 else ("trivial" if complexity < 0.3 else "moderate")
        a_label     = "ambiguous" if ambiguity >= 0.6 else "clear"
        scope_label = scope if scope != "single" else ""
        parts       = [p for p in [c_label, a_label, scope_label] if p]
        if domain_hints:
            parts.append(f"domain: {'+'.join(domain_hints[:2])}")
        reasoning = ", ".join(parts) if parts else "no strong signals"

    # ── Enforcement ─────────────────────────────────────────────────────────
    from .enforcement import _read_current_model
    prev_model = _read_current_model() or "sonnet"
    for alias in ("haiku", "sonnet", "opus"):
        if alias in prev_model:
            prev_model = alias
            break

    action, enforce_msg = enforcement.apply(
        verdict_model=model,
        plan_mode=plan_mode,
        confidence=confidence,
        session_id=session_id,
        cfg=cfg,
    )

    # ── Compact advisor ─────────────────────────────────────────────────────
    compact_hint: dict[str, Any] = haiku_extras.get("compact_hint", {})
    if not compact_hint:
        suggest_c, compact_reason = preloader.should_compact(session_id, cfg)
        if suggest_c:
            compact_hint = {"suggest_compact": True, "reason": compact_reason}

    # ── Skill preloading ─────────────────────────────────────────────────────
    max_skills: int = int(cfg.get("hook_context_top_k_skills", 3))
    skill_names, plugin_names = preloader.load_skills(domain_hints, cfg, top_k=max_skills)

    # ── Thin-prompt enrichment ───────────────────────────────────────────────
    enriched_msg = preloader.enrich_thin_prompt(prompt, session_id, cfg)
    enrichment_applied = enriched_msg is not None
    enrichment_source  = ""
    enrichment_chars   = 0
    if enriched_msg:
        # Detect which source was used from the injected prefix
        if "Session context:" in enriched_msg:
            if "session_context" in enriched_msg or "context_summary" in enriched_msg:
                enrichment_source = "session_context"
            else:
                enrichment_source = "recent_task"
        enrichment_chars = len(enriched_msg) - len(prompt)

    # ── Build verdict ────────────────────────────────────────────────────────
    latency_ms = int((time.monotonic() - t0) * 1000)
    v = Verdict(
        model=model,
        plan_mode=plan_mode,
        preload_skills=skill_names,
        preload_plugins=plugin_names,
        confidence=confidence,
        reasoning=reasoning,
        tier_used=tier_used,
        complexity=complexity,
        ambiguity=ambiguity,
        scope=scope,
        domain_hints=domain_hints,
        settings_opt=haiku_extras.get("settings_opt", {}),
        compact_hint=compact_hint,
        subtasks=haiku_extras.get("subtasks", []),
        enforcement=action,
        prev_model=prev_model,
        enrichment_applied=enrichment_applied,
        enrichment_source=enrichment_source,
        enrichment_chars=enrichment_chars,
        latency_ms=latency_ms,
        tier1_ms=tier1_ms,
        tier2_ms=tier2_ms,
        tier3_ms=tier3_ms,
        session_id=session_id,
        prompt_preview=prompt[:80],
    )

    # ── Audit log ────────────────────────────────────────────────────────────
    log_path: str = cfg.get(
        "router_log",
        str(Path.home() / ".claude" / "mcp-skill-hub" / "router.jsonl"),
    )
    append_audit_log(v, log_path)

    # ── Build hook output ────────────────────────────────────────────────────
    output: dict[str, Any] = {"decision": "allow"}

    parts: list[str] = []
    system_msg = format_system_message(v)
    if system_msg:
        parts.append(system_msg)
    if enforce_msg:
        parts.append(enforce_msg)
    if plan_mode and action != "hard_switch":
        parts.append("[Router] This prompt looks like it warrants plan mode (Shift+Tab).")

    if parts:
        output["systemMessage"] = "\n".join(parts)

    if enriched_msg:
        output["userMessage"] = enriched_msg

    # ── S5 F-PROMPT: apply prompt rewriters (opt-in via config) ──────────────
    if cfg.get("router_improve_prompt_enabled", False):
        try:
            from . import rewriters as _rw
            from ..store import SkillStore

            store = SkillStore()
            try:
                improved = _rw.improve_prompt(enriched_msg or prompt, store, cfg=cfg)
            finally:
                store.close()
            if improved.applied:
                output["userMessage"] = improved.prompt
        except Exception:
            pass

    if "systemMessage" not in output and "userMessage" not in output:
        return {}

    return output
