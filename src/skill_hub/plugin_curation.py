"""L1/L2 autonomous plugin curation.

Lets the escalation ladder's cheap tiers (L1 local Ollama / L2 remote gateway)
decide whether a stale *enabled* plugin should be disabled — and, when
``plugin_curation_auto`` is set, apply that decision server-side by flipping the
bit in ``~/.claude/settings.json`` via :func:`plugin_registry.toggle`, without
the main frontier LLM (the interactive Claude) in the loop.

Why only disable is automated:

* **Disable** has an unambiguous signal — a plugin is enabled in settings.json
  but has had no ``session_log`` activity in the staleness window
  (``profiles.auto_curate_candidates``). The cheap tier only has to confirm the
  plugin is genuinely dormant, so the decision is safe to automate.
* **Enable** is intentionally left to :func:`server.suggest_plugins` and the
  main LLM: autonomously enabling a *new* plugin requires knowing the upcoming
  task, which a background loop does not have. The decision op below accepts an
  ``enable`` action too (the mechanism is symmetric), but candidate generation
  for enable is deferred — see the module docstring in ``profiles``.

The decision is a single-turn structured-output call (``op=plugin_curation``,
routed to the *cheap* tier by ``_OP_ROUTING``), so the main coding loop never
pays for it. Nothing is written unless ``plugin_curation_auto`` is True; the
gate mirrors ``skill_evolution_auto`` (#137) and ``memory_supersede`` (#136).
"""
from __future__ import annotations

import json
import logging

from . import config as _cfg

log = logging.getLogger(__name__)

_DECIDE_PROMPT = """You maintain a developer's set of enabled Claude Code plugins.
A plugin below is currently ENABLED but shows no recent usage. Decide whether it
should be disabled to reduce clutter and token overhead, or kept.

Plugin: {plugin_id}
Last used: {last_used_at}
Sessions in the last {stale_days} days: {sessions}
Also currently enabled: {enabled_list}

Disable it only if it is clearly dormant and nothing in the enabled set suggests
it is a dependency of active work. When unsure, keep it. Return ONLY a JSON
object, no prose:
{{"action": "disable" | "keep", "reason": "<one short clause>"}}
"""


def candidates(store, *, stale_days: int | None = None) -> list[dict]:
    """Enabled-but-dormant plugins worth a disable decision.

    Thin wrapper over :func:`profiles.auto_curate_candidates` so callers depend
    on this module rather than reaching into ``profiles`` directly.
    """
    from . import profiles
    days = int(stale_days if stale_days is not None
               else _cfg.get("plugin_curation_stale_days") or 14)
    try:
        return profiles.auto_curate_candidates(store, stale_days=days)
    except Exception as exc:  # noqa: BLE001
        log.debug("plugin_curation: candidate scan failed: %s", exc)
        return []


def decide(store, candidate: dict, *, stale_days: int | None = None) -> dict | None:
    """Ask the cheap ladder tier (L1/L2) whether ``candidate`` should be disabled.

    Returns ``{plugin_id, action, reason}`` (action is ``disable`` or ``keep``)
    or None when the ladder returns nothing usable. Makes no changes.
    """
    pid = candidate.get("plugin_id")
    if not pid:
        return None

    from . import plugin_registry
    enabled = sorted(k for k, v in plugin_registry._enabled_map().items() if v)
    days = int(stale_days if stale_days is not None
               else _cfg.get("plugin_curation_stale_days") or 14)
    prompt = _DECIDE_PROMPT.format(
        plugin_id=pid,
        last_used_at=candidate.get("last_used_at") or "never",
        sessions=candidate.get("sessions_last_window", 0),
        stale_days=days,
        enabled_list=", ".join(p for p in enabled if p != pid) or "(none)",
    )

    from .llm.request import request
    try:
        out = request("cheap", prompt, op="plugin_curation",
                      timeout=45, max_tokens=200)
    except Exception as exc:  # noqa: BLE001
        log.debug("plugin_curation: ladder decision failed for %s: %s", pid, exc)
        return None

    parsed = _parse_decision(out)
    if not parsed:
        return None
    action = str(parsed.get("action", "")).strip().lower()
    if action not in ("disable", "keep"):
        return None
    return {"plugin_id": pid, "action": action,
            "reason": str(parsed.get("reason", ""))[:200]}


def _parse_decision(text: str) -> dict | None:
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def run_curation(store, *, apply: bool | None = None,
                 limit: int | None = None) -> list[dict]:
    """Decide (and optionally apply) disable actions for dormant plugins.

    Gated by ``plugin_curation_enabled``. ``apply`` defaults to
    ``plugin_curation_auto``; when True, disable decisions are written to
    ``~/.claude/settings.json`` server-side via :func:`plugin_registry.toggle`
    (a restart applies them). Returns one dict per decision:
    ``{plugin_id, action, reason, applied}``.
    """
    if not _cfg.get("plugin_curation_enabled"):
        return []
    do_apply = bool(_cfg.get("plugin_curation_auto")) if apply is None else bool(apply)
    cap = int(limit if limit is not None
              else _cfg.get("plugin_curation_max_per_session") or 5)

    results: list[dict] = []
    for cand in candidates(store)[:cap]:
        decision = decide(store, cand)
        if not decision:
            continue
        applied = False
        if decision["action"] == "disable" and do_apply:
            from . import plugin_registry
            try:
                plugin_registry.toggle(decision["plugin_id"], False)
                applied = True
            except Exception as exc:  # noqa: BLE001
                log.debug("plugin_curation: toggle failed for %s: %s",
                          decision["plugin_id"], exc)
        results.append({**decision, "applied": applied})

    if results:
        n_disable = sum(1 for r in results if r["action"] == "disable")
        n_applied = sum(1 for r in results if r["applied"])
        log.info("plugin_curation: %d decision(s), %d disable, %d applied",
                 len(results), n_disable, n_applied)
    return results
