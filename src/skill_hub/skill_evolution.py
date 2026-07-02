"""L2 skill evolution: draft skill-version proposals from feedback (#137).

The ``skill_versions`` table and ``skill_evolution_*`` config keys existed but
nothing drove them. This wires the feedback signals already collected
(``skill_injections`` + the ``feedback`` table, aggregated by
``get_skill_usage_stats``) to the escalation ladder: pick skills that get
injected but rated unhelpful (or below the neutral feedback score), ask the
ladder (op=evolve_skill) to draft an improved body, and record it as a
*proposal* in ``skill_versions`` — with a change_reason. The on-disk SKILL.md
is never touched; skills live in read-only plugin caches, so a human reviews
proposals via ``get_evolved_skills_summary`` before anything is adopted.
"""
from __future__ import annotations

import json
import logging

from . import config as _cfg

log = logging.getLogger(__name__)

_EVOLVE_PROMPT = """You improve a reusable "skill" — an instruction block an AI
coding agent loads to do a task better. This skill keeps getting selected but
rated unhelpful, so it needs a sharper version.

Current skill:
=== BEGIN SKILL ===
{content}
=== END SKILL ===

Feedback signal: injected {injections} times, helpful {helpful}, unhelpful
{unhelpful}, feedback score {feedback_score}.

Rewrite the skill so it is clearer, more actionable, and more likely to be
useful for its stated purpose. Keep the same scope — do NOT invent new
capabilities. Return ONLY a JSON object:
{{"content": "<the full improved skill body>", "reason": "<one short clause on what you changed and why>"}}
"""

_FEEDBACK_FLOOR = 0.9   # skills below neutral (1.0) are evolution candidates


def candidates(store, *, limit: int = 5, min_injections: int = 3) -> list[dict]:
    """Skills worth evolving: enough exposure, but a poor helpfulness signal.

    Sorted worst-first (lowest feedback score, then most unhelpful votes).
    """
    try:
        stats = store.get_skill_usage_stats()
    except Exception as exc:  # noqa: BLE001
        log.debug("skill_evolution: usage stats failed: %s", exc)
        return []

    picked = []
    for s in stats:
        injections = s.get("injections") or 0
        if injections < min_injections:
            continue
        helpful = s.get("helpful") or 0
        unhelpful = s.get("unhelpful") or 0
        score = s.get("feedback_score") or 1.0
        poor = score < _FEEDBACK_FLOOR or (unhelpful > 0 and unhelpful >= helpful)
        if poor:
            picked.append(s)

    picked.sort(key=lambda s: (s.get("feedback_score") or 1.0,
                               -(s.get("unhelpful") or 0)))
    return picked[:limit]


def propose_evolution(store, skill_id: str, *, reason: str = "") -> dict | None:
    """Draft an improved version of ``skill_id`` and record it as a proposal.

    Returns ``{skill_id, version, change_reason}`` or None when no skill/content
    is found or the ladder returns nothing usable. Never overwrites the disk file.
    """
    skill = store.get_skill(skill_id)
    content = (skill or {}).get("content") or store.get_skill_content(skill_id)
    if not content:
        return None

    stats = {s["id"]: s for s in store.get_skill_usage_stats()}.get(skill_id, {})
    prompt = _EVOLVE_PROMPT.format(
        content=content[:6000],
        injections=stats.get("injections", "?"),
        helpful=stats.get("helpful", "?"),
        unhelpful=stats.get("unhelpful", "?"),
        feedback_score=round(stats.get("feedback_score", 1.0), 2),
    )

    from .llm.request import request
    try:
        out = request("smart", prompt, op="evolve_skill",
                      timeout=90, max_tokens=1200)
    except Exception as exc:  # noqa: BLE001
        log.debug("skill_evolution: ladder draft failed for %s: %s", skill_id, exc)
        return None

    proposed = _parse_proposal(out)
    if not proposed or not proposed.get("content"):
        return None
    new_body = proposed["content"]
    if new_body.strip() == content.strip():
        return None   # no change → not a proposal

    change_reason = (reason + " " if reason else "") + str(proposed.get("reason", ""))
    version = store.save_skill_version(
        skill_name=skill_id,
        skill_json=json.dumps({"proposed_content": new_body, "source": "ladder"}),
        change_reason=change_reason.strip()[:500],
        local_example=json.dumps(stats)[:500],
    )
    return {"skill_id": skill_id, "version": version,
            "change_reason": change_reason.strip()[:200]}


def _parse_proposal(text: str) -> dict | None:
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


def run_evolution(store, *, limit: int | None = None) -> list[dict]:
    """Batch pass — gated by ``skill_evolution_auto``. Returns proposals made."""
    if not _cfg.get("skill_evolution_auto"):
        return []
    cap = limit if limit is not None else int(
        _cfg.get("skill_evolution_max_per_session") or 3
    )
    made = []
    for c in candidates(store, limit=cap):
        try:
            proposal = propose_evolution(store, c["id"],
                                         reason="auto: low feedback signal")
        except Exception as exc:  # noqa: BLE001
            log.debug("skill_evolution: propose failed for %s: %s", c.get("id"), exc)
            continue
        if proposal:
            made.append(proposal)
    if made:
        log.info("skill_evolution: %d proposal(s) drafted", len(made))
    return made
