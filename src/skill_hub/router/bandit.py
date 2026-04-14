"""S4 F-ROUTE — ε-greedy bandit over model tiers.

Picks between ``tier_cheap``, ``tier_mid``, ``tier_smart`` keyed by a
``(task_class, domain)`` bucket. Rewards are recorded via
``record_reward(tier, task_class, domain, success)``; selection uses the
tier with the highest success rate with probability ``1-ε``, and a uniform
random tier with probability ``ε``.

Bucketing rules (kept small on purpose):
- ``task_class`` ∈ {trivial, simple, moderate, complex} derived from
  ``HeuristicSignals.complexity``.
- ``domain`` = first element of ``domain_hints`` or ``"_none"``.

Rewards live in the ``model_rewards`` table. Counts are partial-reward
friendly (success stored as REAL so 0.5 is a valid "meh" signal).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from .. import config as _cfg

_TIERS = ("tier_cheap", "tier_mid", "tier_smart")
_DEFAULT_EPSILON = 0.1
_DEFAULT_PRIOR_TRIALS = 1.0          # Laplace smoothing
_DEFAULT_PRIOR_SUCCESSES = 0.7        # assume tiers work most of the time
_MIN_TRIALS_FOR_EXPLOIT = 3           # below this, still sample widely


@dataclass
class RouteDecision:
    tier: str
    model: str
    confidence: float
    reasoning: str
    stats: dict[str, Any]


def bucket(complexity: float, domain_hints: list[str] | None) -> tuple[str, str]:
    """Derive ``(task_class, domain)`` bucket from heuristic signals."""
    c = max(0.0, min(1.0, float(complexity)))
    if c < 0.15:
        task_class = "trivial"
    elif c < 0.4:
        task_class = "simple"
    elif c < 0.7:
        task_class = "moderate"
    else:
        task_class = "complex"
    domain = (domain_hints or [None])[0] or "_none"
    return task_class, str(domain)


def _tier_model(tier: str) -> str:
    providers = _cfg.get("llm_providers") or {}
    if isinstance(providers, dict) and providers.get(tier):
        return str(providers[tier])
    return ""


def _fetch_stats(store: Any, task_class: str, domain: str) -> dict[str, dict[str, float]]:
    rows = store._conn.execute(
        "SELECT tier, trials, successes FROM model_rewards"
        " WHERE task_class = ? AND domain = ?",
        (task_class, domain),
    ).fetchall()
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        out[r["tier"]] = {"trials": float(r["trials"]), "successes": float(r["successes"])}
    return out


def _success_rate(trials: float, successes: float) -> float:
    prior_t = _DEFAULT_PRIOR_TRIALS
    prior_s = _DEFAULT_PRIOR_SUCCESSES
    return (successes + prior_s) / (trials + prior_t)


def _default_tier(task_class: str) -> str:
    """Fallback when no stats exist yet — pick by complexity."""
    return {
        "trivial": "tier_cheap",
        "simple": "tier_cheap",
        "moderate": "tier_mid",
        "complex": "tier_smart",
    }.get(task_class, "tier_cheap")


def select_tier(
    store: Any,
    complexity: float,
    domain_hints: list[str] | None,
    *,
    epsilon: float | None = None,
    allow_exploration: bool = True,
) -> RouteDecision:
    """Return the chosen tier for this bucket.

    Uses ε-greedy with Laplace-smoothed success rates. If no tier has
    cleared ``_MIN_TRIALS_FOR_EXPLOIT`` trials yet we stay in pure explore
    mode (weighted toward the complexity-based default so cold-start is
    sensible).
    """
    eps = float(epsilon) if epsilon is not None else float(
        _cfg.get("router_bandit_epsilon") or _DEFAULT_EPSILON
    )
    task_class, domain = bucket(complexity, domain_hints)
    stats = _fetch_stats(store, task_class, domain)

    default = _default_tier(task_class)
    trial_counts = {t: stats.get(t, {}).get("trials", 0.0) for t in _TIERS}
    max_trials = max(trial_counts.values()) if trial_counts else 0.0

    reasoning: str
    if allow_exploration and max_trials < _MIN_TRIALS_FOR_EXPLOIT:
        # Cold start — use default tier with small random exploration.
        if random.random() < eps:
            chosen = random.choice(_TIERS)
            reasoning = f"cold-start explore (max_trials={max_trials:.0f})"
        else:
            chosen = default
            reasoning = f"cold-start default for task_class={task_class}"
        confidence = 0.3
    elif allow_exploration and random.random() < eps:
        chosen = random.choice(_TIERS)
        reasoning = f"ε-explore (ε={eps:.2f})"
        confidence = 0.4
    else:
        rates = {t: _success_rate(trial_counts[t], stats.get(t, {}).get("successes", 0.0))
                 for t in _TIERS}
        chosen = max(rates, key=lambda k: rates[k])
        confidence = max(0.3, min(0.95, rates[chosen]))
        reasoning = (
            f"exploit best-rate tier ({chosen} @ {rates[chosen]:.2f}, "
            f"trials={trial_counts[chosen]:.0f})"
        )

    return RouteDecision(
        tier=chosen,
        model=_tier_model(chosen),
        confidence=confidence,
        reasoning=reasoning,
        stats={
            "task_class": task_class,
            "domain": domain,
            "per_tier": {
                t: {
                    "trials": trial_counts[t],
                    "successes": stats.get(t, {}).get("successes", 0.0),
                    "rate": _success_rate(trial_counts[t], stats.get(t, {}).get("successes", 0.0)),
                }
                for t in _TIERS
            },
        },
    )


def record_reward(
    store: Any,
    tier: str,
    task_class: str,
    domain: str,
    success: float,
) -> None:
    """Add one trial for ``(task_class, domain, tier)`` with ``success``
    in ``[0.0, 1.0]``. Partial rewards allowed (0.5 = mixed outcome).
    """
    if tier not in _TIERS:
        raise ValueError(f"unknown tier: {tier!r}")
    s = max(0.0, min(1.0, float(success)))
    store._conn.execute(
        """
        INSERT INTO model_rewards (task_class, domain, tier, trials, successes, updated_at)
        VALUES (?, ?, ?, 1, ?, datetime('now'))
        ON CONFLICT(task_class, domain, tier) DO UPDATE SET
            trials = trials + 1,
            successes = successes + excluded.successes,
            updated_at = datetime('now')
        """,
        (task_class, domain, tier, s),
    )
    store._conn.commit()


def summary(store: Any) -> list[dict[str, Any]]:
    rows = store._conn.execute(
        "SELECT task_class, domain, tier, trials, successes, updated_at"
        " FROM model_rewards ORDER BY trials DESC"
    ).fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append({
            "task_class": r["task_class"],
            "domain": r["domain"],
            "tier": r["tier"],
            "trials": int(r["trials"]),
            "successes": float(r["successes"]),
            "rate": _success_rate(float(r["trials"]), float(r["successes"])),
            "updated_at": r["updated_at"],
        })
    return out
