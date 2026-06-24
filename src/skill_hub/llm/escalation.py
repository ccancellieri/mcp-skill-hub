"""Availability-, complexity-, and quota-aware selection over the registry.

Walks ``load_registry()`` in order; for each provider resolves credentials and
skips it when a required key is missing or when the candidate model is on
cooldown. Picks a model whose complexity class matches the task. On a quota /
429 signal the caller marks the model on cooldown and re-selects.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass

from .. import config as _cfg
from .credentials import resolve_credentials
from .registry import Provider, ProviderModel, load_registry

_COOLDOWN: dict[str, float] = {}   # model id → epoch when it may be retried

_QUOTA_RE = re.compile(r"\b429\b|rate.?limit|quota|insufficient_quota|overloaded", re.I)


@dataclass
class Selection:
    model: str
    api_base: str | None
    api_key: str | None
    provider: str
    level: str


def reset_cooldowns() -> None:
    _COOLDOWN.clear()


def mark_cooldown(model: str, *, seconds: int | None = None) -> None:
    ttl = seconds if seconds is not None else int(_cfg.get("llm_cooldown_seconds") or 3600)
    _COOLDOWN[model] = time.time() + ttl


def is_cooled(model: str, *, now: float | None = None) -> bool:
    exp = _COOLDOWN.get(model)
    if exp is None:
        return False
    if (now or time.time()) >= exp:
        _COOLDOWN.pop(model, None)
        return False
    return True


def looks_like_quota_error(exc: Exception) -> bool:
    return bool(_QUOTA_RE.search(str(exc)))


def _wanted_class(complexity: float) -> str:
    return "heavy" if float(complexity) >= 0.5 else "light"


def _provider_needs_key(p: Provider) -> bool:
    # Local ollama needs no key; everything else does.
    return p.kind != "ollama"


def _pick_model(p: Provider, wanted: str, exclude: set[str]) -> ProviderModel | None:
    avail = [m for m in p.models if m.id not in exclude and not is_cooled(m.id)]
    if not avail:
        return None
    for m in avail:
        if m.complexity == wanted:
            return m
    return avail[0]   # fall back to any available model in this provider


def select(complexity: float, *, exclude: set[str] | None = None) -> Selection | None:
    exclude = exclude or set()
    wanted = _wanted_class(complexity)
    for p in load_registry():
        api_base, api_key = resolve_credentials(p)
        if _provider_needs_key(p) and not api_key:
            continue
        m = _pick_model(p, wanted, exclude)
        if m is None:
            continue
        return Selection(model=m.id, api_base=api_base, api_key=api_key,
                         provider=p.name, level=p.level)
    return None
