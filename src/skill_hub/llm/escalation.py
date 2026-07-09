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

# --- pass-level circuit breaker (#139) --------------------------------------
# A single caller retrying every provider on every call can hot-loop the whole
# registry when nothing is reachable (seen during the 2026-07 WAL-starvation
# incident: ~185 error events in 10 minutes). When one full pass over the
# ladder ends with every attempted provider erroring, trip the breaker so the
# next calls fail fast instead of re-walking a registry that just proved dead.
_BREAKER_COOLDOWN_SECONDS = 600   # 10 min
_breaker_until: float = 0.0


def breaker_tripped(*, now: float | None = None) -> bool:
    """True while the circuit breaker is open (calls should fail fast)."""
    return (now if now is not None else time.time()) < _breaker_until


def record_ladder_pass(success: bool) -> None:
    """Update the breaker after one full pass over the ladder.

    ``success=True`` (any provider served the call) resets the breaker.
    ``success=False`` (every attempted provider errored) opens it for
    ``_BREAKER_COOLDOWN_SECONDS``.
    """
    global _breaker_until
    if success:
        _breaker_until = 0.0
    else:
        _breaker_until = time.time() + _BREAKER_COOLDOWN_SECONDS


def reset_breaker() -> None:
    global _breaker_until
    _breaker_until = 0.0


@dataclass
class Selection:
    model: str
    api_base: str | None
    api_key: str | None
    provider: str
    kind: str = ""
    personal: bool = False


def reset_cooldowns() -> None:
    _COOLDOWN.clear()


def mark_cooldown(model: str, *, seconds: int | None = None) -> None:
    ttl = seconds if seconds is not None else int(_cfg.get("llm_cooldown_seconds") or 3600)
    _COOLDOWN[model] = time.time() + ttl


def is_cooled(model: str, *, now: float | None = None) -> bool:
    exp = _COOLDOWN.get(model)
    if exp is None:
        return False
    if (now if now is not None else time.time()) >= exp:
        _COOLDOWN.pop(model, None)
        return False
    return True


def looks_like_quota_error(exc: Exception) -> bool:
    return bool(_QUOTA_RE.search(str(exc)))


# --- local (Ollama) reachability ------------------------------------------
# The local daemon is auto-killed under load. Rather than pin a local model,
# try it, and fail every call, we probe once (cached) and, when it is down,
# cool the local level so the ladder skips straight to the first reachable
# remote level (L0 → Lx). The probe stays out of ``select()`` so the pure
# selection logic — and its tests — are unaffected.
_REACH_CACHE: dict[str, tuple[float, bool]] = {}   # "ollama" → (expiry, reachable)
_REACH_TTL = 30.0


def reset_reachability() -> None:
    _REACH_CACHE.clear()


def ollama_daemon_reachable(*, ttl: float = _REACH_TTL) -> bool:
    """Cheap, cached probe: is the local Ollama daemon answering right now?

    Cached for ``ttl`` seconds when the daemon is up, or for
    ``ollama_down_probe_ttl_seconds`` (config, default 120s) when it is down.
    A refused connection (daemon stopped/killed under load) returns fast; the
    longer down-TTL prevents bursts of doomed calls from re-probing on every
    attempt.
    """
    now = time.time()
    hit = _REACH_CACHE.get("ollama")
    if hit is not None and now < hit[0]:
        return hit[1]
    ok = False
    try:
        import httpx
        from ..ollama_client import get_ollama_client
        base = get_ollama_client().get_api_base(None)
        if base:
            httpx.get(f"{base}/api/tags", timeout=2.0)
            ok = True
    except Exception:  # noqa: BLE001 - any failure means "treat as down"
        ok = False
    if ok:
        cache_ttl = ttl
    else:
        cache_ttl = float(_cfg.get("ollama_down_probe_ttl_seconds") or 120)
    _REACH_CACHE["ollama"] = (now + cache_ttl, ok)
    return ok


def has_remote_provider() -> bool:
    """Cheap, network-free check: is at least one non-local provider configured
    with resolvable credentials?

    Walks the registry and resolves credentials (env / config lookups only — no
    HTTP). Used by the hot path to decide whether firing an async remote-ladder
    enrichment worker can succeed before spending a subprocess on it. Returns
    False when only local Ollama is configured, so a down daemon degrades to
    deterministic output instead of spawning a doomed worker.
    """
    for p in load_registry():
        if p.kind == "ollama":
            continue
        _, api_key = resolve_credentials(p)
        if api_key:
            return True
    return False


def cool_ollama(*, seconds: int = 30) -> None:
    """Put every local (ollama-kind) model on a short cooldown so the ladder
    skips the dead local level instead of trying and failing on each call.

    A short TTL (default 30s) lets local resume quickly once load drops and the
    daemon is back, without re-probing on every call in between.
    """
    for p in load_registry():
        if p.kind == "ollama":
            for m in p.models:
                mark_cooldown(m.id, seconds=seconds)


def _wanted_class(complexity: float) -> str:
    return "heavy" if float(complexity) >= 0.5 else "light"


def _provider_needs_key(p: Provider) -> bool:
    # Local ollama needs no key; everything else does.
    return p.kind != "ollama"


def _pick_model(p: Provider, wanted: str, exclude: set[str],
                domain: str | None) -> ProviderModel | None:
    avail = [m for m in p.models
             if not m.embed and m.id not in exclude and not is_cooled(m.id)]
    if domain is not None:
        avail = [m for m in avail if domain in m.tags]
    if not avail:
        return None
    for m in avail:
        if m.complexity == wanted:
            return m
    return avail[0]   # fall back to any available model in this provider


def _walk(wanted: str, exclude: set[str], domain: str | None) -> Selection | None:
    """Walk the ladder in `order`; return the first usable model.

    When `domain` is set it acts as a hard filter (cheapest tier carrying a
    matching specialist wins); the caller falls back to a domain-agnostic walk
    when no specialist is available anywhere.
    """
    for p in load_registry():
        api_base, api_key = resolve_credentials(p)
        if _provider_needs_key(p) and not api_key:
            continue
        m = _pick_model(p, wanted, exclude, domain)
        if m is None:
            continue
        return Selection(model=m.id, api_base=api_base, api_key=api_key,
                         provider=p.name, kind=p.kind, personal=p.personal)
    return None


def select(complexity: float, *, domain: str | None = None,
           exclude: set[str] | None = None) -> Selection | None:
    exclude = exclude or set()
    wanted = _wanted_class(complexity)
    dom = (domain or "").strip().lower() or None
    if dom is not None:
        hit = _walk(wanted, exclude, dom)
        if hit is not None:
            return hit
        # No specialist for this domain is reachable — degrade gracefully to
        # the normal availability/complexity ladder rather than failing.
    return _walk(wanted, exclude, None)


def pick_model_for(domain: str, *, complexity: float = 0.5,
                   exclude: set[str] | None = None) -> Selection | None:
    """Reusable selector: best reachable model for a named *domain*.

    Thin wrapper over :func:`select` for callers (skills, agents, CLI) that want
    a domain-tuned model without thinking about the ladder. ``complexity``
    defaults to mid; pass higher for harder tasks. Returns None only when the
    whole registry is unreachable.
    """
    return select(complexity, domain=domain, exclude=exclude)


# --- embed lane (#134) ------------------------------------------------------
# Chat and embeddings are separate lanes over the same registry: a model
# record flagged ``embed: true`` is never picked for chat (see ``_pick_model``)
# and only those records are candidates here. Anthropic has no embeddings API,
# so anthropic-kind providers are skipped.

def _embed_candidates(p: Provider, exclude: set[str]) -> ProviderModel | None:
    for m in p.models:
        if m.embed and m.id not in exclude and not is_cooled(m.id):
            return m
    return None


def select_embed(*, exclude: set[str] | None = None) -> Selection | None:
    """First usable embedding model walking the ladder in ``order``.

    The caller owns dimension compatibility (an index built with one model
    cannot be queried with another's vectors) and marks failed models on
    cooldown before re-selecting with ``exclude``.
    """
    exclude = exclude or set()
    for p in load_registry():
        if p.kind == "anthropic":
            continue
        if p.kind == "ollama" and not p.api_base:
            continue   # local daemon — the plain "ollama" backend's job
        api_base, api_key = resolve_credentials(p)
        if _provider_needs_key(p) and not api_key:
            continue
        m = _embed_candidates(p, exclude)
        if m is None:
            continue
        return Selection(model=m.id, api_base=api_base, api_key=api_key,
                         provider=p.name, kind=p.kind, personal=p.personal)
    return None


def has_ladder_embed_provider() -> bool:
    """Network-free: is any *remote* embed-capable provider configured?

    Remote means an ollama-kind provider with an explicit ``api_base`` (a
    second Ollama host) or a credentialed openai_compatible provider. The
    local daemon is the plain "ollama" backend's job, not the ladder's.
    """
    for p in load_registry():
        if not any(m.embed for m in p.models):
            continue
        if p.kind == "ollama":
            if p.api_base:
                return True
            continue
        if p.kind == "openai_compatible":
            _, api_key = resolve_credentials(p)
            if api_key:
                return True
    return False
