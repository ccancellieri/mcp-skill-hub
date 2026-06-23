"""Generic, provider-agnostic model registry.

One place to resolve model identity, pricing, and availability so the rest of
the codebase stops hardcoding ``claude-*`` IDs and stale prices in scattered
dicts. Designed to stay in sync with the live Claude lineup automatically and
to work for any provider (Anthropic, Ollama, OpenAI, ...), not just Claude.

Three responsibilities:

1. **Resolve** a tier alias (``tier_cheap`` / ``tier_smart`` / ...) or a short
   family name (``haiku`` / ``sonnet`` / ``opus`` / ``fable``) to a full model
   ID, driven by ``config.llm_providers`` — the same map the litellm adapter
   already uses, so resolution never diverges from what actually runs.

2. **Price** any model. Pricing is derived from litellm's built-in
   ``model_cost`` table when available — that table covers hundreds of models
   across providers and is updated upstream, so prices stay in sync without us
   maintaining them. A small static table backstops models litellm does not
   know yet (e.g. Claude Fable 5) and supplies the short family aliases.

3. **Recognise** what is available right now: configured tiers, Ollama's live
   ``/api/tags`` list, and litellm's known catalogue. This is the
   provider-agnostic equivalent of asking Claude Code's ``/models`` what it can
   reach, without hardcoding a lineup.

Nothing here imports litellm at module load; the dependency is optional and
looked up lazily so import stays cheap and offline-safe.
"""
from __future__ import annotations

import re
from typing import Any

# Blended-rate weighting: assume 30% of tokens are input, 70% output. Matches
# the heuristic the dashboard cost report has always used.
_INPUT_WEIGHT = 0.30
_OUTPUT_WEIGHT = 0.70

# Known provider prefixes we strip to get a bare model id for matching.
_PROVIDER_PREFIXES = ("anthropic/", "vertex_ai/", "bedrock/", "openai/", "ollama/")

# Static fallback prices in USD per 1M tokens (input, output). Only consulted
# when litellm's model_cost has no entry — chiefly for models too new for
# litellm and for the short family aliases the router logs by name. Keep the
# Claude lineup here current as the backstop.
_STATIC_USD_PER_M: dict[str, tuple[float, float]] = {
    # Short family aliases (router verdicts log these bare names). Values mirror
    # litellm's current Claude pricing as a backstop when litellm is absent.
    "fable":  (10.0, 50.0),
    "haiku":  (1.0, 5.0),
    "sonnet": (3.0, 15.0),
    "opus":   (5.0, 25.0),
    # Full Claude IDs — backstop in case litellm lags the lineup.
    "claude-fable-5":   (10.0, 50.0),
    "claude-haiku-4-5": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-8":  (5.0, 25.0),
}

# Map a bare short family alias to the tier that serves it, for callers that
# want "the current sonnet" rather than a pinned ID. Fable is intentionally
# absent: it is not wired to a tier (and is not a cheap model), so it is only
# a known/priced model, not a tier alias.
_FAMILY_TO_TIER = {
    "haiku": "tier_mid",
    "sonnet": "tier_smart",
    "opus": "tier_planner",
}

_DATE_SUFFIX = re.compile(r"[-@]\d{8}.*$|@default$|:\w[\w.\-]*$")


def _cfg() -> Any:
    from . import config as _c
    return _c


def bare_id(model: str) -> str:
    """Strip provider prefix and date/version suffix → a comparable bare id.

    ``anthropic/claude-opus-4-8@default`` → ``claude-opus-4-8``;
    ``ollama/qwen2.5-coder:3b`` keeps its tag-free stem ``qwen2.5-coder``.
    """
    m = (model or "").strip()
    for pfx in _PROVIDER_PREFIXES:
        if m.startswith(pfx):
            m = m[len(pfx):]
            break
    m = _DATE_SUFFIX.sub("", m)
    return m


def _litellm_cost(model: str) -> tuple[float, float] | None:
    """Return (input_per_m, output_per_m) from litellm.model_cost, or None."""
    try:
        import litellm
    except Exception:
        return None
    table = getattr(litellm, "model_cost", None)
    if not table:
        return None
    bare = bare_id(model)
    # Try the id as given, then a few provider-qualified variants, then bare.
    candidates = [model, bare, f"anthropic/{bare}", f"vertex_ai/{bare}", f"openai/{bare}"]
    for key in candidates:
        entry = table.get(key)
        if entry:
            inp = entry.get("input_cost_per_token")
            out = entry.get("output_cost_per_token")
            if inp is not None and out is not None:
                return inp * 1_000_000, out * 1_000_000
    return None


def price_per_m(model: str) -> dict[str, float] | None:
    """USD per 1M tokens for *model*: ``{input, output, blended}`` or None.

    Prefers litellm's live table; falls back to the static backstop (which also
    covers short family aliases like ``haiku``). Returns None for unknown models
    (e.g. local Ollama models, which are free) so callers can skip them.
    """
    rates = _litellm_cost(model)
    if rates is None:
        bare = bare_id(model)
        rates = _STATIC_USD_PER_M.get(model) or _STATIC_USD_PER_M.get(bare)
    if rates is None:
        return None
    inp, out = rates
    return {
        "input": inp,
        "output": out,
        "blended": round(inp * _INPUT_WEIGHT + out * _OUTPUT_WEIGHT, 4),
    }


def blended_usd_per_m(model: str) -> float | None:
    """Blended (30/70) USD per 1M tokens for *model*, or None if unknown/free."""
    p = price_per_m(model)
    return p["blended"] if p else None


def resolve_tier(tier_or_alias: str) -> str:
    """Resolve a tier alias or short family name to a full model ID.

    ``tier_smart`` → the configured ``anthropic/claude-sonnet-4-6``;
    ``sonnet`` → resolved via its tier. Unknown input is returned unchanged so
    a caller may also pass a full ID through.
    """
    providers: dict = dict(_cfg().get("llm_providers") or {})
    key = (tier_or_alias or "").strip()
    if key in providers:
        return providers[key]
    if key in _FAMILY_TO_TIER and _FAMILY_TO_TIER[key] in providers:
        return providers[_FAMILY_TO_TIER[key]]
    default_tier = _cfg().get("llm_default_tier") or "tier_cheap"
    return providers.get(key, providers.get(default_tier, key))


def provider_of(model: str) -> str:
    """Best-effort provider label for a (possibly prefixed) model ID."""
    m = (model or "").strip()
    for pfx in _PROVIDER_PREFIXES:
        if m.startswith(pfx):
            return pfx.rstrip("/")
    return "anthropic" if "claude" in m else "unknown"


def active_lineup() -> list[dict[str, Any]]:
    """The configured tiers resolved to model IDs + provider + blended price.

    This is the "in sync with the current lineup" view: edit
    ``config.llm_providers`` (one place) and this reflects it, with prices
    pulled live from litellm.
    """
    providers: dict = dict(_cfg().get("llm_providers") or {})
    out: list[dict[str, Any]] = []
    for tier, model in providers.items():
        out.append({
            "tier": tier,
            "model": model,
            "provider": provider_of(model),
            "blended_usd_per_m": blended_usd_per_m(model),
        })
    return out


def latest_in_family(family: str) -> str | None:
    """Highest ``claude-<family>-<major>-<minor>`` id known, or None.

    Scans litellm's catalogue (plus the static backstop) for the given family
    and returns the newest version — e.g. ``opus`` → ``claude-opus-4-8`` once
    litellm lists 4-8, so the lineup follows upstream without code edits.
    """
    pat = re.compile(rf"^claude-{re.escape(family)}-(\d+)-(\d+)$")
    candidates: set[str] = set(_STATIC_USD_PER_M)
    try:
        import litellm

        for k in (getattr(litellm, "model_cost", None) or {}):
            candidates.add(bare_id(k))
    except Exception:
        pass
    best: tuple[tuple[int, int], str] | None = None
    for cid in candidates:
        m = pat.match(cid)
        if m:
            ver = (int(m.group(1)), int(m.group(2)))
            if best is None or ver > best[0]:
                best = (ver, cid)
    return best[1] if best else None


def sync_lineup(dry_run: bool = False) -> dict[str, Any]:
    """Upgrade configured Claude tiers to the latest known version per family.

    This is "keep in sync with the current lineup": each tier pinned to a
    ``claude-<family>-*`` id is bumped to ``latest_in_family``. Non-Claude
    tiers (Ollama, embeddings) are left untouched. Persists the new
    ``llm_providers`` map unless *dry_run*. Returns the list of changes.
    """
    providers: dict = dict(_cfg().get("llm_providers") or {})
    fam_pat = re.compile(r"^claude-(opus|sonnet|haiku)-")
    changes: list[dict[str, str]] = []
    updated = dict(providers)
    for tier, model in providers.items():
        if "claude" not in (model or ""):
            continue
        prefix = ""
        bare = model
        for pfx in _PROVIDER_PREFIXES:
            if model.startswith(pfx):
                prefix, bare = pfx, model[len(pfx):]
                break
        fm = fam_pat.match(bare)
        if not fm:
            continue
        latest = latest_in_family(fm.group(1))
        if latest and bare_id(bare) != latest:
            new_id = f"{prefix}{latest}"
            changes.append({"tier": tier, "from": model, "to": new_id})
            updated[tier] = new_id
    applied = bool(changes) and not dry_run
    if applied:
        _cfg().set("llm_providers", updated)
    return {"changes": changes, "applied": applied, "dry_run": dry_run}


def _ollama_models() -> list[str]:
    """Live list of installed Ollama models via /api/tags (empty on failure)."""
    try:
        import json
        import urllib.request

        base = str(_cfg().get("ollama_base") or "http://localhost:11434").rstrip("/")
        with urllib.request.urlopen(f"{base}/api/tags", timeout=2.0) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return sorted(m.get("name", "") for m in data.get("models", []) if m.get("name"))
    except Exception:
        return []


def available_models() -> dict[str, list[str]]:
    """Models recognised as reachable right now, provider-agnostic.

    - ``configured``: the IDs wired into ``llm_providers`` tiers.
    - ``ollama``: live from the Ollama daemon (``/api/tags``).
    - ``known``: litellm's catalogue restricted to Anthropic Claude entries
      (the lineup we keep in sync); empty if litellm is unavailable.
    """
    providers: dict = dict(_cfg().get("llm_providers") or {})
    known: list[str] = []
    try:
        import litellm

        table = getattr(litellm, "model_cost", None) or {}
        known = sorted({
            bare_id(k) for k in table
            if "claude" in k and k.startswith(("anthropic/", "vertex_ai/"))
        })
    except Exception:
        known = []
    return {
        "configured": sorted(set(providers.values())),
        "ollama": _ollama_models(),
        "known": known,
    }
