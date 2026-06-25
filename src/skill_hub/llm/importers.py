"""Import / sync auxiliary providers from external configs into the registry.

Supports three source formats — ``opencode`` (the user's opencode config),
``openai`` (a generic ``{baseURL, apiKey?, models}`` payload) and ``litellm``
(a ``model_list`` config). Each is normalised to provider records that carry
only model *ids*; merging then **preserves** the hand-tuned ``complexity`` /
``tags`` / caps already set on surviving model ids — source configs don't carry
those, so a naive overwrite would degrade routing.

Secrets never leak into a diff: previews carry a credential-source *label*
only, never the key value. Opencode-sourced providers keep their secret in the
opencode file (referenced, not copied); generic/litellm keys with an inline
secret are stored as ``{source: "inline"}`` in the local (untracked) config.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

SUPPORTED_FORMATS = ("opencode", "openai", "litellm")
_DEFAULT_LEVEL = "L3"


@dataclass
class NormalizedProvider:
    """A provider distilled from a source format — models as bare ids."""
    name: str
    kind: str
    api_base: str
    api_key: dict           # {source, ref}; {} means no credential needed
    model_ids: list[str]
    match_ref: str = ""     # opencode provider id, used to match existing records


# ── format parsers ──────────────────────────────────────────────────────────

def _kind_from_npm(npm: str) -> str:
    n = (npm or "").lower()
    if "anthropic" in n:
        return "anthropic"
    if "ollama" in n:
        return "ollama"
    return "openai_compatible"


def _models_to_ids(models: Any) -> list[str]:
    """Accept opencode's dict-keyed models, a list of ids, or a list of dicts."""
    if isinstance(models, dict):
        return [str(k) for k in models.keys()]
    if isinstance(models, list):
        out: list[str] = []
        for m in models:
            if isinstance(m, str) and m.strip():
                out.append(m.strip())
            elif isinstance(m, dict) and m.get("id"):
                out.append(str(m["id"]))
        return out
    return []


def _name_from_url(url: str, fallback: str) -> str:
    return urlparse(url).hostname or fallback


def normalize_opencode(payload: dict) -> list[NormalizedProvider]:
    providers = (payload.get("provider") or {}) if isinstance(payload, dict) else {}
    out: list[NormalizedProvider] = []
    for pid, prov in providers.items():
        if not isinstance(prov, dict):
            continue
        out.append(NormalizedProvider(
            name=str(prov.get("name") or pid),
            kind=_kind_from_npm(prov.get("npm") or ""),
            api_base="",  # resolved live from the opencode file at call time
            api_key={"source": "opencode", "ref": str(pid)},
            model_ids=_models_to_ids(prov.get("models")),
            match_ref=str(pid),
        ))
    return out


def normalize_openai(payload: dict) -> list[NormalizedProvider]:
    if not isinstance(payload, dict):
        return []
    base = str(payload.get("baseURL") or payload.get("api_base")
               or payload.get("base_url") or "")
    key = payload.get("apiKey") or payload.get("api_key")
    api_key = {"source": "inline", "ref": str(key)} if key else {}
    return [NormalizedProvider(
        name=str(payload.get("name") or _name_from_url(base, "openai-provider")),
        kind="openai_compatible",
        api_base=base,
        api_key=api_key,
        model_ids=_models_to_ids(payload.get("models")),
    )]


def normalize_litellm(payload: dict) -> list[NormalizedProvider]:
    if not isinstance(payload, dict):
        return []
    model_list = payload.get("model_list")
    if not isinstance(model_list, list):
        return []
    # One provider per distinct endpoint (api_base); ungrouped entries collapse
    # into a single "_default" provider with no base.
    groups: dict[str, dict] = {}
    order: list[str] = []
    for entry in model_list:
        if not isinstance(entry, dict):
            continue
        params = entry.get("litellm_params") or {}
        model = str(params.get("model") or entry.get("model_name") or "").strip()
        if not model:
            continue
        base = str(params.get("api_base") or "")
        gkey = base or "_default"
        if gkey not in groups:
            groups[gkey] = {"base": base, "key": params.get("api_key"), "ids": []}
            order.append(gkey)
        if model not in groups[gkey]["ids"]:
            groups[gkey]["ids"].append(model)
    out: list[NormalizedProvider] = []
    for gkey in order:
        g = groups[gkey]
        api_key = {"source": "inline", "ref": str(g["key"])} if g["key"] else {}
        out.append(NormalizedProvider(
            name=_name_from_url(g["base"], "litellm-provider"),
            kind="openai_compatible",
            api_base=g["base"],
            api_key=api_key,
            model_ids=g["ids"],
        ))
    return out


def normalize(fmt: str, payload: dict) -> list[NormalizedProvider]:
    if fmt == "opencode":
        return normalize_opencode(payload)
    if fmt == "openai":
        return normalize_openai(payload)
    if fmt == "litellm":
        return normalize_litellm(payload)
    raise ValueError(f"unsupported format: {fmt!r}")


def read_opencode_config() -> dict:
    """Read the user's opencode config (config.json then opencode.json)."""
    home = Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))
    for fname in ("config.json", "opencode.json"):
        try:
            doc = json.loads((home / "opencode" / fname).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(doc, dict) and doc.get("provider"):
            return doc
    return {}


# ── matching + merge + diff ───────────────────────────────────────────────────

def _cred_label(api_key: dict) -> str:
    src = (api_key or {}).get("source")
    ref = (api_key or {}).get("ref") or ""
    if src == "opencode":
        return f"opencode:{ref}" if ref else "opencode"
    if src == "env":
        return f"env:{ref}" if ref else "env"
    if src == "inline":
        return "inline"
    return "none"


def _find_match(current: list[dict], inc: NormalizedProvider) -> int | None:
    """Index of the existing record that *inc* should merge into, or None.

    Opencode providers match on the credential ref (stable even when the
    display name was customised, e.g. ref ``agent-platform`` -> ``work-gateway``).
    Others match on name, then on a non-empty api_base.
    """
    for i, rec in enumerate(current):
        if not isinstance(rec, dict):
            continue
        ak = rec.get("api_key") or {}
        if inc.match_ref:
            if ak.get("source") == "opencode" and str(ak.get("ref")) == inc.match_ref:
                return i
            continue
        if rec.get("name") and str(rec.get("name")).lower() == inc.name.lower():
            return i
        if inc.api_base and rec.get("api_base") == inc.api_base:
            return i
    return None


def _merge_one(existing: dict | None, inc: NormalizedProvider,
               *, order_hint: int) -> tuple[dict, dict]:
    inc_ids = list(dict.fromkeys(inc.model_ids))  # dedupe, preserve order

    if existing is None:
        models = [{"id": mid, "complexity": "light", "tags": []} for mid in inc_ids]
        rec = {
            "name": inc.name, "level": _DEFAULT_LEVEL, "kind": inc.kind,
            "api_base": inc.api_base, "api_key": dict(inc.api_key),
            "enabled": True, "order": order_hint, "models": models,
        }
        diff = {
            "provider": inc.name, "status": "new", "matched_name": None,
            "level": _DEFAULT_LEVEL, "kind": inc.kind, "api_base": inc.api_base,
            "cred_label": _cred_label(inc.api_key),
            "models_added": inc_ids, "models_removed": [], "models_kept": [],
        }
        return rec, diff

    # Update: preserve per-model tuning (complexity/tags/caps) for surviving ids.
    existing_models = {
        str(m.get("id")): m
        for m in (existing.get("models") or [])
        if isinstance(m, dict) and m.get("id")
    }
    new_models: list[dict] = []
    added: list[str] = []
    kept: list[str] = []
    for mid in inc_ids:
        if mid in existing_models:
            new_models.append(existing_models[mid])
            kept.append(mid)
        else:
            new_models.append({"id": mid, "complexity": "light", "tags": []})
            added.append(mid)
    removed = [mid for mid in existing_models if mid not in set(inc_ids)]

    rec = dict(existing)  # keep name/level/order/enabled and any extra keys
    rec["kind"] = existing.get("kind") or inc.kind
    # Opencode incoming carries api_base="" (resolved live) — keep existing then.
    rec["api_base"] = inc.api_base or existing.get("api_base", "")
    # Credential: take the incoming one, but never silently *downgrade* a
    # deliberate env/opencode credential to an inline secret on a re-sync. An
    # existing inline key may still be rotated; an empty incoming keeps existing.
    existing_cred = existing.get("api_key") or {}
    if inc.api_key and existing_cred.get("source") not in ("env", "opencode"):
        rec["api_key"] = dict(inc.api_key)
    else:
        rec["api_key"] = existing_cred
    rec["models"] = new_models
    diff = {
        "provider": existing.get("name") or inc.name, "status": "update",
        "matched_name": existing.get("name"),
        "level": existing.get("level"), "kind": rec["kind"],
        "api_base": rec["api_base"], "cred_label": _cred_label(rec["api_key"]),
        "models_added": added, "models_removed": removed, "models_kept": kept,
    }
    return rec, diff


def merge_registry(current: list, incoming: list[NormalizedProvider]) -> tuple[list[dict], list[dict]]:
    """Merge *incoming* into *current*, returning (new_registry, diffs)."""
    result: list[dict] = [dict(r) for r in (current or []) if isinstance(r, dict)]
    diffs: list[dict] = []
    max_order = max([int(r.get("order", 100)) for r in result] + [0])
    for inc in incoming:
        idx = _find_match(result, inc)
        if idx is None:
            max_order += 10
            rec, diff = _merge_one(None, inc, order_hint=max_order)
            result.append(rec)
        else:
            rec, diff = _merge_one(result[idx], inc, order_hint=0)
            result[idx] = rec
        diffs.append(diff)
    return result, diffs


def diff_registry(current: list, incoming: list[NormalizedProvider]) -> list[dict]:
    """Secret-free preview of what merging *incoming* would change."""
    return merge_registry(current, incoming)[1]
