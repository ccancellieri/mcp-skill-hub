"""Resolve provider credentials from the user's opencode config, env, or an
inline config reference. Never reads or writes a secret to a tracked file.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .registry import Provider


def _config_home() -> Path:
    return Path(os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config"))


def _data_home() -> Path:
    return Path(os.environ.get("XDG_DATA_HOME") or (Path.home() / ".local" / "share"))


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def opencode_provider(provider_id: str) -> dict | None:
    """Return ``{baseURL, apiKey, models}`` for an opencode provider id.

    Looks in opencode's ``config.json`` / ``opencode.json`` ``provider.<id>``
    records (the ``options`` block carries ``baseURL``/``apiKey``), then falls
    back to ``auth.json`` for a bare api key.
    """
    if not provider_id:
        return None
    for fname in ("config.json", "opencode.json"):
        doc = _read_json(_config_home() / "opencode" / fname)
        prov = (doc.get("provider") or {}).get(provider_id)
        if isinstance(prov, dict):
            opts = prov.get("options") or {}
            return {
                "baseURL": opts.get("baseURL") or prov.get("baseURL") or "",
                "apiKey": opts.get("apiKey") or prov.get("apiKey") or "",
                "models": prov.get("models") or {},
            }
    auth = _read_json(_data_home() / "opencode" / "auth.json")
    entry = auth.get(provider_id)
    if isinstance(entry, dict) and entry.get("key"):
        return {"baseURL": "", "apiKey": entry["key"], "models": {}}
    if isinstance(entry, str):
        return {"baseURL": "", "apiKey": entry, "models": {}}
    return None


def resolve_credentials(provider: Provider) -> tuple[str | None, str | None]:
    """Return ``(api_base, api_key)`` for *provider*. Either may be None."""
    spec = provider.api_key or {}
    source = spec.get("source")
    ref = spec.get("ref") or ""
    if source == "inline":
        return (provider.api_base or None, ref or None)
    if source == "env":
        key = os.environ.get(ref) if ref else None
        base = os.environ.get(ref + "_BASE") if ref else None
        return (base or provider.api_base or None, key)
    if source == "opencode":
        rec = opencode_provider(ref or provider.name)
        if rec:
            return (rec.get("baseURL") or provider.api_base or None,
                    rec.get("apiKey") or None)
        return (provider.api_base or None, None)
    # No credentials needed (e.g. local ollama).
    return (provider.api_base or None, None)
