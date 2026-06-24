"""Generic, data-driven provider registry for the auxiliary LLM ladder.

Each provider is a plain config record so any OpenAI-compatible (or Anthropic,
or Ollama) backend can be added without code changes. Ordered by ``order``;
the escalation engine walks the list in that order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import config as _cfg

_VALID_LEVELS = {"L1", "L2", "L3", "L3_fallback", "personal"}
_VALID_KINDS = {"openai_compatible", "anthropic", "ollama"}


@dataclass
class ProviderModel:
    id: str
    complexity: str = "light"          # "light" | "heavy"
    monthly_cap_tokens: int | None = None
    tags: list[str] = field(default_factory=list)   # specializations, e.g. ["python","git"]


@dataclass
class Provider:
    name: str
    level: str
    kind: str
    api_base: str = ""
    api_key: dict = field(default_factory=dict)   # {source, ref}
    enabled: bool = True
    order: int = 100
    models: list[ProviderModel] = field(default_factory=list)


def _parse_model(raw: dict) -> ProviderModel | None:
    if not isinstance(raw, dict) or not raw.get("id"):
        return None
    cx = raw.get("complexity") or "light"
    if cx not in ("light", "heavy"):
        cx = "light"
    cap = raw.get("monthly_cap_tokens")
    cap = int(cap) if isinstance(cap, (int, float)) else None
    raw_tags = raw.get("tags") or []
    tags = [str(t).strip().lower() for t in raw_tags if isinstance(t, str) and t.strip()] \
        if isinstance(raw_tags, list) else []
    return ProviderModel(id=str(raw["id"]), complexity=cx, monthly_cap_tokens=cap, tags=tags)


def _parse_provider(raw: dict) -> Provider | None:
    if not isinstance(raw, dict):
        return None
    name = raw.get("name")
    level = raw.get("level")
    kind = raw.get("kind")
    if not name or level not in _VALID_LEVELS or kind not in _VALID_KINDS:
        return None
    models = [m for m in (_parse_model(x) for x in (raw.get("models") or [])) if m]
    return Provider(
        name=str(name),
        level=str(level),
        kind=str(kind),
        api_base=str(raw.get("api_base") or ""),
        api_key=dict(raw.get("api_key") or {}),
        enabled=bool(raw.get("enabled", True)),
        order=int(raw.get("order", 100)),
        models=models,
    )


def load_registry() -> list[Provider]:
    """Return enabled providers from config, sorted by ``order`` ascending."""
    raw = _cfg.get("llm_provider_registry") or []
    if not isinstance(raw, list):
        return []
    provs = [p for p in (_parse_provider(r) for r in raw) if p and p.enabled]
    return sorted(provs, key=lambda p: p.order)
