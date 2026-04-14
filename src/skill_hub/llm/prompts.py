"""Versioned prompt registry — load YAML prompt files from ``prompts/``.

Each YAML file has shape::

    name: rerank
    version: 1
    model_hint: tier_cheap   # optional — suggested tier
    template: |
      You are a relevance judge. Score 0.0-1.0 ...

Callers::

    prompt = load_prompt("rerank").format(query=q, name=n, description=d)

Missing files raise FileNotFoundError — prompts should be checked into git.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@dataclass
class Prompt:
    name: str
    version: int
    template: str
    model_hint: str | None = None
    metadata: dict[str, Any] | None = None

    def format(self, **kwargs: Any) -> str:
        return self.template.format(**kwargs)


def load_prompt(name: str) -> Prompt:
    import yaml
    path = _PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"prompt not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return Prompt(
        name=str(data.get("name") or name),
        version=int(data.get("version") or 1),
        template=str(data.get("template") or ""),
        model_hint=data.get("model_hint"),
        metadata={k: v for k, v in data.items()
                  if k not in {"name", "version", "template", "model_hint"}} or None,
    )
