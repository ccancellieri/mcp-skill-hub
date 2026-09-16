"""Compatibility API for prompt enrichment, backed by scoped context retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class RewriterResult:
    prefix: str = ""
    body: str | None = None
    note: str = ""
    applied: bool = False


@dataclass
class ImproveResult:
    prompt: str
    original: str
    applied: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


_REGISTRY: dict[str, Callable] = {}
_BUILTINS = {"add_skill_context", "add_recent_tasks"}


def register(name: str, fn: Callable) -> None:
    _REGISTRY[name] = fn


def available() -> list[str]:
    return sorted(_BUILTINS | set(_REGISTRY))


def improve_prompt(
    prompt: str, store: Any, rewriters: list[str] | None = None,
    cfg: dict[str, Any] | None = None, *, cwd: str = "",
    session_id: str = "", task_id: int | None = None,
) -> ImproveResult:
    """Append sourced context; never paraphrase or replace the original text."""
    from ..context_service import build_context

    names = rewriters if rewriters is not None else sorted(_BUILTINS)
    if names == ["all"]:
        names = available()
    prefixes: list[str] = []
    applied: list[str] = []
    notes: list[str] = []
    if _BUILTINS.intersection(names):
        try:
            result = build_context(prompt, cwd=cwd, session_id=session_id,
                                   task_id=task_id, store=store, cfg=cfg)
            if result["context"]:
                prefixes.append(result["context"])
                applied.append("context")
            notes.extend(result["warnings"])
        except Exception as exc:
            notes.append(f"context: error: {exc}")
    for name in names:
        if name in _BUILTINS:
            continue
        if name == "normalize_language":
            notes.append("normalize_language: retired; original prompt preserved")
            continue
        fn = _REGISTRY.get(name)
        if fn is None:
            notes.append(f"{name}: unknown")
            continue
        try:
            result = fn(prompt, store, cfg or {})
            if result.body is not None:
                notes.append(f"{name}: body replacement ignored")
            if result.prefix:
                prefixes.append(result.prefix)
                applied.append(name)
            if result.note:
                notes.append(result.note)
        except Exception as exc:
            notes.append(f"{name}: error: {exc}")
    enriched = prompt if not prefixes else prompt + "\n\n" + "\n".join(prefixes)
    return ImproveResult(enriched, prompt, applied, notes)
