"""Resolve the single raw-memory/wiki retrieval backend selection."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def selected_memory_backend(cfg: Any = None) -> str | None:
    """Return ``raw`` or ``wiki``; invalid explicit values fail closed as None.

    Missing keys choose the conservative ``raw`` source. The project config
    module supplies its default through ``get``; plain mappings are supported
    for tests and request-scoped callers.
    """
    if cfg is None:
        from . import config as cfg
    key = "memory_retrieval_backend"
    if isinstance(cfg, Mapping):
        value = cfg.get(key, "raw")
    else:
        raw_config = getattr(cfg, "load_config", None)
        if callable(raw_config):
            try:
                value = raw_config().get(key, "raw")
            except Exception:  # noqa: BLE001 - fail closed below
                value = None
        else:
            getter = getattr(cfg, "get", None)
            if not callable(getter):
                return None
            try:
                value = getter(key)
            except Exception:  # noqa: BLE001 - fail closed below
                value = None
            if value is None:
                value = "raw"
    return value if isinstance(value, str) and value in {"raw", "wiki"} else None
