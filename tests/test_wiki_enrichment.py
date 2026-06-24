"""Tests for LLM-wiki snippets in per-prompt context enrichment (G3).

``_wiki_context_snippets`` makes the migrated wiki contribute to the auto-injected
context (it was previously absent from `_dynamic_context_stage` and the keyword
fallback). Reuses the hybrid wiki index and degrades to [] on any miss/error.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub import cli  # noqa: E402


def test_wiki_snippets_formats_results(tmp_path, monkeypatch):
    from skill_hub import wiki as _wiki
    monkeypatch.setattr(_wiki, "query", lambda *a, **k: {"results": [
        {"slug": "olap-dims", "title": "OLAP Dimensions", "body": "Paginated cube dims via STAC."},
        {"slug": "dekadal", "title": "Dekadal Calendar", "body": "10-day periods, D3 lengths."},
    ]})
    cfg = {"wiki_preload_enabled": True, "wiki_enabled": True,
           "wiki_root": str(tmp_path), "wiki_private_scopes": {}}
    snips = cli._wiki_context_snippets(object(), cfg, "how do olap dims paginate", top_k=2)
    assert len(snips) == 2
    assert snips[0].startswith("Wiki [[olap-dims]] OLAP Dimensions:")
    assert "Paginated cube dims" in snips[0]


def test_wiki_snippets_disabled_returns_empty(tmp_path, monkeypatch):
    from skill_hub import wiki as _wiki
    monkeypatch.setattr(_wiki, "query", lambda *a, **k: {"results": [{"slug": "x", "title": "X", "body": "y"}]})
    cfg = {"wiki_preload_enabled": False, "wiki_enabled": True, "wiki_root": str(tmp_path)}
    assert cli._wiki_context_snippets(object(), cfg, "q") == []


def test_wiki_snippets_missing_root_returns_empty(tmp_path):
    cfg = {"wiki_preload_enabled": True, "wiki_enabled": True,
           "wiki_root": str(tmp_path / "nope")}
    assert cli._wiki_context_snippets(object(), cfg, "q") == []


class _ModuleCfg:
    """Mimics the config *module* passed in production: one-arg ``get`` only.

    Regression guard — the helper must not call ``get(key, default)`` (two
    args), which the real config module rejects with ``TypeError`` and would
    silently disable wiki injection on every live hook.
    """

    def __init__(self, values):
        self._v = values

    def get(self, key):  # noqa: A003 - intentionally one positional arg
        return self._v.get(key)


def test_wiki_snippets_works_with_one_arg_config_module(tmp_path, monkeypatch):
    from skill_hub import wiki as _wiki
    monkeypatch.setattr(_wiki, "query", lambda *a, **k: {"results": [
        {"slug": "olap-dims", "title": "OLAP Dimensions", "body": "Paginated cube dims."},
    ]})
    cfg = _ModuleCfg({"wiki_preload_enabled": True, "wiki_enabled": True,
                      "wiki_root": str(tmp_path), "wiki_private_scopes": {}})
    snips = cli._wiki_context_snippets(object(), cfg, "olap dims", top_k=1)
    assert len(snips) == 1
    assert snips[0].startswith("Wiki [[olap-dims]] OLAP Dimensions:")


def test_wiki_snippets_swallows_query_error(tmp_path, monkeypatch):
    from skill_hub import wiki as _wiki

    def _boom(*a, **k):
        raise RuntimeError("embed backend down")

    monkeypatch.setattr(_wiki, "query", _boom)
    cfg = {"wiki_preload_enabled": True, "wiki_enabled": True, "wiki_root": str(tmp_path)}
    assert cli._wiki_context_snippets(object(), cfg, "q") == []
