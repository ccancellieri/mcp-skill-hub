"""Tests for memexp.intelligent_merge — uses an injected fake LLM provider."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from memexp import intelligent_merge


class FakeProvider:
    def __init__(self, response: str):
        self.response = response
        self.calls: list[tuple[str, dict]] = []

    def complete(self, prompt: str, **kwargs):
        self.calls.append((prompt, kwargs))
        return self.response


@pytest.fixture(autouse=True)
def _restore_factory():
    yield
    intelligent_merge.set_provider_factory(None)


def _use_provider(p):
    intelligent_merge.set_provider_factory(lambda: p)


def test_merge_row_returns_short_circuit_when_identical(tmp_path):
    p = FakeProvider('{"id": "x"}')
    _use_provider(p)
    out = intelligent_merge.merge_row(
        "skills",
        {"id": "x"},
        {"id": "x", "name": "A"},
        {"id": "x", "name": "A"},
        cache_path=tmp_path / "cache.sqlite",
    )
    assert out == {"id": "x", "name": "A"}
    assert p.calls == []  # identical rows skip the LLM


def test_merge_row_calls_llm_and_writes_cache(tmp_path):
    p = FakeProvider('{"id": "x", "name": "MERGED", "content": "merged-body"}')
    _use_provider(p)
    cache = tmp_path / "cache.sqlite"

    out = intelligent_merge.merge_row(
        "skills",
        {"id": "x"},
        {"id": "x", "name": "LOCAL", "content": "local-body"},
        {"id": "x", "name": "INCOMING", "content": "incoming-body"},
        cache_path=cache,
    )
    assert out == {"id": "x", "name": "MERGED", "content": "merged-body"}
    assert len(p.calls) == 1
    assert "skills" in p.calls[0][0]
    # cache row written
    assert cache.exists()
    conn = sqlite3.connect(cache)
    n = conn.execute("SELECT COUNT(*) FROM merge_cache").fetchone()[0]
    conn.close()
    assert n == 1

    # second call hits cache, no new LLM call
    out2 = intelligent_merge.merge_row(
        "skills",
        {"id": "x"},
        {"id": "x", "name": "LOCAL", "content": "local-body"},
        {"id": "x", "name": "INCOMING", "content": "incoming-body"},
        cache_path=cache,
    )
    assert out2 == out
    assert len(p.calls) == 1


def test_merge_row_pk_is_sacred(tmp_path):
    p = FakeProvider('{"id": "tampered", "name": "X", "content": "Y"}')
    _use_provider(p)
    out = intelligent_merge.merge_row(
        "skills",
        {"id": "original"},
        {"id": "original", "name": "L", "content": "l"},
        {"id": "original", "name": "I", "content": "i"},
        cache_path=tmp_path / "c.sqlite",
    )
    assert out["id"] == "original"  # PK never overridden by LLM


def test_merge_row_falls_back_on_invalid_json(tmp_path):
    p = FakeProvider("not json at all")
    _use_provider(p)
    out = intelligent_merge.merge_row(
        "skills",
        {"id": "x"},
        {"id": "x", "name": "L"},
        {"id": "x", "name": "I"},
        cache_path=tmp_path / "c.sqlite",
    )
    assert out == {"id": "x", "name": "I"}  # falls back to incoming


def test_merge_row_strips_json_fences(tmp_path):
    p = FakeProvider('```json\n{"id": "x", "name": "M"}\n```')
    _use_provider(p)
    out = intelligent_merge.merge_row(
        "skills",
        {"id": "x"},
        {"id": "x", "name": "L"},
        {"id": "x", "name": "I"},
        cache_path=tmp_path / "c.sqlite",
    )
    assert out == {"id": "x", "name": "M"}


def test_merge_markdown_caches_and_returns_text(tmp_path):
    p = FakeProvider("merged body\nwith two lines\n")
    _use_provider(p)
    cache = tmp_path / "c.sqlite"

    out = intelligent_merge.merge_markdown(
        "n.md", "local body\n", "incoming body\n", cache_path=cache,
    )
    assert "merged body" in out
    # second call hits cache
    intelligent_merge.merge_markdown(
        "n.md", "local body\n", "incoming body\n", cache_path=cache,
    )
    assert len(p.calls) == 1


def test_merge_markdown_falls_back_when_llm_raises(tmp_path):
    class BoomProvider:
        def complete(self, prompt, **kw):
            raise RuntimeError("network down")
    _use_provider(BoomProvider())
    out = intelligent_merge.merge_markdown(
        "n.md", "local\n", "incoming\n", cache_path=tmp_path / "c.sqlite",
    )
    assert out == "incoming\n"  # falls back to incoming on failure
