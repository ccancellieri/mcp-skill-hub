"""Ladder-synthesized context digests (#135): cache, queueing, refresh, dedupe."""
from __future__ import annotations

import pytest

from skill_hub.compression import dedupe_snippets
from skill_hub.compression import digest as dg
from skill_hub.store import SkillStore


@pytest.fixture()
def store(tmp_path):
    s = SkillStore(db_path=tmp_path / "skill_hub.db")
    yield s
    s.close()


LONG_DOC = ("Decision: the escalation ladder rotates on quota errors. " * 30).strip()


def _pending_rows(store):
    return store._conn.execute(
        "SELECT key, digest, content FROM context_digests"
    ).fetchall()


def test_short_content_is_squeezed_not_queued(store):
    text, is_digest = dg.digest_or_squeezed(store, "wiki:short", "tiny   doc")
    assert is_digest is False
    assert text == "tiny doc"
    assert not _pending_rows(store)


def test_miss_returns_squeezed_and_queues(store):
    text, is_digest = dg.digest_or_squeezed(store, "wiki:long", LONG_DOC)
    assert is_digest is False
    assert "escalation ladder" in text
    rows = _pending_rows(store)
    assert len(rows) == 1
    assert rows[0]["key"] == "wiki:long"
    assert rows[0]["digest"] == ""
    assert rows[0]["content"] == LONG_DOC


def test_refresh_pending_builds_then_hit(store, monkeypatch):
    dg.digest_or_squeezed(store, "wiki:long", LONG_DOC)
    monkeypatch.setattr(dg, "build_digest", lambda text: "the condensed digest")

    assert dg.refresh_pending(store) == 1
    rows = _pending_rows(store)
    assert rows[0]["digest"] == "the condensed digest"
    assert rows[0]["content"] == ""          # raw source cleared after build

    text, is_digest = dg.digest_or_squeezed(store, "wiki:long", LONG_DOC)
    assert is_digest is True
    assert text == "the condensed digest"


def test_changed_content_invalidates_digest(store, monkeypatch):
    dg.digest_or_squeezed(store, "wiki:long", LONG_DOC)
    monkeypatch.setattr(dg, "build_digest", lambda text: "digest v1")
    dg.refresh_pending(store)

    changed = LONG_DOC + "\nNew decision appended."
    text, is_digest = dg.digest_or_squeezed(store, "wiki:long", changed)
    assert is_digest is False                # stale hash → raw injected
    rows = _pending_rows(store)
    assert rows[0]["digest"] == ""           # re-queued for the new content
    assert rows[0]["content"] == changed


def test_refresh_pending_skips_failed_builds(store, monkeypatch):
    dg.digest_or_squeezed(store, "wiki:long", LONG_DOC)
    monkeypatch.setattr(dg, "build_digest", lambda text: "")
    assert dg.refresh_pending(store) == 0
    assert _pending_rows(store)[0]["digest"] == ""   # still pending


def test_build_digest_rejects_non_compressing_output(monkeypatch):
    import importlib
    req = importlib.import_module("skill_hub.llm.request")
    monkeypatch.setattr(req, "request", lambda *a, **k: "x" * 20_000)
    assert dg.build_digest("short source") == ""
    monkeypatch.setattr(req, "request", lambda *a, **k: "a real digest")
    assert dg.build_digest(LONG_DOC) == "a real digest"


def test_build_digest_routes_op_context_digest(monkeypatch):
    import importlib
    req = importlib.import_module("skill_hub.llm.request")
    captured: dict = {}

    def fake_request(tier, prompt, **kw):
        captured.update(kw, tier=tier)
        return "digest"

    monkeypatch.setattr(req, "request", fake_request)
    dg.build_digest(LONG_DOC)
    assert captured["op"] == "context_digest"
    assert captured["tier"] == "cheap"


def test_dedupe_snippets_drops_repeated_long_lines():
    dup = "The escalation ladder rotates to the next provider on a quota error."
    parts = [
        f"--- Skill [a] ---\n{dup}\nSkill-only detail line goes right here ok.",
        f"Wiki [[ladder]] escalation\n{dup}",
        "Memory [x]: unrelated fact",
    ]
    out = dedupe_snippets(parts)
    joined = "\n\n".join(out)
    assert joined.count("quota error") == 1
    assert "Skill-only detail" in joined
    assert "unrelated fact" in joined
    # The wiki part reduced to its (short) prefix line survives; short lines
    # are never deduped.
    assert any(p.startswith("Wiki [[ladder]]") for p in out)


def test_dedupe_snippets_removes_emptied_parts():
    dup = "One single very long duplicated sentence used as the whole snippet."
    out = dedupe_snippets([dup, dup])
    assert out == [dup]
