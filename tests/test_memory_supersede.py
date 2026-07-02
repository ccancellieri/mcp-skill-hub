"""Outdate contradicted memory on new decisions (#136)."""
from __future__ import annotations

import pytest

from skill_hub import memory_supersede as ms
from skill_hub.store import SkillStore


@pytest.fixture()
def store(tmp_path):
    s = SkillStore(db_path=tmp_path / "skill_hub.db")
    yield s
    s.close()


def _write(tmp_path, name, body):
    p = tmp_path / name
    p.write_text(f"---\nname: {name}\ndescription: d\ntype: project\n---\n\n{body}\n",
                 encoding="utf-8")
    return str(p)


def test_extract_json_array_handles_noise():
    assert ms._extract_json_array('here: [{"id":"a","reason":"x"}] done') == [
        {"id": "a", "reason": "x"}
    ]
    assert ms._extract_json_array("no json") == []
    assert ms._extract_json_array('{"id":"a"}') == []   # object, not array


def test_stamp_frontmatter_adds_and_is_idempotent(tmp_path):
    doc = _write(tmp_path, "old.md", "The old decision was X.")
    assert ms._stamp_frontmatter(doc, "new.md") is True
    text = (tmp_path / "old.md").read_text()
    assert "superseded_by: new.md" in text
    assert text.count("superseded_by:") == 1
    assert "The old decision was X." in text          # body preserved

    # Re-running with a different superseder replaces, never duplicates.
    assert ms._stamp_frontmatter(doc, "newer.md") is True
    text2 = (tmp_path / "old.md").read_text()
    assert text2.count("superseded_by:") == 1
    assert "superseded_by: newer.md" in text2


def test_stamp_frontmatter_skips_files_without_frontmatter(tmp_path):
    p = tmp_path / "plain.md"
    p.write_text("just a body, no frontmatter\n", encoding="utf-8")
    assert ms._stamp_frontmatter(str(p), "new.md") is False


def test_run_supersede_disabled_is_noop(store, monkeypatch):
    monkeypatch.setattr(ms._cfg, "get", lambda k, d=None: False if k == "memory_supersede_enabled" else d)
    assert ms.run_supersede(store, new_doc_id="x", new_name="x",
                            new_text="anything", namespace="memory:user-project") == {
        "skipped": "disabled"
    }


def test_run_supersede_no_candidates_skips_judge(store, monkeypatch):
    monkeypatch.setattr(store, "search_vectors", lambda *a, **k: [])
    called = {"judge": False}
    monkeypatch.setattr(ms, "_judge", lambda *a, **k: called.__setitem__("judge", True) or [])
    out = ms.run_supersede(store, new_doc_id="x", new_name="x",
                           new_text="a new decision", namespace="memory:user-project")
    assert out == {"candidates": 0}
    assert called["judge"] is False


def test_run_supersede_marks_contradicted(store, tmp_path, monkeypatch):
    old = _write(tmp_path, "old.md", "We route memory writes to the local model only.")
    other = _write(tmp_path, "other.md", "Unrelated note about wiki indexing.")

    monkeypatch.setattr(store, "search_vectors", lambda *a, **k: [
        {"doc_id": old, "raw_score": 0.9},
        {"doc_id": other, "raw_score": 0.8},
    ])
    # Ladder judges only the old entry as superseded.
    monkeypatch.setattr(ms, "_judge", lambda new_text, cands: [
        {"id": old, "reason": "now routed via the ladder"}
    ])

    out = ms.run_supersede(store, new_doc_id=_write(tmp_path, "new.md", "Memory writes now go via the ladder."),
                           new_name="new.md", namespace="memory:user-project",
                           new_text="Memory writes now go via the ladder.")

    assert out["candidates"] == 2
    assert out["superseded"] == [old]
    assert "superseded_by: new.md" in (tmp_path / "old.md").read_text()
    assert "superseded_by:" not in (tmp_path / "other.md").read_text()

    rows = store._conn.execute(
        "SELECT action, doc_id, reason FROM memory_audit"
    ).fetchall()
    assert len(rows) == 1
    assert rows[0]["action"] == "supersede"
    assert rows[0]["doc_id"] == old
    assert "ladder" in rows[0]["reason"]


def test_find_candidates_excludes_self(store, monkeypatch):
    monkeypatch.setattr(store, "search_vectors", lambda *a, **k: [
        {"doc_id": "self.md", "raw_score": 1.0},
        {"doc_id": "other.md", "raw_score": 0.8},
    ])
    got = ms.find_candidates(store, "text", "memory:user-project",
                             top_k=6, min_sim=0.7, exclude_doc_id="self.md")
    assert [c["doc_id"] for c in got] == ["other.md"]
