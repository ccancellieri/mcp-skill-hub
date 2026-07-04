"""Regression test: `_dynamic_context_stage` must not crash when the local
LLM's skill-lifecycle JSON includes an explicit ``null`` for a field instead
of omitting it.

``dict.get(key, default)`` only falls back to ``default`` when ``key`` is
ABSENT — if the key is present with value ``None`` (which local models
regularly emit for "nothing changed"), ``.get`` returns ``None`` verbatim.
Before the fix, ``lifecycle.get("context_summary", context_summary)``
returned ``None`` in that case and the very next line's
``len(new_summary)`` raised ``TypeError: object of type 'NoneType' has no
len()`` — silently downgrading every such turn to the crude keyword-only
context fallback (no relevance floor, no agent-I/O guidance).
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_dynamic_context_stage_survives_null_lifecycle_fields(tmp_path, monkeypatch):
    import skill_hub.cli as cli
    from skill_hub.store import SkillStore

    monkeypatch.setenv("HOME", str(tmp_path))
    db = tmp_path / "skill_hub.db"
    store = SkillStore(db_path=db)
    store.save_session_context(
        session_id="sess-null", loaded_skills=[],
        context_summary="prior summary must survive", message_count=0,
        recent_messages=[],
    )
    store.close()

    # Bypass real vector/FTS search entirely — the point of this test is the
    # lifecycle-dict handling, not candidate retrieval — so stub `search` to
    # always surface one candidate regardless of the (dummy) embed vector.
    _candidate = {
        "id": "local:test-skill", "name": "test-skill", "target": "claude",
        "description": "a candidate skill for the dynamic context stage",
        "content": "skill body", "file_path": "", "plugin": "",
    }
    monkeypatch.setattr(SkillStore, "search", lambda self, *a, **k: [_candidate])

    monkeypatch.setattr(cli, "SkillStore", lambda *a, **k: SkillStore(db_path=db))
    monkeypatch.setattr(cli, "embed", lambda text: [0.1, 0.2, 0.3, 0.4])
    cli._complexity_centroids = None  # force recompute with the stubbed embed above
    monkeypatch.setattr(cli, "ollama_available", lambda *a, **k: True)
    monkeypatch.setattr(cli, "should_run_llm", lambda *a, **k: True)

    # The exact shape observed in production: `keep`/`add`/`drop` come back as
    # valid lists but `context_summary` is explicitly null ("nothing to add").
    monkeypatch.setattr(cli, "eval_skill_lifecycle", lambda **kw: {
        "keep": [], "add": [], "drop": [], "context_summary": None,
    })

    result = cli._dynamic_context_stage(
        "short message", "sess-null", {}, triage_hint=None,
    )

    # Must not raise, and must not silently drop the prior rolling summary.
    assert result is None or isinstance(result, str)
    store2 = SkillStore(db_path=db)
    ctx = store2.get_session_context("sess-null")
    store2.close()
    assert ctx["context_summary"] == "prior summary must survive"
