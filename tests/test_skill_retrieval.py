"""Focused behavior checks for the explicit, broad skill shortlist."""
from __future__ import annotations

from skill_hub.context_service import build_context
from skill_hub.store import Skill, SkillStore


def _skill(store: SkillStore, name: str, description: str, *, content: str = "") -> None:
    store.upsert_skill(Skill(
        id=f"test:{name}", name=name, description=description,
        content=content or f"# {name}", file_path=f"/skills/{name}/SKILL.md",
        plugin="test",
    ))


def _sources(store: SkillStore, prompt: str, **kwargs) -> list[str]:
    from skill_hub.context_service import _skill_shortlist

    items = _skill_shortlist(store._conn, prompt, kwargs.get("max_skill_items", 20),
                             kwargs.get("include_full_text", False))
    return [item["source"] for item in items if item["kind"] == "skill"]


def test_name_match_does_not_veto_distinct_description_match(tmp_path):
    store = SkillStore(db_path=tmp_path / "skills.db")
    try:
        _skill(store, "slides", "Design clear slide presentations.")
        _skill(store, "data-work", "Create spreadsheet formulas and charts.")

        sources = _sources(store, "Create slides and spreadsheet charts")

        assert sources == ["skill:test:slides", "skill:test:data-work"]
        assert [item["source"] for item in build_context(
            "Create slides and spreadsheet charts", store=store
        )["items"] if item["kind"] == "skill"] == ["skill:test:slides"]
    finally:
        store.close()


def test_precise_description_match_survives_unrelated_constraints(tmp_path):
    store = SkillStore(db_path=tmp_path / "skills.db")
    try:
        _skill(store, "cluster-access", "Configure Kubernetes admission policies.")

        assert _sources(
            store, "Explain Kubernetes without changing production credentials"
        ) == ["skill:test:cluster-access"]
    finally:
        store.close()


def test_content_only_matches_do_not_enter_shortlist(tmp_path):
    store = SkillStore(db_path=tmp_path / "skills.db")
    try:
        _skill(store, "unrelated", "Prepare release announcements.",
               content="spreadsheet spreadsheet spreadsheet")
        _skill(store, "tables", "Build spreadsheet tables.")

        assert _sources(store, "spreadsheet") == ["skill:test:tables"]
    finally:
        store.close()


def test_italian_and_english_task_terms_match_without_substrings(tmp_path):
    store = SkillStore(db_path=tmp_path / "skills.db")
    try:
        _skill(store, "presentations", "Create presentation slides.")
        _skill(store, "other", "Document a present state.")

        assert _sources(store, "Crea una presentazione") == ["skill:test:presentations"]
        assert _sources(store, "presentation") == ["skill:test:presentations"]
    finally:
        store.close()


def test_generic_meta_prompt_abstains_but_explicit_skill_name_is_kept(tmp_path):
    store = SkillStore(db_path=tmp_path / "skills.db")
    try:
        _skill(store, "slides", "Create a presentation.")
        _skill(store, "unnamed", ">-")

        assert _sources(store, "Which skill should I use for this task?") == []
        assert _sources(store, "Use $slides") == ["skill:test:slides"]
        assert _sources(store, "Use $test:unnamed") == ["skill:test:unnamed"]
    finally:
        store.close()


def test_full_description_ranks_but_snippet_is_bounded_and_body_is_lazy(tmp_path):
    store = SkillStore(db_path=tmp_path / "skills.db")
    try:
        _skill(store, "long", "Introductory words. " * 100 + "Configure Kubernetes clusters.",
               content="BODY SECRET")

        statements: list[str] = []
        store._conn.set_trace_callback(statements.append)
        try:
            from skill_hub.context_service import _skill_shortlist
            items = _skill_shortlist(store._conn, "Kubernetes clusters", 20)
        finally:
            store._conn.set_trace_callback(None)
        skill = next(item for item in items if item["kind"] == "skill")
        assert skill["source"] == "skill:test:long"
        assert len(skill["text"]) <= 1200
        assert "_full_text" not in skill
        assert not any("select content from skills" in statement.lower() for statement in statements)

        full = _skill_shortlist(store._conn, "Kubernetes clusters", 20,
                                include_full_text=True)
        assert next(item for item in full if item["kind"] == "skill")["_full_text"] == "BODY SECRET"
    finally:
        store.close()


def test_shortlist_respects_cap_and_stable_id_ties(tmp_path):
    store = SkillStore(db_path=tmp_path / "skills.db")
    try:
        for name in reversed("abcdefghijklmnopqrstuvwxy"):
            _skill(store, name, "Guide spreadsheet work.")

        first = _sources(store, "spreadsheet", max_skill_items=5)
        second = _sources(store, "spreadsheet", max_skill_items=5)

        assert first == [f"skill:test:{name}" for name in "abcde"]
        assert second == first
        assert _sources(store, "spreadsheet", max_skill_items=0) == []
        assert _sources(store, "spreadsheet", max_skill_items=50) == [
            f"skill:test:{name}" for name in "abcdefghijklmnopqrst"
        ]
    finally:
        store.close()
