"""The interactive composer preserves input and uses server-owned source references."""
from fastapi.testclient import TestClient


def client():
    from skill_hub.services import registry
    from skill_hub.webapp.main import create_app
    registry.set_registry(registry.ServiceRegistry([]))
    return TestClient(create_app(store="fixture"))


def test_prepare_passes_explicit_projects_and_escapes_candidates(monkeypatch):
    from skill_hub.webapp.routes import context
    received = {}
    def prepare(prompt, **kwargs):
        received.update(prompt=prompt, **kwargs)
        return {"draft_id": "draft", "original_prompt": prompt, "mode": "manual",
                "token_budget": 900, "selected_ids": ["candidate"], "warnings": [],
                "needs_review": True, "selector_version": None,
                "candidates": [{"candidate_id": "candidate", "title": "<script>bad</script>",
                    "source": "memory:local", "kind": "memory", "text": "Keep 42 unchanged.",
                    "project_root": "/projects/a", "estimated_tokens": 5, "reason": "Relevant",
                    "updated_at": None}]}
    monkeypatch.setattr(context, "prepare_composition", prepare)
    response = client().post("/context/prepare", data={"prompt": "  Original\n", "projects": "/projects/a\n/projects/b", "token_budget": "900", "mode": "training"})
    assert response.status_code == 200
    assert received["prompt"] == "  Original\n"
    assert received["project_roots"] == ["/projects/a", "/projects/b"]
    assert "&lt;script&gt;bad&lt;/script&gt;" in response.text
    assert "<script>bad</script>" not in response.text
    assert 'value="candidate"' in response.text
    assert received["mode"] == "manual"
    assert "Review this selection" in response.text


def test_known_projects_are_unchecked_and_only_submitted_roots_are_authorized(monkeypatch):
    from skill_hub.webapp.routes import context

    known = ["/projects/alpha", "/projects/beta"]
    monkeypatch.setattr(context, "_known_project_roots", lambda store: known)
    page = client().get("/context")
    assert page.status_code == 200
    assert 'value="/projects/alpha"' in page.text
    assert 'value="/projects/beta"' in page.text
    assert 'value="/projects/alpha" checked' not in page.text
    assert 'value="/projects/beta" checked' not in page.text

    received = {}

    def prepare(prompt, **kwargs):
        received.update(prompt=prompt, **kwargs)
        return {
            "draft_id": "draft", "original_prompt": prompt, "mode": "manual",
            "token_budget": 900, "selected_ids": [], "warnings": [],
            "needs_review": True, "selector_version": None, "candidates": [],
        }

    monkeypatch.setattr(context, "prepare_composition", prepare)
    response = client().post("/context/prepare", data={
        "prompt": "work", "selected_projects": "/projects/alpha",
        "projects": "/custom/project\n/projects/alpha", "token_budget": "900",
    })
    assert response.status_code == 200
    assert received["project_roots"] == ["/projects/alpha", "/custom/project"]
    assert "/projects/beta" not in received["project_roots"]


def test_compose_sends_only_source_ids_and_explicit_excerpts(monkeypatch):
    from skill_hub.webapp.routes import context
    received = {}
    def compose(draft_id, **kwargs):
        received.update(draft_id=draft_id, **kwargs)
        return {"composition_id": "composition", "original_prompt": "original", "context": "evidence", "estimated_tokens": 3,
                "items": [], "warnings": [], "mode": "manual"}
    monkeypatch.setattr(context, "compose_context", compose)
    response = client().post("/context/compose", data={"draft_id": "draft", "selected_ids": ["a", "b"],
        "rejected_ids": "c", "excerpt:a": "short source", "confirmed": "true", "context": "forged evidence"})
    assert response.status_code == 200
    assert received["selected_ids"] == ["a", "b"]
    assert received["rejected_ids"] == []
    assert received["excerpts"] == {"a": "short source"}
    assert received["confirmed"] is False
    assert "context" not in received
    assert "Copy context" in response.text


def test_unchecked_candidate_excerpt_does_not_block_preview(monkeypatch):
    from skill_hub.webapp.routes import context
    received = {}

    def compose(draft_id, **kwargs):
        received.update(draft_id=draft_id, **kwargs)
        assert set(kwargs["excerpts"]) <= set(kwargs["selected_ids"])
        return {"composition_id": "composition", "original_prompt": "original",
                "context": "selected evidence", "estimated_tokens": 5,
                "items": [], "warnings": [], "mode": "manual"}

    monkeypatch.setattr(context, "compose_context", compose)
    response = client().post("/context/compose", data={
        "draft_id": "draft", "selected_ids": "b",
        "excerpt:a": "left in an unchecked textarea",
        "excerpt:b": "selected passage",
    })
    assert response.status_code == 200
    assert received["excerpts"] == {"b": "selected passage"}


def test_invalid_budget_and_stale_sources_have_visible_errors(monkeypatch):
    from skill_hub.webapp.routes import context
    monkeypatch.setattr(context, "prepare_composition", lambda *a, **k: {})
    def stale(*args, **kwargs):
        raise ValueError("Source changed; prepare the context again.")
    monkeypatch.setattr(context, "compose_context", stale)
    c = client()
    assert c.post("/context/prepare", data={"prompt": "work", "token_budget": "no"}).status_code == 422
    response = c.post("/context/compose", data={"draft_id": "old"})
    assert response.status_code == 422
    assert "Source changed" in response.text


def test_prompt_optimization_is_explicit_and_separate(monkeypatch):
    from skill_hub.webapp.routes import context
    monkeypatch.setattr(context, "optimize_prompt", lambda prompt: {
        "original_prompt": prompt, "optimized_prompt": prompt, "diff": "", "before_tokens": 3,
        "after_tokens": 3, "transformations": []})
    response = client().post("/context/optimize", data={"prompt": "Never change 42"})
    assert response.status_code == 200
    assert "Never change 42" in response.text
    assert "Original prompt" in response.text
    assert "Proposed prompt" in response.text


def test_prepare_never_composes_automatically_or_exposes_learning_controls(monkeypatch):
    from skill_hub.webapp.routes import context
    monkeypatch.setattr(context, "prepare_composition", lambda *a, **k: {
        "draft_id": "auto", "mode": "automatic", "needs_review": False,
        "selected_ids": [], "candidates": [], "warnings": [], "token_budget": 900})
    calls = []
    def compose(*args, **kwargs):
        calls.append(kwargs)
        return {"composition_id": "result", "original_prompt": "work", "context": "", "estimated_tokens": 0,
                "warnings": [], "confirmed": False}
    monkeypatch.setattr(context, "compose_context", compose)
    response = client().post("/context/prepare", data={"prompt": "work", "mode": "automatic"})
    assert response.status_code == 200
    assert calls == []
    assert "Preview selection" in response.text

    page = client().get("/context")
    assert page.status_code == 200
    assert 'name="mode"' not in page.text
    assert "Confirm selection and teach" not in page.text
    assert "Learning and model versions" not in page.text
    assert "/context/learning" not in page.text


def test_learning_actions_are_not_exposed_as_web_routes():
    c = client()
    assert c.post("/context/learning/train").status_code == 404
    assert c.post("/context/learning/promote").status_code == 404
    assert c.post("/context/outcome").status_code == 404
