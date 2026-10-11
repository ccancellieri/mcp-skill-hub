"""Automatic skill suggestions require domain evidence, preserving invocation."""
import pytest

from skill_hub.context_service import build_context
from skill_hub.store import Skill, SkillStore


@pytest.fixture
def store(tmp_path):
    store = SkillStore(db_path=tmp_path / "fixture.db")
    for name, description in [
        ("testing-dags", "Test Airflow DAG scheduling and task dependencies."),
        ("screen-reader-testing", "Test website accessibility with a screen reader."),
        ("slides", "Create slide presentations and speaker notes."),
        ("spreadsheets", "Create spreadsheet formulas and charts."),
        ("api-architect", "Design API service contracts and endpoint schemas."),
        ("imagegen", "Generate or edit images and illustrations."),
        ("documents", "Create Word documents with tables."),
    ]:
        store.upsert_skill(Skill(id=f"fixture:{name}", name=name, description=description,
                                content="POISON BODY generic test prompt", plugin="fixture",
                                file_path=f"/fixture/{name}.md"))
    yield store
    store.close()


def selected(store, prompt):
    result = build_context(prompt, store=store, cfg={"memory_retrieval_backend": "raw"})
    assert result["original_prompt"] == prompt
    assert "POISON BODY" not in result["context"]
    return {item["source"] for item in result["items"]}


@pytest.mark.parametrize("prompt", [
    "testiamo e proseguiamo chiudendo issues", "Procedi con i test.",
    "Please test and verify.", "Continue debugging.",
    "What does API stand for?", "Cosa significa API?",
])
def test_generic_followups_and_definitions_do_not_infer_specialties(store, prompt):
    assert selected(store, prompt) == set()


@pytest.mark.parametrize("prompt,expected", [
    ("Test Airflow DAG scheduling", "testing-dags"),
    ("Test website accessibility", "screen-reader-testing"),
    ("Verifica accessibility con screen reader", "screen-reader-testing"),
    ("Create a slide presentation", "slides"),
    ("Crea una presentazione con diapositive", "slides"),
    ("Crea fogli con grafici", "spreadsheets"),
    ("Crea un documento Word", "documents"),
    ("Genera un'immagine", "imagegen"),
])
def test_specialized_subjects_and_bilingual_artifacts_still_match(store, prompt, expected):
    assert selected(store, prompt) == {f"skill:fixture:{expected}"}


@pytest.mark.parametrize("prompt", [
    "Use $testing-dags", "Usa $fixture:testing-dags",
    "What does fixture:testing-dags stand for?", "Cosa significa $testing-dags?",
])
def test_explicit_identifier_overrides_generic_and_definition_guards(store, prompt):
    assert "skill:fixture:testing-dags" in selected(store, prompt)


def test_technical_identifier_does_not_match_its_component_words(store):
    store.upsert_skill(Skill(id="fixture:cache_key", name="cache_key",
                            description="Inspect cache_key metadata.", content="fixture",
                            plugin="fixture", file_path="/fixture/cache_key.md"))
    assert selected(store, "Inspect cache_key") == {"skill:fixture:cache_key"}
    assert selected(store, "Inspect other_cache_key") == set()


@pytest.mark.parametrize("prompt", [
    "Debug acronym parser failures", "Test the acronym parser",
    "Fix acronym parser failures", "Repair the acronym parser",
])
def test_definition_vocabulary_does_not_suppress_operational_requests(store, prompt):
    store.upsert_skill(Skill(id="fixture:parser", name="parser",
                            description="Diagnose parser errors and tokenization failures.",
                            content="fixture", plugin="fixture", file_path="/fixture/parser.md"))
    assert selected(store, prompt) == {"skill:fixture:parser"}


@pytest.mark.parametrize("prompt", [
    "Generate an image showing an acronym", "Prepara un'immagine con una sigla",
    "Edit the image with acronym text", "Modifica l'immagine con una sigla",
])
def test_artifact_actions_with_definition_words_still_match(store, prompt):
    assert selected(store, prompt) == {"skill:fixture:imagegen"}


@pytest.mark.parametrize("name", ["needle", ""])
def test_explicit_name_or_legacy_leaf_is_not_crowded_out_by_description_hits(store, name):
    for number in range(110):
        store.upsert_skill(Skill(id=f"crowd:{number}", name=f"distractor-{number}",
                                description="Needle", content="fixture", plugin="crowd",
                                file_path=f"/fixture/{number}.md"))
    store.upsert_skill(Skill(id="fixture:needle", name=name,
                            description="Useful specific domain.", content="needle",
                            plugin="fixture", file_path="/fixture/needle.md"))
    assert selected(store, "Use $needle") == {"skill:fixture:needle"}
