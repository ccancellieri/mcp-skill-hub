"""Offline selection evaluation must preserve scope and count failures honestly."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


def bench():
    path = Path(__file__).resolve().parents[1] / "benchmarks/local_selector.py"
    assert path.is_file(), "local selector benchmark is missing"
    spec = importlib.util.spec_from_file_location("local_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scores_allow_many_or_no_skills_and_reject_unknown_ids():
    module = bench()
    candidates = [{"source": "skill:a"}, {"source": "skill:b"}]
    assert module.select_ids({"skill:a": .8, "skill:b": .9}, candidates, .7) == ["skill:b", "skill:a"]
    assert module.select_ids({"skill:a": .2, "skill:b": .1}, candidates, .7) == []
    assert module.select_ids({"skill:a": None, "skill:b": .9}, candidates, .7) == ["skill:b"]
    with pytest.raises(ValueError):
        module.select_ids({"skill:foreign": .9}, candidates, .7)
    for score in (float("nan"), 1.1, -.1):
        with pytest.raises(ValueError):
            module.select_ids({"skill:a": score, "skill:b": .4}, candidates, .7)


def test_rizzo_preserves_abstentions_and_rejects_missing_answers():
    module = bench()
    payload = {"prompt": "p", "evidence": [], "candidates": [
        {"id": "skill:a", "name": "a", "description": "A"},
        {"id": "skill:b", "name": "b", "description": "B"},
    ]}

    class Backend:
        def decide(self, request):
            return {"answers": {
                "skill:a": {"status": "ok", "probabilities": {"true": .8}, "input_tokens": 3},
                "skill:b": {"status": "uncertain", "probabilities": {}, "input_tokens": 4},
            }}

    scores, tokens, statuses = module.score_backend("rizzo", Backend(), payload)
    assert scores == {"skill:a": .8, "skill:b": None}
    assert tokens == 7
    assert statuses == {"skill:a": "ok", "skill:b": "uncertain"}

    class MissingAnswer(Backend):
        def decide(self, request):
            return {"answers": {"skill:a": {
                "status": "ok", "probabilities": {"true": .8}, "input_tokens": 3,
            }}}

    with pytest.raises(ValueError, match="exactly the candidate IDs"):
        module.score_backend("rizzo", MissingAnswer(), payload)


def test_payload_preserves_prompt_only_uses_retrieved_memory():
    module = bench()
    prompt = "Analizza il contratto API.\nMantieni questa riga."
    payload = module.decision_payload(prompt, [
        {"kind": "memory", "source": "memory:own", "text": "Own contract"},
        {"kind": "skill", "source": "skill:a", "text": "Skill description"},
    ], [{"source": "skill:a", "text": "Skill description", "title": "Contract"}])
    assert payload["prompt"] == prompt
    assert payload["evidence"] == [{"source": "memory:own", "text": "Own contract"}]
    assert payload["candidates"][0]["id"] == "skill:a"


def test_selector_reads_full_shortlisted_description_without_loading_skill_body(tmp_path):
    module = bench()
    description = "API matching " + "context " * 180 + "DECISIVE TAIL"
    store = module.fixture_store({"skills": [{
        "id": "api", "name": "API", "description": description,
        "content": "PRIVATE SKILL BODY", "plugin": "test", "file_path": "/fixture/api.md",
    }], "tasks": [], "wiki": [], "memories": []}, tmp_path / "fixture.db")
    prompt = "API matching\nKeep original line."
    try:
        baseline, candidates = module.retrieve({"prompt": prompt, "cwd": ""}, store)
        payload = module.decision_payload(prompt, baseline["items"], candidates)
    finally:
        store.close()
    assert len(candidates[0]["text"]) <= 1200
    assert "DECISIVE TAIL" not in candidates[0]["text"]
    assert candidates[0]["_description"] == description
    assert payload["prompt"] == prompt
    assert payload["candidates"][0]["description"] == description
    assert "PRIVATE SKILL BODY" not in str(payload)


def test_retrieve_keeps_conservative_baseline_and_broad_experimental_shortlist(tmp_path):
    module = bench()
    store = module.fixture_store({"skills": [
        {"id": "slides", "name": "slides", "description": "Design clear slide presentations.",
         "content": "slides", "plugin": "test", "file_path": "/fixture/slides.md"},
        {"id": "data", "name": "data-work", "description": "Create spreadsheet formulas and charts.",
         "content": "data", "plugin": "test", "file_path": "/fixture/data.md"},
    ], "tasks": [], "wiki": [], "memories": []}, tmp_path / "fixture.db")
    try:
        baseline, candidates = module.retrieve({
            "prompt": "Create slides and spreadsheet charts", "cwd": ""}, store)
    finally:
        store.close()
    assert [item["source"] for item in baseline["items"] if item["kind"] == "skill"] == [
        "skill:slides"]
    assert [item["source"] for item in candidates] == ["skill:slides", "skill:data"]


def test_missing_scope_prevents_memory_from_entering_model(tmp_path):
    module = bench()
    corpus = {
        "skills": [{"id": "api", "name": "api", "description": "Review an API contract.",
                    "content": "API", "plugin": "test", "file_path": "/fixture/api.md"}],
        "tasks": [], "wiki": [],
        "memories": [{"key": "secret", "project": "/repo/b", "text": "API PRIVATE"}],
    }
    store = module.fixture_store(corpus, tmp_path / "fixture.db")
    try:
        baseline, candidates = module.retrieve({"prompt": "API contract", "cwd": ""}, store)
        payload = module.decision_payload("API contract", baseline["items"], candidates)
        assert payload["evidence"] == []
        assert candidates[0]["source"] == "skill:api"
    finally:
        store.close()


def test_threshold_scores_the_fitted_context_not_raw_selector_ids():
    module = bench()
    evidence = [{"kind": "memory", "source": f"memory:{i}", "title": str(i), "text": "e"}
                for i in range(5)]
    candidates = [
        {"kind": "skill", "source": "skill:wrong", "title": "wrong", "text": "wrong"},
        {"kind": "skill", "source": "skill:right", "title": "right", "text": "right"},
    ]
    calibration = [{"expected": ["skill:wrong", "skill:right"],
                    "scores": {"skill:wrong": .6, "skill:right": .8}, "error": None,
                    "baseline": {"items": evidence}, "candidates": candidates}]

    # Raw scoring prefers .6 because it selects both expected IDs. The fitted output has room
    # for one skill, making .6 and .8 tie; the documented high-threshold tie-break chooses .8.
    assert module.choose_threshold(calibration) == .8


def test_explicit_skill_survives_six_item_fit_without_reordering_other_skills():
    module = bench()
    evidence = [{"kind": "memory", "source": f"memory:{i}", "title": str(i), "text": "e"}
                for i in range(6)]
    candidates = [
        {"kind": "skill", "source": "skill:explicit", "title": "explicit", "text": "requested"},
        {"kind": "skill", "source": "skill:other", "title": "other", "text": "related"},
    ]
    row = {"scores": {"skill:explicit": 1., "skill:other": .8},
           "backend_statuses": {"skill:explicit": "explicit", "skill:other": "ok"},
           "candidates": candidates, "baseline": {"items": evidence}}
    fitted = module.fitted_items(row, .5)
    assert [item["source"] for item in fitted] == [
        "skill:explicit", "memory:0", "memory:1", "memory:2", "memory:3", "memory:4"]
    assert module.select_ids(row["scores"], candidates, .5) == ["skill:explicit", "skill:other"]
    assert module.fitted_items({**row, "backend_statuses": {}}, .5) == evidence


def test_report_distinguishes_raw_selected_skills_from_budget_omissions(monkeypatch):
    module = bench()
    evidence = [{"kind": "memory", "source": f"memory:{i}", "title": str(i), "text": "e"}
                for i in range(6)]
    candidates = [
        {"kind": "skill", "source": "skill:explicit", "title": "explicit", "text": "requested"},
        {"kind": "skill", "source": "skill:other", "title": "other", "text": "related"},
    ]
    monkeypatch.setattr(module, "retrieve", lambda case, store: (
        {"original_prompt": case["prompt"], "items": evidence}, candidates))
    monkeypatch.setattr(module, "score_backend", lambda name, backend, payload: (
        {"skill:explicit": 1., "skill:other": .8}, 10,
        {"skill:explicit": "explicit", "skill:other": "ok"}))
    monkeypatch.setattr(module, "choose_threshold", lambda rows: .5)

    class Tokenizer:
        def encode(self, text):
            return text.split()

    corpus = {"cases": [{"id": "report", "split": "test", "prompt": "$explicit",
                         "cwd": "/repo", "expected_sources": ["skill:explicit"]}]}
    row = module.run(corpus, "qwen3_reranker", object(), Tokenizer())["rows"][0]
    assert row["raw_selected_skills"] == ["skill:explicit", "skill:other"]
    assert row["selected"] == ["skill:explicit"]
    assert row["budget_omitted_skills"] == ["skill:other"]


def test_failures_abstentions_forbidden_hits_and_foreign_leaks_are_separate():
    module = bench()
    rows = [
        {"expected": [], "selected": [], "error": "backend unavailable", "elapsed_ms": 1,
         "primary_tokens": 10, "selector_tokens": 0, "backend_abstentions": 0,
         "forbidden_source_hits": [], "foreign_source_leakage": False},
        {"expected": [], "selected": [], "error": None, "elapsed_ms": 1,
         "primary_tokens": 10, "selector_tokens": 0, "backend_abstentions": 2,
         "forbidden_source_hits": ["task:same-scope"], "foreign_source_leakage": False},
        {"expected": ["skill:a"], "selected": ["skill:a"], "error": None, "elapsed_ms": 1,
         "primary_tokens": 10, "selector_tokens": 2, "backend_abstentions": 0,
         "forbidden_source_hits": [], "foreign_source_leakage": True},
    ]
    summary = module.summarize(rows)
    assert summary["failures"] == 1
    assert summary["correct_abstention_rate"] == .5
    assert summary["backend_abstentions"] == 2
    assert summary["forbidden_source_hit_cases"] == 1
    assert summary["forbidden_source_hits"] == 1
    assert summary["foreign_source_leakage"] == 1


def test_local_models_require_existing_paths():
    module = bench()
    with pytest.raises(ValueError, match="local"):
        module.load_backend("openjev", "Qwen/Qwen2.5-0.5B-Instruct")
    with pytest.raises(ValueError, match="local"):
        module.load_backend("qwen3_reranker", "")


def test_source_hashes_capture_harness_corpus_model_and_backend_revision(tmp_path):
    module = bench()
    corpus = tmp_path / "corpus.json"
    harness = tmp_path / "harness.py"
    model = tmp_path / "model"
    model.mkdir()
    corpus.write_text("corpus")
    harness.write_text("harness")
    (model / "weights.safetensors").write_text("weights")

    hashes = module.capture_source_hashes(corpus, harness, model, "backend-rev")

    assert hashes == {
        "corpus_sha256": hashlib.sha256(b"corpus").hexdigest(),
        "harness_sha256": hashlib.sha256(b"harness").hexdigest(),
        "backend_source_revision": "backend-rev",
        "model_files": {"weights.safetensors": hashlib.sha256(b"weights").hexdigest()},
    }


def test_baseline_run_does_not_try_to_calibrate_selector(monkeypatch):
    module = bench()
    corpus_path = Path(__file__).resolve().parents[1] / "benchmarks" / "local_selector_cases.json"
    corpus = json.loads(corpus_path.read_text())

    def unexpected_calibration(rows):
        raise AssertionError("baseline has no selector scores to calibrate")

    class Tokenizer:
        @staticmethod
        def encode(text):
            return text.split()

    monkeypatch.setattr(module, "choose_threshold", unexpected_calibration)
    result = module.run(corpus, "baseline", None, Tokenizer())
    assert result["threshold"] is None


def test_rizzo_cleanup_closes_session_on_worker_before_shutdown():
    module = bench()
    events = []

    class Result:
        def result(self):
            events.append("result")

    class Worker:
        def submit(self, callback):
            events.append("submit")
            callback()
            return Result()

        def shutdown(self, wait):
            events.append(("shutdown", wait))

    class Session:
        def close(self):
            events.append("close")

    class ModelBackend:
        metadata = {"runtime": "llama.cpp", "device_name": "Fake Metal"}
        session = Session()

    class Engine:
        backend = ModelBackend()
        _worker = Worker()

    engine = Engine()
    assert module.local_backend_metadata("rizzo", engine) == ModelBackend.metadata
    module.close_local_backend("rizzo", engine)
    assert events == ["submit", "close", "result", ("shutdown", True)]


def test_non_rizzo_cleanup_is_a_noop():
    module = bench()
    module.close_local_backend("openjev", object())
    assert module.local_backend_metadata("openjev", object()) is None


@pytest.mark.parametrize("backend_name", ["qwen3_reranker", "kev", "laya"])
def test_local_adapter_preserves_explicit_skill_invocation(backend_name):
    module = bench()
    payload = {"prompt": "Use $api for this request", "evidence": [], "candidates": [
        {"id": "skill:api", "name": "API", "description": "API work"},
        {"id": "skill:maps", "name": "Maps", "description": "Map work"},
    ]}

    class Backend:
        def score(self, request):
            return {"skill:api": .01, "skill:maps": .8}, 12, {
                "skill:api": "ok", "skill:maps": "ok"}

    scores, tokens, statuses = module.score_backend(backend_name, Backend(), payload)
    assert scores == {"skill:api": 1.0, "skill:maps": .8}
    assert tokens == 12
    assert statuses == {"skill:api": "explicit", "skill:maps": "ok"}
    assert module.select_ids(scores, [{"source": key} for key in scores], .9) == ["skill:api"]


def test_qwen_source_hashes_include_adapter_and_skill_retrieval(tmp_path):
    module = bench()
    corpus = tmp_path / "corpus.json"
    harness = tmp_path / "harness.py"
    model = tmp_path / "model"
    model.mkdir()
    corpus.write_text("corpus")
    harness.write_text("harness")
    (model / "config.json").write_text("model")
    hashes = module.capture_source_hashes(corpus, harness, model, None,
                                          backend_name="qwen3_reranker")
    adapter = Path(__file__).resolve().parents[1] / "benchmarks/qwen_reranker.py"
    assert hashes["backend_adapter_sha256"] == hashlib.sha256(adapter.read_bytes()).hexdigest()
    assert hashes["model_files"] == {"config.json": hashlib.sha256(b"model").hexdigest()}
