"""The optional Kev scorer must preserve inputs, reject overflow and reuse exact state."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def selector():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "kev_selector.py"
    spec = importlib.util.spec_from_file_location("kev_selector", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fake_api(monkeypatch):
    class Request:
        def __init__(self, state, questions):
            self.state = state
            self.questions = questions

    def to_record(request):
        return {"state": request.state, "questions": [
            {"instr": question["instructions"]} for question in request.questions.values()]}, None

    monkeypatch.setitem(sys.modules, "kev", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "kev.api", SimpleNamespace(
        SystemOneRequest=Request, to_record=to_record))


class Prob:
    def __init__(self, yes):
        self.yes = yes

    def tolist(self):
        return [1 - self.yes, self.yes]


class Model:
    def __init__(self):
        self.records = []
        self.fresh = 0
        self.reused = 0

    def encode(self, tokenizer, record, *, max_state, max_branch, strict):
        assert strict is True
        self.records.append(record)
        state = [ord(char) for char in record["state"]]
        if len(state) > max_state:
            raise ValueError("state exceeds token limit")
        branches = [[ord(char) for char in row["instr"]] for row in record["questions"]]
        if any(len(state) + len(branch) > max_branch for branch in branches):
            raise ValueError("branch exceeds token limit")
        return {"ids": state + [char for branch in branches for char in branch],
                "seg": [0] * len(state) + [i for i, branch in enumerate(branches, 1)
                                                for _ in branch],
                "state_truncated": False}

    def probs_and_prefix(self, enc):
        self.fresh += 1
        return [Prob(.8), Prob(.2)][:len(set(enc["seg"]) - {0})], object()

    def probs_with_prefix(self, enc, prefix):
        self.reused += 1
        return [Prob(.8), Prob(.2)][:len(set(enc["seg"]) - {0})]


def payload(prompt="Keep this ORIGINAL\nline."):
    return {"prompt": prompt,
            "evidence": [{"source": "memory:own", "text": "Verified API contract"}],
            "candidates": [{"id": "skill:a", "name": "API", "description": "Review API contracts"},
                           {"id": "skill:b", "name": "Maps", "description": "Render maps"}]}


def test_scores_independent_questions_with_complete_input_and_reuses_exact_state(fake_api):
    module = selector()
    model = Model()
    backend = module.KevSelector(None, model, {}, max_state=1000, max_branch=2000,
                                 max_packed=3000)
    scores, tokens, statuses = backend.score(payload())
    assert scores == {"skill:a": .8, "skill:b": .2}
    assert statuses == {"skill:a": "ok", "skill:b": "ok"}
    assert tokens > 0
    assert model.fresh == 1 and model.reused == 0
    assert "Keep this ORIGINAL\nline." in model.records[0]["state"]
    assert "Verified API contract" in model.records[0]["state"]
    assert "Review API contracts" in model.records[0]["questions"][0]["instr"]
    assert "Render maps" in model.records[0]["questions"][1]["instr"]
    assert "expected_sources" not in str(model.records[0])

    backend.score(payload())
    assert model.fresh == 1 and model.reused == 1
    assert backend.metadata["cache_hits"] == 1
    assert backend.metadata["cache_misses"] == 1
    backend.score(payload(prompt="Different request"))
    assert model.fresh == 2 and model.reused == 1


def test_overflow_rejected_before_inference(fake_api):
    module = selector()
    model = Model()
    backend = module.KevSelector(None, model, {}, max_state=100, max_branch=200,
                                 max_packed=300)
    with pytest.raises(ValueError, match="state exceeds"):
        backend.score(payload(prompt="p" * 500))
    assert model.fresh == model.reused == 0

    backend = module.KevSelector(None, model, {}, max_state=1000, max_branch=1000,
                                 max_packed=1000)
    long_description = payload()
    long_description["candidates"][0]["description"] = "d" * 1000
    with pytest.raises(ValueError, match="branch exceeds"):
        backend.score(long_description)
    assert model.fresh == model.reused == 0


def test_rejects_duplicate_or_more_than_twenty_candidates(fake_api):
    module = selector()
    backend = module.KevSelector(None, Model(), {}, max_state=1000, max_branch=2000,
                                 max_packed=3000)
    duplicate = payload()
    duplicate["candidates"][1]["id"] = "skill:a"
    with pytest.raises(ValueError, match="unique"):
        backend.score(duplicate)
    too_many = payload()
    too_many["candidates"] = [dict(id=f"skill:{i}", name="n", description="d")
                              for i in range(21)]
    with pytest.raises(ValueError, match="at most 20"):
        backend.score(too_many)


def test_loader_refuses_device_fallback_and_unpinned_bundle(tmp_path):
    module = selector()
    with pytest.raises(ValueError, match="no device fallback"):
        module.load(tmp_path, "cpu")
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="revision differs"):
        module.load(tmp_path, "mlx")
