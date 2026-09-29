"""The optional Laya adapter must reject SDK clipping and keep scoring local."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def selector(monkeypatch):
    module_path = Path(__file__).resolve().parents[1] / "benchmarks" / "laya_selector.py"
    spec = importlib.util.spec_from_file_location("laya_selector", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def encode_text(tok, text, **kwargs):
        assert kwargs.get("truncation") is not True
        return {"input_ids": text.split()}

    def render_options(q):
        return [f"{key}: {value}" for key, value in q["crit"].items()]

    monkeypatch.setitem(sys.modules, "laya.common", SimpleNamespace(
        encode_text=encode_text, render_options=render_options))
    return module


class Device(str):
    @property
    def type(self):
        return str(self)


class Agent:
    device = Device("mps")
    dtype = "float32"
    amp_enabled = False
    mps_amp_min_rows = 5
    tok = SimpleNamespace(mask_token="[MASK]")
    cpu_fallback_count = 0

    def __init__(self):
        self.calls = []

    @staticmethod
    def _to_internal(q):
        return {"t": q["type"], "ins": q["instructions"], "crit": q["criteria"]}

    def system_one(self, state, questions, **kwargs):
        self.calls.append((state, questions, kwargs))
        return {"answers": {key: {"probabilities": {"A": 0.75, "B": 0.25}}
                            for key in questions}, "usage": {"input_tokens": 193}}


def payload(description="Review API contracts"):
    return {"prompt": "Keep this ORIGINAL\nline.",
            "evidence": [{"source": "memory:own", "text": "Verified API contract"}],
            "candidates": [{"id": "skill:a", "name": "API", "description": description},
                           {"id": "skill:b", "name": "Maps", "description": "Render maps"}]}


def test_independent_neutral_choices_preserve_full_prompt_evidence_and_descriptions(monkeypatch):
    module = selector(monkeypatch)
    agent = Agent()
    scores, tokens, statuses = module.LayaSelector(agent).score(payload())
    assert scores == {"skill:a": 0.75, "skill:b": 0.75}
    assert tokens == 193
    assert statuses == {"skill:a": "ok", "skill:b": "ok"}
    state, questions, kwargs = agent.calls[0]
    assert "Keep this ORIGINAL\nline." in state
    assert "[memory:own] Verified API contract" in state
    assert questions["skill:a"]["instructions"].endswith("Description: Review API contracts")
    assert questions["skill:b"]["instructions"].endswith("Description: Render maps")
    assert all(q["type"] == "choice" and list(q["criteria"]) == ["A", "B"]
               for q in questions.values())
    assert kwargs == {"max_len": 8192, "head_max_len": 512}
    assert "expected_sources" not in state + str(questions)


@pytest.mark.parametrize("field,value,message", [
    ("prompt", "word " * 8200, "exceeds 8192 token limit"),
    ("description", "word " * 520, "exceeds head token limit"),
    ("prompt", "contains [MASK] special token", "contains a mask token"),
], ids=["state-overflow", "head-overflow", "mask-token"])
def test_rejects_sdk_clipping_before_inference(monkeypatch, field, value, message):
    module = selector(monkeypatch)
    agent = Agent()
    case = payload()
    if field == "description":
        case["candidates"][0]["description"] = value
    else:
        case[field] = value
    with pytest.raises(ValueError, match=message):
        module.LayaSelector(agent).score(case)
    assert not agent.calls


def test_rejects_more_than_twenty_questions(monkeypatch):
    module = selector(monkeypatch)
    case = payload()
    case["candidates"] = [{"id": f"skill:{n}", "name": "Skill", "description": "Useful"}
                          for n in range(21)]
    with pytest.raises(ValueError, match="at most 20"):
        module.LayaSelector(Agent()).score(case)


def test_rejects_option_clipping_before_inference(monkeypatch):
    module = selector(monkeypatch)
    monkeypatch.setattr(module, "OPTIONS", {"A": "word " * 49, "B": "other"})
    agent = Agent()
    with pytest.raises(ValueError, match="option exceeds SDK 48-token limit"):
        module.LayaSelector(agent).score(payload())
    assert not agent.calls


def test_loader_refuses_missing_model_without_importing_runtime(monkeypatch, tmp_path):
    module = selector(monkeypatch)
    with pytest.raises(ValueError, match="existing local Laya model directory"):
        module.load(tmp_path / "missing")
