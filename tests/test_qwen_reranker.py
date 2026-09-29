"""The experimental Qwen reranker must score only local, complete inputs."""
import importlib.util
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"


def reranker():
    spec = importlib.util.spec_from_file_location("qwen_reranker", BENCHMARKS / "qwen_reranker.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_scores_yes_no_logits_for_each_candidate_without_sending_labels():
    torch = pytest.importorskip("torch")
    module = reranker()

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

        def convert_tokens_to_ids(self, token):
            return {"no": 1, "yes": 2}[token]

        def pad(self, batch, padding, return_tensors):
            ids = batch["input_ids"]
            width = max(map(len, ids))
            return {"input_ids": torch.tensor([[0] * (width - len(row)) + row for row in ids]),
                    "attention_mask": torch.tensor([[0] * (width - len(row)) + [1] * len(row) for row in ids])}

    class Model:
        device = "cpu"

        def __init__(self):
            self.received = []

        def __call__(self, **inputs):
            self.received.append(inputs)
            assert inputs["logits_to_keep"] == 1
            assert inputs["use_cache"] is False
            assert len(inputs["input_ids"]) == 1
            logits = torch.zeros((1, len(inputs["input_ids"][0]), 3))
            logits[0, -1, 2 if len(self.received) == 1 else 1] = 2
            return SimpleNamespace(logits=logits)

    model = Model()
    backend = module.QwenReranker(Tokenizer(), model, torch, max_length=2000)
    payload = {"prompt": "Keep this ORIGINAL\nline.",
               "evidence": [{"source": "memory:own", "text": "Verified API contract"}],
               "candidates": [{"id": "skill:a", "name": "API", "description": "Review API contracts"},
                              {"id": "skill:b", "name": "Maps", "description": "Render maps"}]}

    scores, tokens, statuses = backend.score(payload)

    assert set(scores) == {"skill:a", "skill:b"}
    assert scores["skill:a"] == pytest.approx(1 / (1 + math.exp(-2)))
    assert scores["skill:b"] == pytest.approx(1 / (1 + math.exp(2)))
    assert tokens > 0
    assert statuses == {"skill:a": "ok", "skill:b": "ok"}
    decoded = ["".join(chr(x) for x in call["input_ids"][0].tolist() if x) for call in model.received]
    assert all("Keep this ORIGINAL\nline." in row for row in decoded)
    assert all("Verified API contract" in row for row in decoded)
    assert all("expected_sources" not in row for row in decoded)
    assert "Review API contracts" in decoded[0]
    assert "Render maps" in decoded[1]


def test_overlong_original_prompt_is_rejected_before_inference():
    module = reranker()

    class Tokenizer:
        def encode(self, text, add_special_tokens=False):
            return [ord(char) for char in text]

        def convert_tokens_to_ids(self, token):
            return {"no": 1, "yes": 2}[token]

    class Model:
        def __call__(self, **kwargs):
            raise AssertionError("overlong prompt reached inference")

    backend = module.QwenReranker(Tokenizer(), Model(), object(), max_length=256)
    with pytest.raises(ValueError, match="exceeds.*token limit"):
        backend.score({"prompt": "p" * 500, "evidence": [], "candidates": [
            {"id": "skill:a", "name": "API", "description": "review"}]})


def test_local_loader_sets_offline_guards_and_disables_remote_code(monkeypatch, tmp_path):
    module = reranker()
    calls = []

    class LoadedModel:
        device = "cpu"
        dtype = "torch.bfloat16"

        def eval(self):
            return self

    class Loader:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.append((cls.__name__, path, kwargs))
            return LoadedModel() if cls.__name__ == "Model" else SimpleNamespace(
                encode=lambda text, add_special_tokens=False: [ord(char) for char in text],
                convert_tokens_to_ids=lambda token: {"no": 1, "yes": 2}[token])

    class Tokenizer(Loader):
        pass

    class Model(Loader):
        pass

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=Tokenizer, AutoModelForCausalLM=Model))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: {
        "torch": "test-torch", "transformers": "test-transformers"}[name])
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "0")
    backend = module.load(tmp_path, "cpu")
    assert isinstance(backend, module.QwenReranker)
    assert [call[0] for call in calls] == ["Tokenizer", "Model"]
    assert all(call[1] == str(tmp_path.resolve()) for call in calls)
    assert all(call[2]["local_files_only"] is True and call[2]["trust_remote_code"] is False for call in calls)
    assert calls[1][2]["dtype"] == "auto"
    assert module.os.environ["HF_HUB_OFFLINE"] == "1"
    assert module.os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert backend.metadata["torch_version"] == "test-torch"
    assert backend.metadata["transformers_version"] == "test-transformers"
    assert backend.metadata["model_dtype"] == "torch.bfloat16"
