"""The portable context contract is independent of host-specific events."""
import pytest

from skill_hub import context_cli


@pytest.mark.parametrize("client", ["claude", "codex", "pi", "openclaw"])
def test_clients_share_exact_prompt_and_scope(monkeypatch, client):
    seen = {}

    def build(prompt, **kwargs):
        seen.update(prompt=prompt, **kwargs)
        return {"original_prompt": prompt, "context": "evidence"}

    monkeypatch.setattr(context_cli, "build_context", build)
    prompt = "  inspect this\nthen test\n"
    result = context_cli.prepare_request({"prompt": prompt, "cwd": "/projects/example",
                                          "session_id": client, "task_id": 7})
    assert result["original_prompt"] == prompt
    assert seen == {"prompt": prompt, "cwd": "/projects/example",
                    "session_id": client, "task_id": 7}


@pytest.mark.parametrize("data", [None, [], {}, {"prompt": 5},
    {"prompt": "x", "cwd": []}, {"prompt": "x", "session_id": 5},
    {"prompt": "x", "task_id": True}, {"prompt": "x", "task_id": -1}])
def test_invalid_identity_is_rejected_before_retrieval(monkeypatch, data):
    monkeypatch.setattr(context_cli, "build_context", lambda *a, **kw: pytest.fail("retrieved"))
    with pytest.raises(ValueError):
        context_cli.prepare_request(data)
