"""Boundary tests for deterministic context adapters and the prompt hook."""
from __future__ import annotations

import io
import json
import runpy
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _prompt_router_main():
    return runpy.run_path(str(ROOT / "hooks" / "prompt_router.py"))["main"]


def test_context_cli_accepts_optional_runtime_without_changing_context(monkeypatch):
    from skill_hub import context_cli

    observed = []
    expected = {"original_prompt": "keep me", "context": "same", "items": []}
    monkeypatch.setattr(context_cli, "build_context", lambda *_args, **_kwargs: expected.copy())
    monkeypatch.setattr(context_cli, "observe_runtime", lambda value, **kwargs: observed.append((value, kwargs)))

    result = context_cli.prepare_request({
        "prompt": "keep me", "session_id": "s",
        "runtime": {"client": {"id": "codex"}, "session": {"id": "s"}},
    })

    assert result == expected
    assert observed[0][1]["default_source"] == "caller_reported"


def test_context_cli_ignores_malformed_runtime(monkeypatch):
    from skill_hub import context_cli

    expected = {"original_prompt": "keep me", "context": "", "items": []}
    monkeypatch.setattr(context_cli, "build_context", lambda *_args, **_kwargs: expected.copy())
    monkeypatch.setattr(context_cli, "observe_runtime", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("bad telemetry")))

    assert context_cli.prepare_request({"prompt": "keep me", "runtime": "bad"}) == expected


def test_context_cli_records_adapter_payload_as_adapter_reported(monkeypatch, tmp_path):
    from skill_hub import context_cli, runtime_context

    monkeypatch.setattr(runtime_context, "RUNTIME_DB_PATH", tmp_path / "runtime.db")
    monkeypatch.setattr(context_cli, "build_context", lambda prompt, **_kwargs: {
        "original_prompt": prompt, "context": "", "items": [],
    })
    context_cli.prepare_request({
        "prompt": "keep me",
        "runtime": {
            "client": {"id": "pi"}, "session": {"id": "pi-session"},
            "model": {"id": "model-a"},
            "provenance": {"model_id": "native_event"},
        },
    }, runtime_source="adapter_reported")

    row = runtime_context.list_runtime_sessions()[0]
    assert row["provenance"]["model_id"] == "adapter_reported"
    assert row["provenance"]["client_id"] == "adapter_reported"


def test_context_hook_uses_event_identity_without_global_task_marker(monkeypatch):
    from skill_hub import context_hook

    received = {}

    def fake_route(prompt, *, session_id="", cwd="", task_id=None):
        received.update(
            prompt=prompt, session_id=session_id, cwd=cwd, task_id=task_id,
        )
        return {"userMessage": "evidence"}

    monkeypatch.setattr(context_hook, "route", fake_route)
    monkeypatch.setenv("SKILL_HUB_ACTIVE_TASK_ID", "foreign-task")
    prompt = "First line.\nSecond line must not change."

    output = context_hook.context_output({
        "prompt": prompt, "cwd": "/projects/alpha", "session_id": "session-a",
    })

    assert received == {
        "prompt": prompt, "cwd": "/projects/alpha", "session_id": "session-a",
        "task_id": None,
    }
    assert output["hookSpecificOutput"]["additionalContext"] == "evidence"
    assert prompt == "First line.\nSecond line must not change."


def test_context_hook_returns_empty_when_route_fails(monkeypatch):
    from skill_hub import context_hook

    def boom(*_args, **_kwargs):
        raise RuntimeError("context unavailable")

    monkeypatch.setattr(context_hook, "route", boom)

    assert context_hook.context_output({"prompt": "hello"}) == {}


def test_context_hook_observes_native_claude_event_fields_fail_soft(monkeypatch):
    from skill_hub import context_hook

    observed = []
    monkeypatch.setattr(context_hook, "observe_runtime", lambda value, **kwargs: observed.append((value, kwargs)))
    monkeypatch.setattr(context_hook, "route", lambda *_args, **_kwargs: {"userMessage": "evidence"})

    context_hook.context_output({
        "prompt": "hello", "session_id": "claude-session",
        "model": "claude-opus-4-1", "permission_mode": "default",
    })

    value, kwargs = observed[0]
    assert value["client"]["id"] == "claude-code"
    assert value["session"]["id"] == "claude-session"
    assert value["model"]["id"] == "claude-opus-4-1"
    assert kwargs["default_source"] == "native_event"


def test_prompt_router_passes_exact_original_event_to_worker(monkeypatch, capsys):
    main = _prompt_router_main()
    prompt = "Keep this line.\nAnd this exact second line."
    event = {"prompt": prompt, "cwd": "/projects/alpha", "session_id": "session-a"}
    received = {}

    def fake_run(argv, **kwargs):
        received.update(argv=argv, **kwargs)
        return subprocess.CompletedProcess(argv, 0, '{"hookSpecificOutput":{"additionalContext":"evidence"}}', "")

    monkeypatch.setattr("skill_hub.config.load_config", lambda: {
        "context_enabled": True, "hook_enabled": True, "context_hook_timeout_s": 2.0,
    })
    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(event)))

    main()

    assert json.loads(received["input"]) == event
    assert received["timeout"] == 2.0
    assert json.loads(capsys.readouterr().out) == {
        "hookSpecificOutput": {"additionalContext": "evidence"},
    }
    assert prompt == "Keep this line.\nAnd this exact second line."


def test_prompt_router_timeout_is_quiet(monkeypatch, capsys):
    main = _prompt_router_main()

    def timeout(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("context_hook", 2.0)

    monkeypatch.setattr("skill_hub.config.load_config", lambda: {
        "context_enabled": True, "hook_enabled": True, "context_hook_timeout_s": 2.0,
    })
    monkeypatch.setattr(subprocess, "run", timeout)
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"prompt":"original"}'))

    main()

    assert capsys.readouterr().out == ""


def test_prompt_router_disabled_is_quiet_and_does_not_launch_worker(monkeypatch, capsys):
    main = _prompt_router_main()

    monkeypatch.setattr("skill_hub.config.load_config", lambda: {
        "context_enabled": False, "hook_enabled": True,
    })
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("worker launched")))
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"prompt":"original"}'))

    main()

    assert capsys.readouterr().out == ""
