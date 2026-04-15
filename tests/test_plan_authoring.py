"""Tests for plan_executor.runner + author — resolution chain, fallback order,
in-session directive, validation-retry loop.

No network, no subprocess, no SDK dependency — everything injected via monkeypatch.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub.plan_executor import (  # noqa: E402
    RunnerFailed,
    RunnerUnavailable,
    author_plan,
    resolve_runner,
)
from skill_hub.plan_executor import runner as _runner  # noqa: E402


VALID_YAML = """plan_id: test
goal: a plan
steps:
  - id: T1
    kind: tests
    files: [x.py]
    acceptance: pytest
"""

INVALID_YAML = """plan_id: test
goal: a plan
steps:
  - id: T1
    kind: WRONG_KIND
    files: [x.py]
    acceptance: pytest
"""


# ---- helpers ----------------------------------------------------------------

def _clear_runner_env(monkeypatch):
    for v in ("HUB_PLAN_RUNNER", "CLAUDECODE", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(v, raising=False)


def _force_all_unavailable(monkeypatch):
    """Make in_session, cli, sdk all unavailable so only api can respond."""
    monkeypatch.delenv("CLAUDECODE", raising=False)
    monkeypatch.setattr(_runner.shutil, "which", lambda _: None)
    # Block SDK import.
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", None)


# ---- resolve_runner: ordering & overrides -----------------------------------

def test_in_session_runner_picked_when_claudecode_set(monkeypatch, tmp_path):
    _clear_runner_env(monkeypatch)
    monkeypatch.setenv("CLAUDECODE", "1")
    result = resolve_runner("g", tmp_path, "schema", tmp_path / "p.yaml")
    assert result.runner == "in_session"
    assert result.directive is not None
    assert "AUTHOR PLAN IN-SESSION" in result.directive


def test_cli_runner_picked_when_claude_on_path(monkeypatch, tmp_path):
    _clear_runner_env(monkeypatch)
    monkeypatch.setattr(_runner.shutil, "which", lambda name: "/fake/claude" if name == "claude" else None)

    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout=VALID_YAML, stderr="")
    monkeypatch.setattr(_runner.subprocess, "run", fake_run)

    result = resolve_runner("g", tmp_path, "schema", tmp_path / "p.yaml")
    assert result.runner == "cli"
    assert result.yaml_text.startswith("plan_id:")


def test_api_runner_last_resort(monkeypatch, tmp_path):
    _clear_runner_env(monkeypatch)
    _force_all_unavailable(monkeypatch)

    chat = lambda messages, **kw: VALID_YAML
    result = resolve_runner(
        "g", tmp_path, "schema", tmp_path / "p.yaml", chat_fn=chat,
    )
    assert result.runner == "api"


def test_api_unavailable_without_key(monkeypatch, tmp_path):
    _clear_runner_env(monkeypatch)
    _force_all_unavailable(monkeypatch)
    # No chat_fn injected and no API key → api must fail with RunnerUnavailable,
    # which bubbles up as RunnerFailed from resolve_runner.
    with pytest.raises(RunnerFailed) as exc:
        resolve_runner("g", tmp_path, "schema", tmp_path / "p.yaml")
    assert "no runner succeeded" in str(exc.value)


def test_api_runner_disabled_by_config_even_with_key(monkeypatch, tmp_path):
    """Default config has plan_api_runner_enabled=False. Even with a key
    present and no injected chat_fn, api must refuse to run."""
    _clear_runner_env(monkeypatch)
    _force_all_unavailable(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-fake")
    # Config default is False — no override.
    with pytest.raises(RunnerFailed) as exc:
        resolve_runner("g", tmp_path, "schema", tmp_path / "p.yaml")
    msg = str(exc.value)
    assert "API runner disabled by config" in msg


def test_explicit_override_forced(monkeypatch, tmp_path):
    _clear_runner_env(monkeypatch)
    # CLAUDECODE set, but user forces api.
    monkeypatch.setenv("CLAUDECODE", "1")
    chat = lambda messages, **kw: VALID_YAML
    result = resolve_runner(
        "g", tmp_path, "schema", tmp_path / "p.yaml",
        preferred="api", chat_fn=chat,
    )
    assert result.runner == "api"


def test_env_override_honored(monkeypatch, tmp_path):
    _clear_runner_env(monkeypatch)
    monkeypatch.setenv("CLAUDECODE", "1")
    monkeypatch.setenv("HUB_PLAN_RUNNER", "api")
    chat = lambda messages, **kw: VALID_YAML
    result = resolve_runner(
        "g", tmp_path, "schema", tmp_path / "p.yaml", chat_fn=chat,
    )
    assert result.runner == "api"


def test_invalid_override_raises(monkeypatch, tmp_path):
    _clear_runner_env(monkeypatch)
    with pytest.raises(ValueError):
        resolve_runner(
            "g", tmp_path, "schema", tmp_path / "p.yaml",
            preferred="bogus",  # type: ignore[arg-type]
        )


# ---- cli runner: error handling ---------------------------------------------

def test_cli_nonzero_exit_raises(monkeypatch, tmp_path):
    _clear_runner_env(monkeypatch)
    monkeypatch.setattr(_runner.shutil, "which", lambda _: "/fake/claude")

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="boom")
    monkeypatch.setattr(_runner.subprocess, "run", fake_run)

    with pytest.raises(RunnerFailed):
        _runner.run_cli("g", tmp_path, "schema", tmp_path / "p.yaml")


def test_cli_strips_yaml_fence(monkeypatch, tmp_path):
    _clear_runner_env(monkeypatch)
    monkeypatch.setattr(_runner.shutil, "which", lambda _: "/fake/claude")
    fenced = f"```yaml\n{VALID_YAML}```"

    def fake_run(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=fenced, stderr="")
    monkeypatch.setattr(_runner.subprocess, "run", fake_run)

    result = _runner.run_cli("g", tmp_path, "schema", tmp_path / "p.yaml")
    assert result.yaml_text.startswith("plan_id:")
    assert "```" not in result.yaml_text


# ---- author_plan: end-to-end with validate-retry ----------------------------

def test_author_plan_writes_file(monkeypatch, tmp_path):
    _clear_runner_env(monkeypatch)
    _force_all_unavailable(monkeypatch)

    chat = lambda messages, **kw: VALID_YAML
    result = author_plan(
        "my awesome plan", repo_path=tmp_path,
        plan_dir=tmp_path / "plans", chat_fn=chat,
    )
    assert result.plan_path is not None
    assert result.plan_path.exists()
    assert result.plan_path.name == "my-awesome-plan.yaml"
    assert result.runner == "api"
    assert result.used_api_tokens is True
    assert result.validation_attempts == 1


def test_author_plan_in_session_returns_directive(monkeypatch, tmp_path):
    _clear_runner_env(monkeypatch)
    monkeypatch.setenv("CLAUDECODE", "1")
    result = author_plan(
        "in session goal", repo_path=tmp_path,
        plan_dir=tmp_path / "plans",
    )
    assert result.runner == "in_session"
    assert result.plan_path is None
    assert result.directive is not None
    assert "validate_plan" in result.directive


def test_author_plan_retry_on_invalid_output(monkeypatch, tmp_path):
    _clear_runner_env(monkeypatch)
    _force_all_unavailable(monkeypatch)

    responses = [INVALID_YAML, VALID_YAML]
    def chat(messages, **kw):
        return responses.pop(0)

    result = author_plan(
        "retry goal", repo_path=tmp_path,
        plan_dir=tmp_path / "plans", chat_fn=chat,
    )
    assert result.validation_attempts == 2
    assert result.plan_path.exists()


def test_author_plan_gives_up_after_max_retries(monkeypatch, tmp_path):
    _clear_runner_env(monkeypatch)
    _force_all_unavailable(monkeypatch)

    chat = lambda messages, **kw: INVALID_YAML
    with pytest.raises(RunnerFailed) as exc:
        author_plan(
            "bad goal", repo_path=tmp_path,
            plan_dir=tmp_path / "plans", chat_fn=chat,
        )
    assert "validation failed" in str(exc.value)


def test_slug_safe_for_filesystem(monkeypatch, tmp_path):
    _clear_runner_env(monkeypatch)
    _force_all_unavailable(monkeypatch)

    chat = lambda messages, **kw: VALID_YAML
    result = author_plan(
        "Fix / bug: slashes & spaces!!!", repo_path=tmp_path,
        plan_dir=tmp_path / "plans", chat_fn=chat,
    )
    # Slashes, colons, and bangs replaced with hyphens.
    assert "/" not in result.plan_path.name
    assert result.plan_path.name.endswith(".yaml")
