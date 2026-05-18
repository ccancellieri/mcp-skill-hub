"""Tests for the sandbox interface (M2/W5).

Covers:
* Policy off (no policy.yml) → ``provision`` is a pass-through.
* Policy on, ``subprocess`` mode → subprocess.run is wrapped:
  - cwd is forced to a temp dir,
  - env is stripped (no network-proxy / token vars),
  - executables outside the PATH allowlist are rejected,
  - acceptance command writing outside cwd raises SandboxViolation.
* ``author_plan`` is wired to ``native`` (no wrapping) per resolved scope.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub import sandbox  # noqa: E402
from skill_hub.sandbox import (  # noqa: E402
    SandboxViolation,
    load_policy,
    provision,
)


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


def test_load_policy_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert load_policy() == {}


def test_load_policy_reads_yaml(tmp_path, monkeypatch):
    (tmp_path / "policy.yml").write_text(
        "sandbox:\n  enabled: true\n  modes:\n    run_plan: subprocess\n"
    )
    monkeypatch.chdir(tmp_path)
    pol = load_policy()
    assert pol["enabled"] is True
    assert pol["modes"]["run_plan"] == "subprocess"


# ---------------------------------------------------------------------------
# Default (sandbox disabled) — pass-through
# ---------------------------------------------------------------------------


def test_provision_passthrough_when_no_policy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run = provision("run_plan")
    assert run.mode == "native"  # type: ignore[attr-defined]

    def inner(x, y):
        return x + y

    assert run(inner, 2, 3) == 5


def test_provision_passthrough_when_disabled(tmp_path, monkeypatch):
    (tmp_path / "policy.yml").write_text("sandbox:\n  enabled: false\n")
    monkeypatch.chdir(tmp_path)
    run = provision("run_plan")
    assert run.mode == "native"  # type: ignore[attr-defined]


def test_author_plan_stays_native_even_when_enabled(tmp_path, monkeypatch):
    (tmp_path / "policy.yml").write_text(
        "sandbox:\n  enabled: true\n  modes:\n"
        "    run_plan: subprocess\n"
        "    execute_plan_step: subprocess\n"
        "    author_plan: native\n"
    )
    monkeypatch.chdir(tmp_path)
    run = provision("author_plan")
    assert run.mode == "native"  # type: ignore[attr-defined]


def test_unknown_tool_passes_through_when_enabled(tmp_path, monkeypatch):
    """Tools outside the plan-execution set must never be sandboxed."""
    (tmp_path / "policy.yml").write_text(
        "sandbox:\n  enabled: true\n  modes:\n    search_skills: subprocess\n"
    )
    monkeypatch.chdir(tmp_path)
    run = provision("search_skills")
    assert run.mode == "native"  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# subprocess mode — env / cwd / executable enforcement
# ---------------------------------------------------------------------------


def _enabled_policy() -> dict:
    return {
        "enabled": True,
        "modes": {"run_plan": "subprocess",
                  "execute_plan_step": "subprocess",
                  "author_plan": "native"},
    }


def test_subprocess_mode_strips_env_and_forces_cwd(tmp_path, monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://evil.example:8080")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret-token-do-not-leak")
    monkeypatch.setenv("ARBITRARY_OUTSIDE_VAR", "x")

    captured: dict = {}

    def fake_run(*args, **kwargs):
        captured["env"] = kwargs.get("env")
        captured["cwd"] = kwargs.get("cwd")
        return subprocess.CompletedProcess(args=args, returncode=0,
                                           stdout=b"", stderr=b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    run = provision("run_plan", policy=_enabled_policy())

    def body():
        # Plan tool body shells out; sandbox must hijack this call.
        subprocess.run(["/bin/echo", "hi"], capture_output=True)
        return "ok"

    assert run(body) == "ok"
    env = captured["env"]
    cwd = captured["cwd"]

    # Network-proxy and credential keys must be absent.
    assert "HTTPS_PROXY" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "ARBITRARY_OUTSIDE_VAR" not in env

    # PATH is rebuilt from the allowlist.
    assert env["PATH"]
    for d in env["PATH"].split(os.pathsep):
        assert d in {"/usr/bin", "/bin"}

    # cwd points at a fresh temp dir, not the user's repo.
    # (The temp dir is removed on context-exit; we only check the path shape.)
    assert cwd is not None
    assert "hub-sbx-" in str(cwd) or str(cwd).startswith("/tmp") \
        or str(cwd).startswith("/var")


def test_subprocess_mode_rejects_cwd_escape(tmp_path, monkeypatch):
    """A plan step that hard-codes an outside cwd is refused."""
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: subprocess.CompletedProcess(
                            args=a, returncode=0))

    run = provision("run_plan", policy=_enabled_policy())

    outside = tmp_path / "not-allowed"
    outside.mkdir()

    def body():
        subprocess.run(["/bin/echo", "x"], cwd=str(outside))

    with pytest.raises(SandboxViolation):
        run(body)


def test_subprocess_mode_rejects_executable_outside_allowlist(monkeypatch):
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: subprocess.CompletedProcess(
                            args=a, returncode=0))
    run = provision("run_plan", policy=_enabled_policy())

    def body():
        # /usr/local/bin is *not* in the default allowlist — must be refused.
        subprocess.run(["/usr/local/bin/curl", "https://example.com"])

    with pytest.raises(SandboxViolation):
        run(body)


def test_subprocess_mode_restores_real_subprocess_after_exit(monkeypatch):
    """After the sandboxed call returns, subprocess.run must be the real one."""
    original = subprocess.run
    run = provision("run_plan", policy=_enabled_policy())

    def body():
        return "done"

    run(body)
    assert subprocess.run is original


def test_subprocess_mode_propagates_callable_return(monkeypatch):
    run = provision("run_plan", policy=_enabled_policy())

    def body(a, b, *, c):
        return (a, b, c)

    assert run(body, 1, 2, c=3) == (1, 2, 3)


def test_sequential_provision_calls_restore_correctly(monkeypatch):
    """Two sandboxed calls in sequence must leave subprocess.run pristine."""
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **kw: subprocess.CompletedProcess(
                            args=a, returncode=0))
    original = subprocess.run

    run1 = provision("run_plan", policy=_enabled_policy())
    run2 = provision("execute_plan_step", policy=_enabled_policy())

    def body():
        subprocess.run(["/bin/echo", "ok"])
        return "done"

    assert run1(body) == "done"
    assert run2(body) == "done"
    assert subprocess.run is original
