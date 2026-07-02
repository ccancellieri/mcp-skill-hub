"""Tests for async remote-ladder escalation when the local LLM is down.

Directive: when Ollama (local L0) is down, the per-prompt hot path still runs
deterministic work synchronously (FTS skills), and delegates the LLM enrichment
(skill lifecycle + rolling summary) to the remote ladder in a detached worker
whose result lands in session state for the NEXT turn.

Covers:
(a) has_remote_provider() — True with a credentialed gateway, False ollama-only.
(b) eval_skill_lifecycle / optimize_prompt accept local_only and fast-fail on the
    hot path (local_only=True) when the daemon is down.
(c) _maybe_spawn_async_enrich gating: config-off, no-remote, and per-session lock.
(d) _run_async_enrich persists a refreshed summary + upgraded skills, escalating
    (local_only=False, model=None), and preserves the hot-path message_count.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore

    monkeypatch.setenv("HOME", str(tmp_path))
    return SkillStore(db_path=tmp_path / "skill_hub.db")


_REG_GATEWAY = [
    {
        "name": "gw", "kind": "openai_compatible",
        "api_base": "https://gw/v1", "api_key": {"source": "inline", "ref": "sk"},
        "enabled": True,
        "models": [{"id": "gw/big", "complexity": "heavy"}],
    },
]
_REG_OLLAMA_ONLY = [
    {
        "name": "local", "kind": "ollama",
        "api_base": "http://localhost:11434", "enabled": True,
        "models": [{"id": "qwen2.5:3b", "complexity": "light"}],
    },
]


# ---------------------------------------------------------------------------
# (a) has_remote_provider
# ---------------------------------------------------------------------------

def _parse(raw_list):
    from skill_hub.llm import registry
    return [p for p in (registry._parse_provider(r) for r in raw_list) if p]


def test_has_remote_provider_true_with_gateway(monkeypatch):
    from skill_hub.llm import escalation

    monkeypatch.setattr(escalation, "load_registry", lambda: _parse(_REG_GATEWAY))
    assert escalation.has_remote_provider() is True


def test_has_remote_provider_false_ollama_only(monkeypatch):
    from skill_hub.llm import escalation

    monkeypatch.setattr(escalation, "load_registry", lambda: _parse(_REG_OLLAMA_ONLY))
    assert escalation.has_remote_provider() is False


# ---------------------------------------------------------------------------
# (b) local_only fast-fail on the hot path
# ---------------------------------------------------------------------------

def test_eval_skill_lifecycle_local_only_fast_fails_when_daemon_down(monkeypatch):
    from skill_hub import embeddings

    monkeypatch.setattr(embeddings, "ollama_daemon_reachable", lambda **k: False,
                        raising=False)
    # patch the symbol used inside _generate's local import site
    import skill_hub.llm.escalation as esc
    monkeypatch.setattr(esc, "ollama_daemon_reachable", lambda **k: False)

    called = {"complete": False}

    def _boom(*a, **k):
        called["complete"] = True
        raise AssertionError("must not reach provider on hot path when down")

    import skill_hub.llm as llm
    monkeypatch.setattr(llm, "get_provider",
                        lambda: type("P", (), {"complete": staticmethod(_boom)})())

    out = embeddings.eval_skill_lifecycle(
        message="do a thing", context_summary="ctx",
        loaded_skills=[{"id": "a", "description": "x"}], candidate_skills=[],
        model="qwen2.5:3b", local_only=True,
    )
    # falls back to keep-loaded without touching the provider
    assert called["complete"] is False
    assert out["keep"] == ["a"]


# ---------------------------------------------------------------------------
# (c) spawn gating + lock
# ---------------------------------------------------------------------------

def test_spawn_gated_off_by_config(tmp_path, monkeypatch):
    import skill_hub.cli as cli
    import skill_hub.config as cfg

    monkeypatch.setattr(cfg, "get",
                        lambda k, d=None: False if k == "hook_async_escalation" else d)
    calls = []
    import subprocess
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: calls.append(a) or _FakeProc())
    cli._maybe_spawn_async_enrich("sess", "a message")
    assert calls == []


def test_spawn_gated_off_when_no_remote(tmp_path, monkeypatch):
    import skill_hub.cli as cli
    import skill_hub.config as cfg
    from skill_hub.llm import escalation

    monkeypatch.setattr(cfg, "get",
                        lambda k, d=None: True if k == "hook_async_escalation" else d)
    monkeypatch.setattr(escalation, "has_remote_provider", lambda: False)
    calls = []
    import subprocess
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: calls.append(a) or _FakeProc())
    cli._maybe_spawn_async_enrich("sess", "a message")
    assert calls == []


def test_spawn_then_locked(tmp_path, monkeypatch):
    import skill_hub.cli as cli
    import skill_hub.config as cfg
    from skill_hub.llm import escalation

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("tempfile.gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(cfg, "get",
                        lambda k, d=None: True if k == "hook_async_escalation" else d)
    monkeypatch.setattr(escalation, "has_remote_provider", lambda: True)
    calls = []
    import subprocess
    monkeypatch.setattr(subprocess, "Popen",
                        lambda *a, **k: calls.append(a) or _FakeProc())

    cli._maybe_spawn_async_enrich("sess-lock", "first message")
    cli._maybe_spawn_async_enrich("sess-lock", "second message")
    assert len(calls) == 1  # second is suppressed by the per-session lock


class _FakeProc:
    stdin = None


# ---------------------------------------------------------------------------
# (d) worker persistence
# ---------------------------------------------------------------------------

def test_worker_persists_refreshed_state(tmp_path, monkeypatch):
    import skill_hub.cli as cli
    from skill_hub.store import SkillStore, Skill

    monkeypatch.setenv("HOME", str(tmp_path))
    db = tmp_path / "skill_hub.db"

    s = SkillStore(db_path=db)
    s.upsert_skill(Skill(id="sk:test", name="Testing", target="claude",
                         description="python testing debugging pytest",
                         content="use pytest", file_path="", plugin="p"))
    s.save_session_context(session_id="wsess", loaded_skills=[],
                           context_summary="old", message_count=7,
                           recent_messages=["prev"])
    s.close()

    # isolate the worker's default-path SkillStore onto our tmp db
    monkeypatch.setattr(cli, "SkillStore", lambda *a, **k: SkillStore(db_path=db))

    seen = {}

    def fake_lifecycle(**kw):
        seen.update(kw)
        return {"keep": [], "add": ["sk:test"], "drop": [],
                "context_summary": "REFRESHED"}

    monkeypatch.setattr(cli, "eval_skill_lifecycle", fake_lifecycle)

    cli._run_async_enrich("wsess", "python testing debugging pytest please")

    assert seen.get("local_only") is False   # escalates off the hot path
    assert seen.get("model") is None         # lets the ladder pick a remote level

    s = SkillStore(db_path=db)
    ctx = s.get_session_context("wsess")
    s.close()
    assert ctx["loaded_skills"] == ["sk:test"]
    assert ctx["context_summary"] == "REFRESHED"
    assert ctx["message_count"] == 7          # hot path owns the counter


def test_worker_noops_on_empty(monkeypatch):
    import skill_hub.cli as cli
    # no session / no message → returns without opening a store
    monkeypatch.setattr(cli, "SkillStore",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("must not open store")))
    cli._run_async_enrich("", "hi")
    cli._run_async_enrich("sess", "")
