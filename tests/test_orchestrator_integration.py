"""Integration tests for explicit tooling and deterministic route context."""
from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from skill_hub.orchestrator import (
    OrchestratorResult,
    dispatch_async,
    ensure_tooling_core,
    evaluate,
)
from skill_hub.orchestrator import engine as _engine


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_code_project(
    tmp_path: Path, *, with_codegraph: bool = False, node_count: int = 5
) -> Path:
    """Create a minimal code project in *tmp_path*.

    Writes a ``pyproject.toml`` and a ``.git/`` directory so that
    ``is_code_project()`` and ``_resolve_project_root()`` both recognise it.

    When *with_codegraph* is set, the ``.codegraph/`` index is populated with a
    ``codegraph.db`` holding *node_count* nodes (a non-empty, usable index by
    default — matching ``codegraph init -i``). The probe rejects an empty index,
    so the database must hold at least one node for the index to read as ready.
    """
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "test"\n')
    (tmp_path / ".git").mkdir()
    if with_codegraph:
        cg = tmp_path / ".codegraph"
        cg.mkdir()
        import sqlite3
        con = sqlite3.connect(cg / "codegraph.db")
        try:
            con.execute("CREATE TABLE nodes (id INTEGER PRIMARY KEY, name TEXT)")
            con.executemany(
                "INSERT INTO nodes (name) VALUES (?)",
                [(f"sym_{i}",) for i in range(node_count)],
            )
            con.commit()
        finally:
            con.close()
        now = time.time()
        os.utime(cg, (now, now))
    return tmp_path


def _make_non_code_dir(tmp_path: Path) -> Path:
    """Create a directory with no code-project markers."""
    (tmp_path / "notes.txt").write_text("just a text file\n")
    return tmp_path


def _config_get_factory(overrides: dict):
    """Return a ``config.get`` replacement that merges *overrides* over _DEFAULTS."""
    from skill_hub import config as _cfg
    base = _cfg._DEFAULTS.copy()
    base.update(overrides)
    return lambda k: base.get(k)


def _orch_enabled_config(**extra):
    return _config_get_factory({
        "orchestrator_enabled": True,
        "orchestrator_auto_init": False,
        "orchestrator_auto_init_roots": [],
        "orchestrator_sync_ttl_secs": 300,
        "orchestrator_probe_cache_secs": 60,
        **extra,
    })


def _mark_index_stale(tmp_path: Path) -> None:
    """Backdate the db and drop a newer ``.dirty`` so the index reads as stale."""
    cg = tmp_path / ".codegraph"
    now = time.time()
    os.utime(cg / "codegraph.db", (now - 30, now - 30))
    (cg / ".dirty").write_text(str(int(now * 1000)))


class TestExplicitOrchestratorDisabled:
    def test_evaluate_disabled_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            "skill_hub.config.get",
            lambda k: False if k == "orchestrator_enabled" else None,
        )
        result = evaluate("/tmp", "explore everything")
        assert result.directive == ""
        assert result.decisions == []
        assert result.provision_actions == []


class TestRouteContextOnly:
    def test_route_passes_exact_prompt_and_scoped_identity_to_context(self, monkeypatch):
        import skill_hub.router.route as route_mod

        prompt = "Plan this work.\nKeep every line exactly as written."
        received = {}

        def fake_build_context(value, **kwargs):
            received.update(prompt=value, **kwargs)
            return {"context": "[skill] Use the local pattern."}

        monkeypatch.setattr(route_mod._cfg, "load_config", lambda: {"router_enabled": True, "hook_enabled": True, "context_enabled": True})
        monkeypatch.setattr("skill_hub.context_service.build_context", fake_build_context)

        result = route_mod.route(prompt, cwd="/work/alpha", session_id="session-7", task_id=3)

        assert received == {
            "prompt": prompt, "cwd": "/work/alpha", "session_id": "session-7",
            "task_id": 3, "cfg": {"router_enabled": True, "hook_enabled": True, "context_enabled": True},
        }
        assert result == {"userMessage": "[skill] Use the local pattern."}
        assert prompt == "Plan this work.\nKeep every line exactly as written."

    def test_route_never_calls_llm_enforcement_or_provisioning(self, monkeypatch):
        import skill_hub.router.route as route_mod

        def forbidden(*_args, **_kwargs):
            pytest.fail("retired route dependency was called")

        old_enabled = {
            "router_enabled": True, "hook_enabled": True, "context_enabled": True,
            "router_haiku_classify": True, "orchestrator_enabled": True,
            "orchestrator_auto_init": True,
        }
        monkeypatch.setattr(route_mod._cfg, "load_config", lambda: old_enabled)
        monkeypatch.setattr("skill_hub.router.ollama_client.classify", forbidden)
        monkeypatch.setattr("skill_hub.router.haiku_client.classify", forbidden)
        monkeypatch.setattr("skill_hub.orchestrator.engine.evaluate", forbidden)
        monkeypatch.setattr("skill_hub.orchestrator.engine.dispatch_async", forbidden)
        monkeypatch.setattr("skill_hub.router.enforcement.apply", forbidden, raising=False)
        monkeypatch.setattr("skill_hub.context_service.build_context", lambda *_args, **_kwargs: {"context": "evidence"})

        assert route_mod.route("explore the codebase", cwd="/work/alpha") == {"userMessage": "evidence"}

    def test_disabled_or_failed_context_returns_empty(self, monkeypatch):
        import skill_hub.router.route as route_mod

        monkeypatch.setattr(route_mod._cfg, "load_config", lambda: {"context_enabled": False})
        assert route_mod.route("prompt") == {}

        monkeypatch.setattr(route_mod._cfg, "load_config", lambda: {"router_enabled": True, "hook_enabled": True, "context_enabled": True})
        monkeypatch.setattr("skill_hub.context_service.build_context", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unavailable")))
        assert route_mod.route("prompt") == {}


# ---------------------------------------------------------------------------
# 5. ensure_tooling_core — idempotency + non-fatal
# ---------------------------------------------------------------------------

class TestEnsureToolingCore:
    @pytest.fixture(autouse=True)
    def _codegraph_installed(self, monkeypatch):
        # Dispatch-behaviour tests must not depend on the host having
        # codegraph installed; Popen is stubbed wherever dispatch fires.
        monkeypatch.setattr(
            "skill_hub.orchestrator.registry._resolve_codegraph_bin",
            lambda: "/usr/local/bin/codegraph",
        )

    def test_nonexistent_path_returns_dict_no_exception(self):
        result = ensure_tooling_core("/nonexistent/path/xyz/abc")
        assert isinstance(result, dict)
        for key in ("path", "present", "fresh", "action", "directive"):
            assert key in result, f"missing key: {key}"

    def test_nonexistent_path_present_false(self):
        result = ensure_tooling_core("/nonexistent/path/xyz/abc", refresh=False)
        assert result["present"] is False

    def test_present_index_with_refresh_true_records_refresh(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path, with_codegraph=True)
        _engine._last_dispatch.clear()
        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: None)

        result = ensure_tooling_core(str(tmp_path), refresh=True)

        assert isinstance(result, dict)
        assert result["present"] is True
        assert result["action"] == "refresh_dispatched"

    def test_calling_twice_does_not_raise(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path, with_codegraph=True)
        _engine._last_dispatch.clear()
        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: None)

        first = ensure_tooling_core(str(tmp_path), refresh=True)
        second = ensure_tooling_core(str(tmp_path), refresh=True)

        assert isinstance(first, dict)
        assert isinstance(second, dict)

    def test_absent_index_no_init_action_none(self, tmp_path):
        _make_code_project(tmp_path)
        result = ensure_tooling_core(str(tmp_path), init=False, refresh=False)
        assert result["present"] is False
        assert result["action"] in ("none", "error")

    def test_required_keys_always_present(self, tmp_path):
        result = ensure_tooling_core(str(tmp_path))
        expected = {"path", "present", "fresh", "action", "directive"}
        assert expected <= set(result.keys()), (
            f"missing keys: {expected - set(result.keys())}"
        )

    def test_refresh_dispatched_captures_argv(self, tmp_path, monkeypatch):
        """Ensure the argv dispatched for refresh looks like a codegraph sync call."""
        _make_code_project(tmp_path, with_codegraph=True)
        _engine._last_dispatch.clear()
        captured: list = []

        class _FakePopen:
            def __init__(self, argv, **kw):
                captured.append(argv)

        monkeypatch.setattr("subprocess.Popen", _FakePopen)

        result = ensure_tooling_core(str(tmp_path), refresh=True)

        assert result["action"] == "refresh_dispatched"
        assert captured, "expected Popen to be called for refresh"
        argv = captured[0]
        assert "sync" in argv, f"expected 'sync' in argv: {argv}"


# ---------------------------------------------------------------------------
# 6. Never-blocks / never-raises
# ---------------------------------------------------------------------------

class TestNeverBlocksNeverRaises:
    def test_evaluate_garbage_cwd_returns_orchestrator_result(self):
        result = evaluate("\x00 garbage [", "garbage \x00 message [")
        assert isinstance(result, OrchestratorResult)
        assert isinstance(result.directive, str)
        assert isinstance(result.decisions, list)
        assert isinstance(result.provision_actions, list)

    def test_evaluate_garbage_does_not_raise(self):
        try:
            evaluate("\x00/not/real\x00", "\x00 garbage [")
        except Exception as exc:
            pytest.fail(f"evaluate() raised unexpectedly: {exc}")

    def test_dispatch_async_nonexistent_binary_does_not_raise(self, monkeypatch):
        """dispatch_async with a non-existent binary must not raise."""
        _engine._last_dispatch.clear()
        # Disable debounce by using a unique never-seen-before argv.
        argv = ["__definitely_not_a_real_binary__xyz_test__", "arg1"]
        try:
            dispatch_async([argv])
        except Exception as exc:
            pytest.fail(f"dispatch_async() raised unexpectedly: {exc}")

    def test_dispatch_async_empty_list_noop(self):
        try:
            dispatch_async([])
        except Exception as exc:
            pytest.fail(f"dispatch_async([]) raised unexpectedly: {exc}")

    def test_dispatch_async_popen_failure_does_not_raise(self, monkeypatch):
        def _bad_popen(*a, **kw):
            raise OSError("intentional test failure")

        monkeypatch.setattr("subprocess.Popen", _bad_popen)
        _engine._last_dispatch.clear()
        try:
            dispatch_async([["codegraph", "sync", "/some/path"]])
        except Exception as exc:
            pytest.fail(f"dispatch_async() raised on Popen failure: {exc}")

    def test_ensure_tooling_core_garbage_path_does_not_raise(self):
        try:
            result = ensure_tooling_core("\x00/bad/path\x00", init=False, refresh=False)
        except Exception as exc:
            pytest.fail(f"ensure_tooling_core() raised unexpectedly: {exc}")
        assert isinstance(result, dict)

    def test_route_never_raises_on_bad_cwd(self, monkeypatch):
        monkeypatch.setattr("skill_hub.config.get", _orch_enabled_config())
        monkeypatch.setattr(
            "skill_hub.orchestrator.engine.dispatch_async",
            lambda actions: None,
        )

        from skill_hub.router.route import route
        try:
            route(
                "explore something",
                session_id="t",
                cwd="\x00/not/a/real/path\x00",
                task_id=None,
            )
        except Exception as exc:
            pytest.fail(f"route() raised unexpectedly: {exc}")
