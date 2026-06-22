"""Integration tests for the tooling orchestrator feature.

Covers the route-injection path (``skill_hub.router.route.route``) and the
explicit ``ensure_tooling_core`` path end-to-end.

Constraints enforced here:
- ``skill_hub.server`` is NEVER imported (would open a live DB).
- No real subprocess is launched: ``skill_hub.orchestrator.engine.dispatch_async``
  (and ``subprocess.Popen``) are monkeypatched in every test that could trigger
  provisioning.
"""
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


# ---------------------------------------------------------------------------
# 1. Route injection — missing index (offer path)
# ---------------------------------------------------------------------------

class TestRouteInjectionMissingIndex:
    """route() injects a [tooling] offer directive when the index is absent."""

    def test_system_message_contains_tooling_directive(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path)
        dispatched: list[list[str]] = []

        monkeypatch.setattr("skill_hub.config.get", _orch_enabled_config())
        monkeypatch.setattr(
            "skill_hub.orchestrator.engine.dispatch_async",
            lambda actions: dispatched.extend(actions),
        )
        _engine._probe_cache.clear()

        from skill_hub.router.route import route
        result = route(
            "explore how the auth module works",
            session_id="t",
            cwd=str(tmp_path),
            task_id=None,
        )

        assert isinstance(result, dict)
        sys_msg = result.get("systemMessage", "")
        assert "[tooling]" in sys_msg, f"no [tooling] directive in: {sys_msg!r}"

    def test_missing_directive_mentions_path_or_offer(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path)

        monkeypatch.setattr("skill_hub.config.get", _orch_enabled_config())
        monkeypatch.setattr(
            "skill_hub.orchestrator.engine.dispatch_async",
            lambda actions: None,
        )
        _engine._probe_cache.clear()

        from skill_hub.router.route import route
        result = route(
            "explore how the auth module works",
            session_id="t",
            cwd=str(tmp_path),
            task_id=None,
        )

        sys_msg = result.get("systemMessage", "")
        # The missing directive must either mention the path or the offer text.
        assert (str(tmp_path) in sys_msg or "offer" in sys_msg or "not indexed" in sys_msg), (
            f"expected path/offer mention in: {sys_msg!r}"
        )

    def test_no_real_provisioning_ran(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path)
        popen_called = []

        monkeypatch.setattr("skill_hub.config.get", _orch_enabled_config())
        monkeypatch.setattr(
            "skill_hub.orchestrator.engine.dispatch_async",
            lambda actions: None,  # swallow; no Popen
        )
        monkeypatch.setattr(
            "subprocess.Popen",
            lambda *a, **kw: popen_called.append(a),
        )
        _engine._probe_cache.clear()

        from skill_hub.router.route import route
        route(
            "explore how the auth module works",
            session_id="t",
            cwd=str(tmp_path),
            task_id=None,
        )

        assert popen_called == [], "real Popen was called unexpectedly"


# ---------------------------------------------------------------------------
# 2. Route injection — present index (steer + auto-refresh)
# ---------------------------------------------------------------------------

class TestRouteInjectionPresentIndex:
    """route() steers toward indexed queries and queues a refresh."""

    def test_system_message_steers_toward_codegraph(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path, with_codegraph=True)
        _engine._probe_cache.clear()
        _engine._last_dispatch.clear()

        monkeypatch.setattr("skill_hub.config.get", _orch_enabled_config())
        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: None)

        from skill_hub.router.route import route
        result = route(
            "explore how the auth module works",
            session_id="t",
            cwd=str(tmp_path),
            task_id=None,
        )

        sys_msg = result.get("systemMessage", "")
        assert "[tooling]" in sys_msg, f"no [tooling] directive in: {sys_msg!r}"
        # The ready directive steers toward indexed queries.
        assert any(
            kw in sys_msg
            for kw in ("indexed", "code-graph", "prefer", "search")
        ), f"expected steering language in: {sys_msg!r}"

    def test_refresh_action_queued(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path, with_codegraph=True)
        _mark_index_stale(tmp_path)  # only a stale index warrants an auto-sync
        _engine._probe_cache.clear()
        _engine._last_dispatch.clear()

        dispatched: list[list[str]] = []

        # "everywhere" mode authorises the auto-sync (offer mode would only surface it).
        monkeypatch.setattr(
            "skill_hub.config.get",
            _orch_enabled_config(orchestrator_mode="everywhere"),
        )

        def _recording_dispatch(actions: list[list[str]]) -> None:
            dispatched.extend(actions)
            # Do NOT call Popen.

        # route.py imports dispatch_async from skill_hub.orchestrator at call
        # time; patch at that namespace so the route's local binding sees our
        # recorder.
        monkeypatch.setattr(
            "skill_hub.orchestrator.dispatch_async",
            _recording_dispatch,
        )
        # Also patch the engine-level name so ensure_tooling_core / internal
        # evaluate paths don't launch a real subprocess.
        monkeypatch.setattr(
            "skill_hub.orchestrator.engine.dispatch_async",
            _recording_dispatch,
        )
        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: None)

        from skill_hub.router.route import route
        route(
            "explore the codebase",
            session_id="t",
            cwd=str(tmp_path),
            task_id=None,
        )

        # At least one "sync"-style argv must have been recorded.
        sync_actions = [a for a in dispatched if "sync" in a]
        assert sync_actions, (
            f"expected a sync/refresh action; got dispatched={dispatched}"
        )

    def test_refresh_argv_shape(self, tmp_path, monkeypatch):
        """The queued refresh argv must contain 'codegraph' and 'sync'."""
        _make_code_project(tmp_path, with_codegraph=True)
        _mark_index_stale(tmp_path)
        _engine._probe_cache.clear()
        _engine._last_dispatch.clear()
        monkeypatch.setattr(
            "skill_hub.config.get",
            _orch_enabled_config(orchestrator_mode="everywhere"),
        )

        result = evaluate(
            str(tmp_path),
            "explore the codebase",
        )
        # provision_actions from evaluate() directly.
        sync_actions = [a for a in result.provision_actions if "sync" in a]
        assert sync_actions, f"no sync action; got {result.provision_actions}"
        first = sync_actions[0]
        assert any("codegraph" in tok for tok in first), (
            f"argv does not reference codegraph binary: {first}"
        )


# ---------------------------------------------------------------------------
# 3. Disabled switch
# ---------------------------------------------------------------------------

class TestOrchestratorDisabled:
    """When orchestrator_enabled is False, route() must produce no [tooling] directive."""

    def test_disabled_no_tooling_directive(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path)

        monkeypatch.setattr(
            "skill_hub.config.get",
            _config_get_factory({"orchestrator_enabled": False}),
        )

        from skill_hub.router.route import route
        result = route(
            "explore how the auth module works",
            session_id="t",
            cwd=str(tmp_path),
            task_id=None,
        )

        sys_msg = result.get("systemMessage", "")
        assert "[tooling]" not in sys_msg, (
            f"[tooling] found despite disabled switch: {sys_msg!r}"
        )

    def test_evaluate_disabled_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            "skill_hub.config.get",
            lambda k: False if k == "orchestrator_enabled" else None,
        )
        result = evaluate("/tmp", "explore everything")
        assert result.directive == ""
        assert result.decisions == []
        assert result.provision_actions == []


# ---------------------------------------------------------------------------
# 4. Non-code / non-matching — no directive
# ---------------------------------------------------------------------------

class TestNoDirectiveWhenNotApplicable:
    """No [tooling] directive for non-code directories or non-explore messages."""

    def test_non_code_dir_no_directive(self, tmp_path, monkeypatch):
        _make_non_code_dir(tmp_path)
        monkeypatch.setattr("skill_hub.config.get", _orch_enabled_config())
        _engine._probe_cache.clear()

        result = evaluate(str(tmp_path), "explore everything here")
        assert result.directive == "", (
            f"expected empty directive for non-code dir, got: {result.directive!r}"
        )

    def test_non_explore_message_no_directive(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path)
        monkeypatch.setattr("skill_hub.config.get", _orch_enabled_config())
        _engine._probe_cache.clear()

        result = evaluate(str(tmp_path), "write a haiku about Python")
        assert result.directive == "", (
            f"expected empty directive for non-explore message, got: {result.directive!r}"
        )

    def test_route_non_code_no_tooling(self, tmp_path, monkeypatch):
        _make_non_code_dir(tmp_path)
        monkeypatch.setattr("skill_hub.config.get", _orch_enabled_config())
        monkeypatch.setattr(
            "skill_hub.orchestrator.engine.dispatch_async",
            lambda actions: None,
        )
        _engine._probe_cache.clear()

        from skill_hub.router.route import route
        result = route(
            "explore this directory",
            session_id="t",
            cwd=str(tmp_path),
            task_id=None,
        )

        sys_msg = result.get("systemMessage", "")
        assert "[tooling]" not in sys_msg, (
            f"unexpected [tooling] in non-code dir: {sys_msg!r}"
        )


# ---------------------------------------------------------------------------
# 5. ensure_tooling_core — idempotency + non-fatal
# ---------------------------------------------------------------------------

class TestEnsureToolingCore:
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
