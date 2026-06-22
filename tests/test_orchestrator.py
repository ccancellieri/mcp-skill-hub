"""Tests for ``skill_hub.orchestrator``.

Style mirrors ``tests/test_compression.py``: no network, no subprocess, all
happy-path + at least one error/validation case per area.

Monkeypatches:
- ``config.get`` is patched inline where needed to avoid touching the config
  file on disk.
- ``dispatch_async`` / ``subprocess.Popen`` are patched so no real codegraph
  process is launched.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from skill_hub.orchestrator import (
    OrchestratorResult,
    REGISTRY,
    Readiness,
    dispatch_async,
    ensure_tooling_core,
    evaluate,
    is_code_project,
    resolve_targets,
)
from skill_hub.orchestrator import engine as _engine
from skill_hub.orchestrator.registry import _signals_codegraph, probe_codegraph


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_code_project(tmp_path: Path, *, with_codegraph: bool = False) -> Path:
    """Create a minimal code project directory."""
    (tmp_path / ".git").mkdir()
    if with_codegraph:
        cg = tmp_path / ".codegraph"
        cg.mkdir()
        # Set mtime to now so it appears fresh.
        now = time.time()
        import os
        os.utime(cg, (now, now))
    return tmp_path


def _make_non_code_dir(tmp_path: Path) -> Path:
    """A directory with no code-project markers."""
    (tmp_path / "notes.txt").write_text("hello")
    return tmp_path


# ---------------------------------------------------------------------------
# resolve_targets
# ---------------------------------------------------------------------------

class TestResolveTargets:
    def test_includes_cwd(self, tmp_path):
        _make_code_project(tmp_path)
        targets = resolve_targets(str(tmp_path), "hello world")
        assert any(t == tmp_path for t in targets), targets

    def test_extracts_existing_dir_from_message(self, tmp_path):
        proj = tmp_path / "myproject"
        proj.mkdir()
        (proj / ".git").mkdir()
        targets = resolve_targets("/tmp", f"explore {proj}")
        assert any(t == proj for t in targets), targets

    def test_ignores_nonexistent_path_in_message(self, tmp_path):
        targets = resolve_targets(str(tmp_path), "explore /nonexistent/path/xyz")
        # /nonexistent/path/xyz shouldn't appear (it doesn't exist)
        assert all(str(t) != "/nonexistent/path/xyz" for t in targets)

    def test_deduplicates_same_root(self, tmp_path):
        _make_code_project(tmp_path)
        targets = resolve_targets(str(tmp_path), f"check {tmp_path}")
        roots = [str(t) for t in targets]
        assert len(roots) == len(set(roots)), "duplicates found"

    def test_never_raises_on_garbage(self):
        result = resolve_targets("\x00\x01", "<<not a path>> ??? \x00")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# is_code_project
# ---------------------------------------------------------------------------

class TestIsCodeProject:
    def test_git_dir_is_code_project(self, tmp_path):
        (tmp_path / ".git").mkdir()
        assert is_code_project(tmp_path)

    def test_pyproject_is_code_project(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text("")
        assert is_code_project(tmp_path)

    def test_empty_dir_is_not_code_project(self, tmp_path):
        assert not is_code_project(tmp_path)

    def test_nonexistent_dir_is_not_code_project(self, tmp_path):
        assert not is_code_project(tmp_path / "ghost")


# ---------------------------------------------------------------------------
# Registry signal matching
# ---------------------------------------------------------------------------

class TestRegistrySignals:
    def test_code_explore_message_matches_codegraph_on_code_dir(self, tmp_path):
        _make_code_project(tmp_path)
        assert _signals_codegraph("explore the codebase", tmp_path)

    def test_trace_message_matches_codegraph(self, tmp_path):
        _make_code_project(tmp_path)
        assert _signals_codegraph("trace the call graph", tmp_path)

    def test_non_exploration_message_does_not_match(self, tmp_path):
        _make_code_project(tmp_path)
        assert not _signals_codegraph("write a poem about Python", tmp_path)

    def test_explore_on_non_code_dir_does_not_match(self, tmp_path):
        _make_non_code_dir(tmp_path)
        assert not _signals_codegraph("explore this", tmp_path)

    def test_registry_contains_codegraph(self):
        ids = [cap.id for cap in REGISTRY]
        assert "code-graph" in ids


# ---------------------------------------------------------------------------
# probe_codegraph
# ---------------------------------------------------------------------------

class TestProbeCodegraph:
    def test_present_when_codegraph_dir_exists(self, tmp_path):
        _make_code_project(tmp_path, with_codegraph=True)
        r = probe_codegraph(tmp_path)
        assert r.present is True

    def test_absent_when_no_codegraph_dir(self, tmp_path):
        _make_code_project(tmp_path, with_codegraph=False)
        r = probe_codegraph(tmp_path)
        assert r.present is False
        assert r.fresh is False

    def test_fresh_when_recently_updated(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path, with_codegraph=True)
        # Ensure stale_age < sync_ttl
        from skill_hub.orchestrator import engine as eng
        monkeypatch.setattr(
            "skill_hub.config.get",
            lambda k: 300 if k == "orchestrator_sync_ttl_secs" else None,
        )
        r = probe_codegraph(tmp_path)
        # Index was just created so it should be fresh
        assert r.present is True
        assert r.fresh is True

    def test_stale_when_old(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path, with_codegraph=True)
        import os
        # Backdate the .codegraph dir by 1000 seconds
        old_time = time.time() - 1000
        cg = tmp_path / ".codegraph"
        os.utime(cg, (old_time, old_time))
        monkeypatch.setattr(
            "skill_hub.config.get",
            lambda k: 300 if k == "orchestrator_sync_ttl_secs" else None,
        )
        r = probe_codegraph(tmp_path)
        assert r.present is True
        assert r.fresh is False
        assert r.stale_age is not None and r.stale_age > 300

    def test_returns_readiness_type(self, tmp_path):
        r = probe_codegraph(tmp_path)
        assert isinstance(r, Readiness)


# ---------------------------------------------------------------------------
# Autonomy policy
# ---------------------------------------------------------------------------

class TestAutonomyPolicy:
    """Verify the offer-vs-auto-init branching in evaluate()."""

    def _config_get(self, overrides: dict):
        """Build a config.get mock from defaults + overrides."""
        from skill_hub import config as _cfg
        defaults = _cfg._DEFAULTS.copy()
        defaults.update(overrides)
        return lambda k: defaults.get(k)

    def test_missing_index_no_auto_init_emits_offer_directive(
        self, tmp_path, monkeypatch
    ):
        """When the index is absent and auto_init is off, the result should
        contain an offer directive and NO init action."""
        _make_code_project(tmp_path)
        dispatched = []

        monkeypatch.setattr(
            "skill_hub.config.get",
            self._config_get({
                "orchestrator_enabled": True,
                "orchestrator_auto_init": False,
                "orchestrator_auto_init_roots": [],
                "orchestrator_sync_ttl_secs": 300,
                "orchestrator_probe_cache_secs": 60,
            }),
        )
        monkeypatch.setattr(
            "skill_hub.orchestrator.engine.dispatch_async",
            lambda actions: dispatched.extend(actions),
        )

        result = evaluate(str(tmp_path), "explore the code")
        assert isinstance(result, OrchestratorResult)
        # Directive must mention the offer / missing form
        assert result.directive != "", "expected a non-empty directive"
        assert "not indexed" in result.directive or "offer" in result.directive
        # No init argv should be in provision_actions
        init_actions = [a for a in result.provision_actions if "init" in a]
        assert init_actions == [], f"unexpected init action: {init_actions}"

    def test_missing_index_with_root_in_auto_init_roots_queues_init(
        self, tmp_path, monkeypatch
    ):
        """When the target root is in orchestrator_auto_init_roots, an init
        action must be queued even without orchestrator_auto_init=True."""
        _make_code_project(tmp_path)
        dispatched = []

        monkeypatch.setattr(
            "skill_hub.config.get",
            self._config_get({
                "orchestrator_enabled": True,
                "orchestrator_auto_init": False,
                "orchestrator_auto_init_roots": [str(tmp_path)],
                "orchestrator_sync_ttl_secs": 300,
                "orchestrator_probe_cache_secs": 60,
            }),
        )
        monkeypatch.setattr(
            "skill_hub.orchestrator.engine.dispatch_async",
            lambda actions: dispatched.extend(actions),
        )
        # Prevent the probe cache from returning a stale "present" entry.
        _engine._probe_cache.clear()

        result = evaluate(str(tmp_path), "explore the code")
        assert isinstance(result, OrchestratorResult)
        init_actions = [a for a in result.provision_actions if "init" in a]
        assert init_actions, (
            f"expected an init action in provision_actions, got: {result.provision_actions}"
        )

    def test_present_index_always_queues_refresh(self, tmp_path, monkeypatch):
        """When the index is present, a refresh must be queued regardless of
        the auto_init setting."""
        _make_code_project(tmp_path, with_codegraph=True)
        _engine._probe_cache.clear()

        monkeypatch.setattr(
            "skill_hub.config.get",
            self._config_get({
                "orchestrator_enabled": True,
                "orchestrator_auto_init": False,
                "orchestrator_auto_init_roots": [],
                "orchestrator_sync_ttl_secs": 300,
                "orchestrator_probe_cache_secs": 60,
            }),
        )
        # Patch subprocess.Popen so nothing actually runs.
        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: None)
        # Reset debounce so the refresh isn't skipped.
        _engine._last_dispatch.clear()

        result = evaluate(str(tmp_path), "explore the code")
        refresh_actions = [a for a in result.provision_actions if "sync" in a]
        assert refresh_actions, (
            f"expected a refresh action in provision_actions, got: {result.provision_actions}"
        )


# ---------------------------------------------------------------------------
# evaluate — error resilience
# ---------------------------------------------------------------------------

class TestEvaluateErrorResilience:
    def test_never_raises_on_garbage_cwd(self):
        result = evaluate("\x00not a path\x00", "some message")
        assert isinstance(result, OrchestratorResult)

    def test_never_raises_on_empty_message(self, tmp_path):
        result = evaluate(str(tmp_path), "")
        assert isinstance(result, OrchestratorResult)

    def test_returns_orchestrator_result_type_always(self):
        result = evaluate("", "")
        assert isinstance(result, OrchestratorResult)
        assert isinstance(result.directive, str)
        assert isinstance(result.decisions, list)
        assert isinstance(result.provision_actions, list)

    def test_disabled_orchestrator_returns_empty(self, monkeypatch):
        monkeypatch.setattr(
            "skill_hub.config.get",
            lambda k: False if k == "orchestrator_enabled" else None,
        )
        result = evaluate("/tmp", "explore everything")
        assert result.directive == ""
        assert result.decisions == []
        assert result.provision_actions == []


# ---------------------------------------------------------------------------
# dispatch_async — debounce + non-fatal
# ---------------------------------------------------------------------------

class TestDispatchAsync:
    def test_empty_actions_noop(self):
        # Must not raise.
        dispatch_async([])

    def test_popen_failure_does_not_raise(self, monkeypatch):
        def _bad_popen(*a, **kw):
            raise OSError("no such file")
        monkeypatch.setattr("subprocess.Popen", _bad_popen)
        _engine._last_dispatch.clear()
        # Should not raise.
        dispatch_async([["codegraph", "sync", "/tmp"]])

    def test_debounce_skips_recent_dispatch(self, monkeypatch):
        launched = []

        class _FakePopen:
            def __init__(self, argv, **kw):
                launched.append(argv)

        monkeypatch.setattr("subprocess.Popen", _FakePopen)
        monkeypatch.setattr(
            "skill_hub.config.get",
            lambda k: 300 if k == "orchestrator_sync_ttl_secs" else None,
        )
        _engine._last_dispatch.clear()
        argv = ["codegraph", "sync", "/tmp/proj"]
        dispatch_async([argv])
        count_after_first = len(launched)
        # Second call within TTL window should be debounced.
        dispatch_async([argv])
        assert len(launched) == count_after_first, "debounce did not skip second dispatch"


# ---------------------------------------------------------------------------
# ensure_tooling_core
# ---------------------------------------------------------------------------

class TestEnsureToolingCore:
    def test_absent_index_no_init_returns_present_false(self, tmp_path):
        _make_code_project(tmp_path)
        result = ensure_tooling_core(str(tmp_path), init=False, refresh=False)
        assert result["present"] is False
        assert result["action"] in ("none", "error")

    def test_present_index_refresh_dispatched(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path, with_codegraph=True)
        _engine._last_dispatch.clear()
        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: None)
        result = ensure_tooling_core(str(tmp_path), init=False, refresh=True)
        assert result["present"] is True
        assert result["action"] == "refresh_dispatched"

    def test_returns_dict_with_required_keys(self, tmp_path):
        result = ensure_tooling_core(str(tmp_path))
        for key in ("path", "present", "fresh", "action", "directive"):
            assert key in result, f"missing key: {key}"

    def test_never_raises_on_bad_path(self):
        result = ensure_tooling_core("\x00/bad/path\x00", init=False, refresh=False)
        assert isinstance(result, dict)
        assert result["action"] == "error" or isinstance(result["present"], bool)
