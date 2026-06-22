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

def _write_codegraph_db(cg: Path, *, node_count: int) -> None:
    """Create a codegraph.db with a ``nodes`` table holding *node_count* rows.

    Mirrors the real index's relevant shape closely enough for the probe's
    cheap ``SELECT count(*) FROM nodes`` readiness check.
    """
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


def _make_code_project(
    tmp_path: Path, *, with_codegraph: bool = False, node_count: int = 5
) -> Path:
    """Create a minimal code project directory.

    When *with_codegraph* is set, the ``.codegraph/`` index is populated with a
    ``codegraph.db`` holding *node_count* nodes — a non-empty (usable) index by
    default, matching what ``codegraph init -i`` produces. Pass ``node_count=0``
    to simulate the present-but-never-indexed state a bare ``codegraph init``
    leaves behind.
    """
    (tmp_path / ".git").mkdir()
    if with_codegraph:
        cg = tmp_path / ".codegraph"
        cg.mkdir()
        _write_codegraph_db(cg, node_count=node_count)
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
# Hardened binary resolution (PATH-hijack defense)
# ---------------------------------------------------------------------------

class TestCodegraphBinResolution:
    def test_resolves_only_within_trusted_dirs(self, monkeypatch):
        from skill_hub.orchestrator import registry as _reg

        captured: dict[str, str | None] = {}

        def _fake_which(name, path=None):
            captured["name"] = name
            captured["path"] = path
            return "/opt/homebrew/bin/codegraph"

        monkeypatch.setattr(_reg.shutil, "which", _fake_which)
        assert _reg._resolve_codegraph_bin() == "/opt/homebrew/bin/codegraph"
        # The PATH passed to which() must be the fixed trusted set, never the
        # inherited environment PATH (which an attacker could poison).
        assert captured["name"] == "codegraph"
        for trusted in _reg._CODEGRAPH_TRUSTED_DIRS:
            assert trusted in captured["path"]

    def test_returns_none_when_not_installed(self, monkeypatch):
        from skill_hub.orchestrator import registry as _reg

        monkeypatch.setattr(_reg.shutil, "which", lambda name, path=None: None)
        assert _reg._resolve_codegraph_bin() is None

    def test_provision_argv_none_when_tool_missing(self, monkeypatch, tmp_path):
        from skill_hub.orchestrator import registry as _reg

        monkeypatch.setattr(_reg.shutil, "which", lambda name, path=None: None)
        assert _reg._codegraph_refresh_argv(tmp_path) is None
        assert _reg._codegraph_init_argv(tmp_path) is None

    def test_provision_argv_present_when_tool_found(self, monkeypatch, tmp_path):
        from skill_hub.orchestrator import registry as _reg

        monkeypatch.setattr(
            _reg.shutil, "which", lambda name, path=None: "/usr/local/bin/codegraph"
        )
        assert _reg._codegraph_refresh_argv(tmp_path) == [
            "/usr/local/bin/codegraph", "sync", str(tmp_path),
        ]
        assert _reg._codegraph_init_argv(tmp_path) == [
            "/usr/local/bin/codegraph", "init", "-i", str(tmp_path),
        ]


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
        # Freshness is measured from the db build time, so backdate the db file
        # (not the directory) by 1000s past the 300s TTL.
        old_time = time.time() - 1000
        db = tmp_path / ".codegraph" / "codegraph.db"
        os.utime(db, (old_time, old_time))
        monkeypatch.setattr(
            "skill_hub.config.get",
            lambda k: 300 if k == "orchestrator_sync_ttl_secs" else None,
        )
        r = probe_codegraph(tmp_path)
        assert r.present is True
        assert r.fresh is False
        assert r.stale_age is not None and r.stale_age > 300

    def test_stale_when_dirty_after_index(self, tmp_path, monkeypatch):
        # A .dirty marker newer than the db means source changed since the last
        # index build — stale regardless of how recent the build was.
        _make_code_project(tmp_path, with_codegraph=True)
        import os
        cg = tmp_path / ".codegraph"
        now = time.time()
        # Index built 5s ago (well within any TTL)...
        os.utime(cg / "codegraph.db", (now - 5, now - 5))
        # ...but a file was edited 1s ago, after that build.
        (cg / ".dirty").write_text(str(int((now - 1) * 1000)))
        monkeypatch.setattr(
            "skill_hub.config.get",
            lambda k: 300 if k == "orchestrator_sync_ttl_secs" else None,
        )
        r = probe_codegraph(tmp_path)
        assert r.present is True
        assert r.fresh is False
        assert r.pending_edits is not None and r.pending_edits > 0
        assert "needs sync" in r.detail

    def test_fresh_when_dirty_before_index(self, tmp_path, monkeypatch):
        # A .dirty marker OLDER than the db means the index already absorbed the
        # last edit — fresh.
        _make_code_project(tmp_path, with_codegraph=True)
        import os
        cg = tmp_path / ".codegraph"
        now = time.time()
        os.utime(cg / "codegraph.db", (now, now))
        (cg / ".dirty").write_text(str(int((now - 100) * 1000)))
        monkeypatch.setattr(
            "skill_hub.config.get",
            lambda k: 300 if k == "orchestrator_sync_ttl_secs" else None,
        )
        r = probe_codegraph(tmp_path)
        assert r.present is True
        assert r.fresh is True
        assert r.pending_edits == 0.0

    def test_returns_readiness_type(self, tmp_path):
        r = probe_codegraph(tmp_path)
        assert isinstance(r, Readiness)

    def test_empty_index_reported_not_present(self, tmp_path):
        # A present-but-never-indexed .codegraph (bare `init`, 0 nodes) must be
        # reported as not-present so the orchestrator offers to (re-)index
        # rather than steering Claude to an empty graph.
        _make_code_project(tmp_path, with_codegraph=True, node_count=0)
        r = probe_codegraph(tmp_path)
        assert r.present is False
        assert r.fresh is False
        assert "0 nodes" in r.detail

    def test_missing_db_reported_not_present(self, tmp_path):
        # .codegraph/ dir without a codegraph.db is an unusable scaffold.
        (tmp_path / ".git").mkdir()
        (tmp_path / ".codegraph").mkdir()
        r = probe_codegraph(tmp_path)
        assert r.present is False

    def test_populated_index_reports_node_count(self, tmp_path):
        _make_code_project(tmp_path, with_codegraph=True, node_count=7)
        r = probe_codegraph(tmp_path)
        assert r.present is True
        assert "7 nodes" in r.detail

    def test_unreadable_db_preserves_legacy_presence(self, tmp_path):
        # Unexpected schema → count is unknown (None) → do NOT downgrade; a future
        # codegraph schema change must never make the probe nag.
        (tmp_path / ".git").mkdir()
        cg = tmp_path / ".codegraph"
        cg.mkdir()
        import sqlite3
        con = sqlite3.connect(cg / "codegraph.db")
        try:
            con.execute("CREATE TABLE not_nodes (id INTEGER)")
            con.commit()
        finally:
            con.close()
        r = probe_codegraph(tmp_path)
        assert r.present is True


class TestCodegraphNodeCount:
    def test_missing_db_is_zero(self, tmp_path):
        from skill_hub.orchestrator import engine as eng
        (tmp_path / ".codegraph").mkdir()
        assert eng._codegraph_node_count(tmp_path / ".codegraph") == 0

    def test_counts_populated_nodes(self, tmp_path):
        from skill_hub.orchestrator import engine as eng
        cg = tmp_path / ".codegraph"
        cg.mkdir()
        _write_codegraph_db(cg, node_count=4)
        assert eng._codegraph_node_count(cg) == 4

    def test_unexpected_schema_is_none(self, tmp_path):
        from skill_hub.orchestrator import engine as eng
        import sqlite3
        cg = tmp_path / ".codegraph"
        cg.mkdir()
        con = sqlite3.connect(cg / "codegraph.db")
        try:
            con.execute("CREATE TABLE other (id INTEGER)")
            con.commit()
        finally:
            con.close()
        assert eng._codegraph_node_count(cg) is None


class TestWorktreeMismatch:
    """A linked worktree borrowing an ancestor checkout's index is the silent
    wrong-branch trap — the probe must flag it distinctly."""

    def _make_worktree_under_indexed_main(self, tmp_path):
        # Main checkout: .git DIR + populated .codegraph/.
        main = tmp_path / "main"
        main.mkdir()
        (main / ".git").mkdir()
        cg = main / ".codegraph"
        cg.mkdir()
        _write_codegraph_db(cg, node_count=5)
        # Linked worktree nested under main: .git FILE, no .codegraph of its own.
        wt = main / ".worktrees" / "feature"
        wt.mkdir(parents=True)
        (wt / ".git").write_text("gitdir: /somewhere/.git/worktrees/feature\n")
        return main, wt

    def test_worktree_with_ancestor_index_is_flagged(self, tmp_path):
        main, wt = self._make_worktree_under_indexed_main(tmp_path)
        r = probe_codegraph(wt)
        assert r.present is False
        assert r.worktree_mismatch is True
        assert r.ancestor_index == str(main / ".codegraph")
        assert "worktree" in r.detail

    def test_main_checkout_is_not_flagged_as_worktree(self, tmp_path):
        main, _ = self._make_worktree_under_indexed_main(tmp_path)
        r = probe_codegraph(main)
        assert r.present is True
        assert r.worktree_mismatch is False

    def test_plain_missing_index_is_not_worktree_mismatch(self, tmp_path):
        # A normal project (no .git FILE, no ancestor index) → generic missing.
        _make_code_project(tmp_path, with_codegraph=False)
        r = probe_codegraph(tmp_path)
        assert r.present is False
        assert r.worktree_mismatch is False

    def test_worktree_directive_warns_and_offers_local_index(self, tmp_path):
        from skill_hub.orchestrator.registry import CODEGRAPH
        _, wt = self._make_worktree_under_indexed_main(tmp_path)
        r = probe_codegraph(wt)
        directive = CODEGRAPH.format_directive_missing(wt, r)
        assert "worktree" in directive
        assert "init -i" in directive
        assert "offer" in directive.lower()


# ---------------------------------------------------------------------------
# Mode resolution
# ---------------------------------------------------------------------------

def _fake_get(overrides: dict):
    """A config.get-style accessor over defaults + overrides."""
    from skill_hub import config as _cfg
    merged = _cfg._DEFAULTS.copy()
    merged.update(overrides)
    return lambda k: merged.get(k)


class TestResolveMode:
    def test_explicit_mode_wins(self):
        from skill_hub.orchestrator.engine import resolve_mode
        for m in ("off", "offer", "auto", "everywhere"):
            assert resolve_mode(_fake_get({"orchestrator_mode": m})) == m

    def test_explicit_mode_case_insensitive(self):
        from skill_hub.orchestrator.engine import resolve_mode
        assert resolve_mode(_fake_get({"orchestrator_mode": "AUTO"})) == "auto"

    def test_invalid_mode_falls_back_to_derivation(self):
        from skill_hub.orchestrator.engine import resolve_mode
        # garbage mode + plain enabled config -> derived "offer"
        assert resolve_mode(_fake_get({
            "orchestrator_mode": "banana",
            "orchestrator_enabled": True,
            "orchestrator_auto_init": False,
            "orchestrator_auto_init_roots": [],
        })) == "offer"

    def test_derive_off_when_disabled(self):
        from skill_hub.orchestrator.engine import resolve_mode
        assert resolve_mode(_fake_get({
            "orchestrator_mode": None, "orchestrator_enabled": False,
        })) == "off"

    def test_derive_everywhere_from_legacy_auto_init(self):
        from skill_hub.orchestrator.engine import resolve_mode
        assert resolve_mode(_fake_get({
            "orchestrator_mode": None, "orchestrator_enabled": True,
            "orchestrator_auto_init": True,
        })) == "everywhere"

    def test_derive_auto_from_legacy_roots(self):
        from skill_hub.orchestrator.engine import resolve_mode
        assert resolve_mode(_fake_get({
            "orchestrator_mode": None, "orchestrator_enabled": True,
            "orchestrator_auto_init": False,
            "orchestrator_auto_init_roots": ["/some/root"],
        })) == "auto"

    def test_derive_offer_default(self):
        from skill_hub.orchestrator.engine import resolve_mode
        assert resolve_mode(_fake_get({
            "orchestrator_mode": None, "orchestrator_enabled": True,
            "orchestrator_auto_init": False, "orchestrator_auto_init_roots": [],
        })) == "offer"


class TestRootUnderParents:
    def test_exact_match(self, tmp_path):
        from skill_hub.orchestrator.engine import _root_under_parents
        assert _root_under_parents(tmp_path, {tmp_path}) is True

    def test_nested_child(self, tmp_path):
        from skill_hub.orchestrator.engine import _root_under_parents
        child = tmp_path / "a" / "b"
        assert _root_under_parents(child, {tmp_path}) is True

    def test_sibling_prefix_does_not_match(self):
        # /work/code must NOT match /work/codex — boundary guard.
        from skill_hub.orchestrator.engine import _root_under_parents
        assert _root_under_parents(Path("/work/codex"), {Path("/work/code")}) is False

    def test_unrelated_does_not_match(self):
        from skill_hub.orchestrator.engine import _root_under_parents
        assert _root_under_parents(Path("/elsewhere"), {Path("/work/code")}) is False

    def test_empty_parents_never_matches(self, tmp_path):
        from skill_hub.orchestrator.engine import _root_under_parents
        assert _root_under_parents(tmp_path, set()) is False


class TestEvaluateModes:
    """End-to-end mode behaviour through evaluate()."""

    def _setup(self, monkeypatch, overrides):
        monkeypatch.setattr("skill_hub.config.get", _fake_get(overrides))
        monkeypatch.setattr(
            "skill_hub.orchestrator.engine.dispatch_async", lambda actions: None
        )
        _engine._probe_cache.clear()

    def test_off_mode_returns_empty(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path)
        self._setup(monkeypatch, {"orchestrator_mode": "off"})
        result = evaluate(str(tmp_path), "explore the code")
        assert result.directive == ""
        assert result.provision_actions == []
        assert result.decisions == []

    def test_offer_mode_offers_no_init(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path)
        self._setup(monkeypatch, {"orchestrator_mode": "offer"})
        result = evaluate(str(tmp_path), "explore the code")
        assert result.directive != ""
        assert [a for a in result.provision_actions if "init" in a] == []

    def test_everywhere_mode_queues_init(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path)
        self._setup(monkeypatch, {"orchestrator_mode": "everywhere"})
        result = evaluate(str(tmp_path), "explore the code")
        assert [a for a in result.provision_actions if "init" in a], (
            f"expected init action, got {result.provision_actions}"
        )

    def test_auto_mode_inits_when_under_parent(self, tmp_path, monkeypatch):
        proj = tmp_path / "repo"
        proj.mkdir()
        _make_code_project(proj)
        self._setup(monkeypatch, {
            "orchestrator_mode": "auto",
            "orchestrator_auto_init_roots": [str(tmp_path)],  # parent of proj
        })
        result = evaluate(str(proj), "explore the code")
        assert [a for a in result.provision_actions if "init" in a], (
            "project under an auto-init parent should queue init"
        )

    def test_auto_mode_offers_when_outside_parent(self, tmp_path, monkeypatch):
        _make_code_project(tmp_path)
        self._setup(monkeypatch, {
            "orchestrator_mode": "auto",
            "orchestrator_auto_init_roots": ["/some/other/place"],
        })
        result = evaluate(str(tmp_path), "explore the code")
        assert [a for a in result.provision_actions if "init" in a] == [], (
            "project outside every auto-init parent must only offer"
        )
        assert result.directive != ""


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

    def _mark_stale(self, tmp_path):
        """Make a present index read as stale (source edited after the build)."""
        import os
        cg = tmp_path / ".codegraph"
        now = time.time()
        os.utime(cg / "codegraph.db", (now - 30, now - 30))
        (cg / ".dirty").write_text(str(int(now * 1000)))

    def test_present_fresh_index_queues_no_refresh(self, tmp_path, monkeypatch):
        """A fresh, trustworthy index needs no sync — no action queued."""
        _make_code_project(tmp_path, with_codegraph=True)
        _engine._probe_cache.clear()
        _engine._last_dispatch.clear()
        monkeypatch.setattr(
            "skill_hub.config.get",
            self._config_get({"orchestrator_mode": "everywhere",
                              "orchestrator_sync_ttl_secs": 300}),
        )
        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: None)

        result = evaluate(str(tmp_path), "explore the code")
        assert [a for a in result.provision_actions if "sync" in a] == [], (
            f"fresh index should not queue a sync, got: {result.provision_actions}"
        )

    def test_stale_index_auto_syncs_in_everywhere_mode(self, tmp_path, monkeypatch):
        """A stale index auto-syncs when the mode permits it."""
        _make_code_project(tmp_path, with_codegraph=True)
        self._mark_stale(tmp_path)
        _engine._probe_cache.clear()
        _engine._last_dispatch.clear()
        monkeypatch.setattr(
            "skill_hub.config.get",
            self._config_get({"orchestrator_mode": "everywhere",
                              "orchestrator_sync_ttl_secs": 300}),
        )
        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: None)

        result = evaluate(str(tmp_path), "explore the code")
        assert [a for a in result.provision_actions if "sync" in a], (
            f"stale index in everywhere mode must queue a sync, got: {result.provision_actions}"
        )

    def test_stale_index_offers_only_in_offer_mode(self, tmp_path, monkeypatch):
        """In offer mode a stale index surfaces a directive but auto-runs nothing."""
        _make_code_project(tmp_path, with_codegraph=True)
        self._mark_stale(tmp_path)
        _engine._probe_cache.clear()
        _engine._last_dispatch.clear()
        monkeypatch.setattr(
            "skill_hub.config.get",
            self._config_get({"orchestrator_mode": "offer",
                              "orchestrator_sync_ttl_secs": 300}),
        )
        monkeypatch.setattr("subprocess.Popen", lambda *a, **kw: None)

        result = evaluate(str(tmp_path), "explore the code")
        assert [a for a in result.provision_actions if "sync" in a] == [], (
            "offer mode must not auto-run sync"
        )
        assert "STALE" in result.directive and "sync" in result.directive


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
