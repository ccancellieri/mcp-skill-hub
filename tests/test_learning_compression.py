"""Tests for the learning-compression fixes.

FIX 1: preloader surfaces matching teachings and merges teaching-suggested skills.
FIX 2: postcompact path runs at LOW pressure; background path still requires IDLE.
FIX 3: postcompact_optimize_apply default is True.
FIX 4: session-end promotion calls the promote path, gated by flag, skips under HIGH.

Isolation: CONFIG_PATH is always redirected to a tmp file.
           SkillStore is opened against a tmp DB.
           skill_hub.server is NEVER imported (module-level live-DB side-effect).
"""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


# ---------------------------------------------------------------------------
# Config isolation — applied to every test in this module
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolated_config(tmp_path, monkeypatch):
    """Point CONFIG_PATH at a tmp file so no real config is read or written."""
    from skill_hub import config as cfg

    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")
    cfg.reset_vault_migration_flag()
    yield


# ---------------------------------------------------------------------------
# Shared store fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_store(tmp_path, monkeypatch):
    """Return a SkillStore backed by a fresh tmp DB; patch DB_PATH to match."""
    from skill_hub.store import SkillStore
    import skill_hub.store as store_mod

    db_path = tmp_path / "skill_hub.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db_path)
    store = SkillStore(db_path=db_path)
    yield store
    store.close()


# ===========================================================================
# FIX 1 — preloader consults teachings and merges teaching-suggested skills
# ===========================================================================

class TestPreloaderTeachings:
    """load_skills() should return a 3-tuple and surface matching teachings."""

    def test_returns_three_tuple(self, monkeypatch):
        """load_skills always returns (skill_names, plugin_names, teaching_text)."""
        import skill_hub.router.preloader as preloader

        # No embeddings available → graceful empty return. Patch the embeddings
        # source module: load_skills does `from ..embeddings import
        # embed_available` fresh each call, so patching the preloader namespace
        # is a no-op (and would only "pass" when no embed backend is installed).
        import skill_hub.embeddings as _emb
        monkeypatch.setattr(_emb, "embed_available", lambda: False)
        result = preloader.load_skills(["python", "fastapi"], cfg={}, top_k=3)
        assert isinstance(result, tuple)
        assert len(result) == 3
        skill_names, plugin_names, teaching_text = result
        assert skill_names == []
        assert plugin_names == []
        assert teaching_text == ""

    def test_teaching_rules_merged_into_result(self, tmp_store, monkeypatch):
        """When a teaching matches the query, its rule text is returned and its
        target skill is merged into the preloaded skills list."""
        import skill_hub.router.preloader as preloader
        from skill_hub.store import SkillStore

        # Seed a teaching with a fake vector that will always match
        fake_vec = [1.0] + [0.0] * 767
        teaching_id = tmp_store.add_teaching(
            rule="when working on API endpoints use fastapi-engineer",
            rule_vector=fake_vec,
            action="suggest",
            target_type="skill",
            target_id="fastapi-engineer",
            weight=1.0,
        )
        assert teaching_id > 0

        # Patch embed_available and embed at the embeddings module level —
        # preloader imports them fresh via `from ..embeddings import ...` each
        # call, so we must patch the source module, not the preloader namespace.
        import skill_hub.embeddings as _emb
        monkeypatch.setattr(_emb, "embed_available", lambda: True)
        monkeypatch.setattr(_emb, "embed", lambda q: fake_vec)

        # Patch SkillStore at the store module level — preloader imports it
        # via `from ..store import SkillStore` so we patch the source.
        import skill_hub.store as _store_mod
        monkeypatch.setattr(_store_mod, "SkillStore", lambda: tmp_store)
        # Prevent double-close
        monkeypatch.setattr(tmp_store, "close", lambda: None)

        # search() returns empty (no skills indexed) so only teachings contribute
        monkeypatch.setattr(tmp_store, "search", lambda vec, top_k=9: [])
        monkeypatch.setattr(tmp_store, "suggest_plugins", lambda vec: [])

        cfg = {"router_use_teachings": True, "teaching_min_similarity": 0.1}
        skill_names, plugin_names, teaching_text = preloader.load_skills(
            ["api", "endpoint"], cfg=cfg, top_k=3
        )

        assert "fastapi-engineer" in skill_names, (
            f"teaching-suggested skill not merged: skill_names={skill_names}"
        )
        assert "when working on API endpoints" in teaching_text

    def test_teaching_disabled_via_config(self, tmp_store, monkeypatch):
        """router_use_teachings=False skips the teaching path entirely."""
        import skill_hub.router.preloader as preloader
        import skill_hub.embeddings as _emb
        import skill_hub.store as _store_mod

        fake_vec = [1.0] + [0.0] * 767
        tmp_store.add_teaching(
            rule="never reached",
            rule_vector=fake_vec,
            action="suggest",
            target_type="skill",
            target_id="should-not-appear",
            weight=1.0,
        )

        monkeypatch.setattr(_emb, "embed_available", lambda: True)
        monkeypatch.setattr(_emb, "embed", lambda q: fake_vec)
        monkeypatch.setattr(_store_mod, "SkillStore", lambda: tmp_store)
        monkeypatch.setattr(tmp_store, "close", lambda: None)
        monkeypatch.setattr(tmp_store, "search", lambda vec, top_k=9: [])
        monkeypatch.setattr(tmp_store, "suggest_plugins", lambda vec: [])

        cfg = {"router_use_teachings": False}
        skill_names, plugin_names, teaching_text = preloader.load_skills(
            ["api"], cfg=cfg, top_k=3
        )

        assert "should-not-appear" not in skill_names
        assert teaching_text == ""

    def test_teaching_text_capped_top_k_and_length(self, tmp_store, monkeypatch):
        """At most _TEACHING_TOP_K teachings are injected and teaching_text is
        bounded to _TEACHING_TEXT_MAX_CHARS."""
        import skill_hub.router.preloader as preloader
        import skill_hub.embeddings as _emb
        import skill_hub.store as _store_mod

        fake_vec = [1.0] + [0.0] * 767
        # Seed 10 long teachings — well above the top_k cap; each rule long
        # enough that 3 of them already approach the char budget.
        for i in range(10):
            tmp_store.add_teaching(
                rule=f"teaching rule number {i} " + ("x" * 300),
                rule_vector=fake_vec,
                action="note",
                target_type="",
                target_id="",
                weight=1.0,
            )

        monkeypatch.setattr(_emb, "embed_available", lambda: True)
        monkeypatch.setattr(_emb, "embed", lambda q: fake_vec)
        monkeypatch.setattr(_store_mod, "SkillStore", lambda: tmp_store)
        monkeypatch.setattr(tmp_store, "close", lambda: None)
        monkeypatch.setattr(tmp_store, "search", lambda vec, top_k=9: [])
        monkeypatch.setattr(tmp_store, "suggest_plugins", lambda vec: [])

        # Spy on search_teachings to confirm the preloader passes top_k=3
        seen_kwargs: dict = {}
        real_search = tmp_store.search_teachings

        def _spy(vec, min_sim=0.6, top_k=None):
            seen_kwargs["top_k"] = top_k
            return real_search(vec, min_sim=min_sim, top_k=top_k)

        monkeypatch.setattr(tmp_store, "search_teachings", _spy)

        cfg = {"router_use_teachings": True, "teaching_min_similarity": 0.1}
        _, _, teaching_text = preloader.load_skills(
            ["api"], cfg=cfg, top_k=3
        )

        assert seen_kwargs.get("top_k") == preloader._TEACHING_TOP_K, (
            f"preloader should cap teachings to top_k=3, passed {seen_kwargs.get('top_k')}"
        )
        # Length is bounded (cap + the single-char ellipsis).
        assert len(teaching_text) <= preloader._TEACHING_TEXT_MAX_CHARS + 1, (
            f"teaching_text not capped: len={len(teaching_text)}"
        )
        # At most 3 rules' worth of newline-joined text (≤ 2 separators).
        assert teaching_text.count("\n") <= preloader._TEACHING_TOP_K - 1


# ===========================================================================
# FIX 2 — postcompact runs at LOW; background optimize_memory gate unchanged
# ===========================================================================

class TestPostcompactPressureGate:
    """should_run_postcompact_optimize uses a configurable ceiling (default LOW).
       should_run_llm("optimize_memory") still requires IDLE."""

    def _make_snapshot(self, pressure_level):
        """Return a SystemSnapshot with the given Pressure."""
        from skill_hub.resource_monitor import Pressure, SystemSnapshot
        import time
        return SystemSnapshot(
            cpu_load_1m=0.1,
            memory_used_pct=0.1,
            memory_available_mb=4096,
            total_memory_mb=8192,
            pressure=pressure_level,
            timestamp=time.monotonic(),
        )

    def test_postcompact_runs_at_low_pressure(self, monkeypatch):
        from skill_hub import resource_monitor as rm
        from skill_hub.resource_monitor import Pressure
        from skill_hub import config as cfg_mod

        monkeypatch.setattr(cfg_mod, "CONFIG_PATH",
                            Path("/tmp/nonexistent-config-test.json"))
        # resource_gating_enabled defaults to True; patch it to ensure gating is on
        monkeypatch.setattr(rm, "snapshot",
                            lambda force=False: self._make_snapshot(Pressure.LOW))

        result = rm.should_run_postcompact_optimize()
        assert result is True, "postcompact should run at LOW pressure"

    def test_postcompact_blocked_at_moderate_pressure(self, monkeypatch):
        from skill_hub import resource_monitor as rm
        from skill_hub.resource_monitor import Pressure
        from skill_hub import config as cfg_mod

        monkeypatch.setattr(rm, "snapshot",
                            lambda force=False: self._make_snapshot(Pressure.MODERATE))

        result = rm.should_run_postcompact_optimize()
        assert result is False, "postcompact should be blocked at MODERATE (above LOW ceiling)"

    def test_background_optimize_memory_still_requires_idle(self, monkeypatch):
        """The existing optimize_memory entry in should_run_llm must stay at IDLE."""
        from skill_hub import resource_monitor as rm
        from skill_hub.resource_monitor import Pressure

        # LOW pressure → should_run_llm("optimize_memory") must return False
        monkeypatch.setattr(rm, "snapshot",
                            lambda force=False: self._make_snapshot(Pressure.LOW))

        result = rm.should_run_llm("optimize_memory")
        assert result is False, (
            "Background optimize_memory must still require IDLE pressure; "
            "only the postcompact path is relaxed"
        )

    def test_postcompact_ceiling_configurable(self, monkeypatch, tmp_path):
        """postcompact_pressure_max=IDLE makes the postcompact path as strict as IDLE."""
        from skill_hub import resource_monitor as rm
        from skill_hub.resource_monitor import Pressure
        from skill_hub import config as cfg_mod

        # Write a config that tightens the ceiling to IDLE
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(json.dumps({"postcompact_pressure_max": "IDLE"}))
        monkeypatch.setattr(cfg_mod, "CONFIG_PATH", cfg_path)
        cfg_mod.reset_vault_migration_flag()

        # LOW pressure (above IDLE) should be blocked when ceiling=IDLE
        monkeypatch.setattr(rm, "snapshot",
                            lambda force=False: self._make_snapshot(Pressure.LOW))

        result = rm.should_run_postcompact_optimize()
        assert result is False, (
            "With postcompact_pressure_max=IDLE, LOW pressure should be blocked"
        )


# ===========================================================================
# FIX 3 — postcompact_optimize_apply default is True
# ===========================================================================

class TestPostcompactDefaultApply:

    def test_postcompact_optimize_apply_default_true(self):
        """Config default for postcompact_optimize_apply must be True."""
        from skill_hub import config as cfg_mod

        # Call get() without setting the key — should return the default True
        val = cfg_mod.get("postcompact_optimize_apply")
        assert val is True, (
            f"postcompact_optimize_apply default should be True, got {val!r}"
        )

    def test_postcompact_cli_uses_apply_true_by_default(self, monkeypatch):
        """When config default is used, optimize_memory is called with dry_run=False."""
        import types
        from skill_hub import cli
        from skill_hub import resource_monitor as rm

        # Let pressure gate pass
        monkeypatch.setattr(rm, "should_run_postcompact_optimize", lambda: True)

        calls: list[dict] = []
        fake_server = types.ModuleType("skill_hub.server")
        fake_server.optimize_memory = (  # type: ignore[attr-defined]
            lambda dry_run=True, bypass_gate=False:
            calls.append({"dry_run": dry_run, "bypass_gate": bypass_gate}) or "REPORT"
        )
        monkeypatch.setitem(sys.modules, "skill_hub.server", fake_server)

        cli._cmd_postcompact_optimize(session_id="s-test")
        # With default (True), dry_run should be False
        assert calls == [{"dry_run": False, "bypass_gate": True}], (
            f"Expected dry_run=False + bypass_gate=True, got calls={calls}"
        )

    def test_postcompact_actually_runs_at_low_not_skipped(self, monkeypatch):
        """End-to-end: at LOW pressure the postcompact path RUNS optimize_memory
        (bypassing the inner IDLE gate) instead of returning the 'skipped' string.

        Guards against the no-op where the widened LOW ceiling clears the outer
        gate but optimize_memory's internal should_run_llm IDLE check re-blocks it.
        """
        import types
        from skill_hub import cli
        from skill_hub import resource_monitor as rm
        from skill_hub.resource_monitor import Pressure, SystemSnapshot

        # System is at LOW pressure — above IDLE, so the inner gate WOULD block.
        low = SystemSnapshot(
            cpu_load_1m=0.1, memory_used_pct=0.1,
            memory_available_mb=4096, total_memory_mb=8192,
            pressure=Pressure.LOW, timestamp=0.0,
        )
        monkeypatch.setattr(rm, "snapshot", lambda force=False: low)

        # Fake optimize_memory mirroring the REAL inner gate: when bypass_gate is
        # False it consults should_run_llm("optimize_memory") (IDLE-only) and at
        # LOW returns the skipped string; when bypass_gate is True it runs.
        def fake_optimize(dry_run=True, bypass_gate=False):
            if not bypass_gate and not rm.should_run_llm("optimize_memory"):
                return "Skipped: system under pressure (LOW, ...)."
            return "=== Memory Optimization Report ===\nran"

        fake_server = types.ModuleType("skill_hub.server")
        fake_server.optimize_memory = fake_optimize  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "skill_hub.server", fake_server)

        result = cli._cmd_postcompact_optimize(session_id="s-low")

        msg = result.get("systemMessage", "")
        assert "Skipped: system under pressure" not in msg, (
            "postcompact was re-blocked by the inner IDLE gate at LOW — "
            "bypass_gate must let it run"
        )
        assert "Memory Optimization Report" in msg, (
            f"postcompact did not run optimize_memory at LOW; got: {msg!r}"
        )


# ===========================================================================
# FIX 4 — session-end L0→L1 promotion
# ===========================================================================

class TestSessionEndPromotion:

    def _seed_l0_vector(self, store, doc_id="doc:test", access_count=1):
        """Insert an L0 vector row directly, bypassing embedding."""
        store._conn.execute(
            "INSERT OR REPLACE INTO vectors "
            "(namespace, doc_id, model, vector, norm, metadata, level, source, access_count) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("session:log", doc_id, "stub", "[]", 0.0,
             None, "L0", "test", access_count),
        )
        store._conn.commit()

    def _make_cli_snapshot(self, monkeypatch, pressure_level):
        """Patch the ``snapshot`` binding in cli.py to the given pressure."""
        from skill_hub import cli
        from skill_hub.resource_monitor import Pressure, SystemSnapshot

        monkeypatch.setattr(
            cli, "snapshot",
            lambda force=False: SystemSnapshot(
                cpu_load_1m=0.1 if pressure_level < Pressure.HIGH else 0.9,
                memory_used_pct=0.1 if pressure_level < Pressure.HIGH else 0.9,
                memory_available_mb=4096 if pressure_level < Pressure.HIGH else 512,
                total_memory_mb=8192,
                pressure=pressure_level,
                timestamp=0.0,
            ),
        )

    def test_session_end_promote_upgrades_l0_to_l1(self, tmp_store, monkeypatch):
        """Accessed L0 entries are promoted to L1 at session end."""
        from skill_hub import cli
        from skill_hub.resource_monitor import Pressure

        self._seed_l0_vector(tmp_store, "doc:active", access_count=1)

        # Patch SkillStore at module level in cli — _run_session_end_promote opens it
        monkeypatch.setattr(cli, "SkillStore", lambda: tmp_store)
        monkeypatch.setattr(tmp_store, "close", lambda: None)
        self._make_cli_snapshot(monkeypatch, Pressure.LOW)

        cli._run_session_end_promote()

        row = tmp_store._conn.execute(
            "SELECT level FROM vectors WHERE doc_id = 'doc:active'",
        ).fetchone()
        assert row is not None
        assert row["level"] == "L1", f"Expected L1 promotion, got {row['level']}"

    def test_session_end_promote_skips_under_high_pressure(self, tmp_store, monkeypatch):
        """Under HIGH pressure the promotion pass is skipped."""
        from skill_hub import cli
        from skill_hub.resource_monitor import Pressure

        self._seed_l0_vector(tmp_store, "doc:nopromote", access_count=1)

        monkeypatch.setattr(cli, "SkillStore", lambda: tmp_store)
        monkeypatch.setattr(tmp_store, "close", lambda: None)
        self._make_cli_snapshot(monkeypatch, Pressure.HIGH)

        cli._run_session_end_promote()

        row = tmp_store._conn.execute(
            "SELECT level FROM vectors WHERE doc_id = 'doc:nopromote'",
        ).fetchone()
        assert row is not None
        assert row["level"] == "L0", "Entry should stay L0 when promotion skipped under HIGH"

    def test_session_end_promote_gated_by_config_flag(self, tmp_store, monkeypatch):
        """session_end_promote=False suppresses promotion entirely."""
        from skill_hub import cli
        from skill_hub import config as cfg_mod
        from skill_hub.resource_monitor import Pressure

        self._seed_l0_vector(tmp_store, "doc:flagoff", access_count=1)

        monkeypatch.setattr(cli, "SkillStore", lambda: tmp_store)
        monkeypatch.setattr(tmp_store, "close", lambda: None)
        self._make_cli_snapshot(monkeypatch, Pressure.LOW)

        # Override config.get so session_end_promote returns False
        monkeypatch.setattr(cfg_mod, "get",
                            lambda k: False if k == "session_end_promote" else None)

        cli._run_session_end_promote()

        row = tmp_store._conn.execute(
            "SELECT level FROM vectors WHERE doc_id = 'doc:flagoff'",
        ).fetchone()
        assert row is not None
        assert row["level"] == "L0", "Entry should stay L0 when session_end_promote=False"

    def test_session_end_promote_skips_zero_access_entries(self, tmp_store, monkeypatch):
        """L0 entries with zero accesses are NOT promoted (left for nightly prune)."""
        from skill_hub import cli
        from skill_hub.resource_monitor import Pressure

        self._seed_l0_vector(tmp_store, "doc:unused", access_count=0)

        monkeypatch.setattr(cli, "SkillStore", lambda: tmp_store)
        monkeypatch.setattr(tmp_store, "close", lambda: None)
        self._make_cli_snapshot(monkeypatch, Pressure.LOW)

        cli._run_session_end_promote()

        row = tmp_store._conn.execute(
            "SELECT level FROM vectors WHERE doc_id = 'doc:unused'",
        ).fetchone()
        assert row is not None
        assert row["level"] == "L0", "Zero-access L0 entries should not be promoted"
