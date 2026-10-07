"""Index freshness sweep (#134): staleness detection, refresh pass, triggers."""
from __future__ import annotations

import json
import time

import pytest

import skill_hub.reindex_sweep as rs


@pytest.fixture()
def cfg_tmp(monkeypatch, tmp_path):
    import skill_hub.config as cfg
    wiki_root = tmp_path / "wiki"
    (wiki_root / "pages").mkdir(parents=True)
    p = tmp_path / "config.json"
    p.write_text(json.dumps({
        "wiki_root": str(wiki_root),
        "memory_retrieval_backend": "wiki",
        "reindex_sweep_enabled": True,
        "reindex_on_task_close": True,
    }))
    monkeypatch.setattr(cfg, "CONFIG_PATH", p)
    return wiki_root


def test_wiki_stale_since_detects_new_page(cfg_tmp):
    assert rs.wiki_stale_since(time.time() + 10) is False
    (cfg_tmp / "pages" / "new.md").write_text("# hi")
    assert rs.wiki_stale_since(0.0) is True
    assert rs.wiki_stale_since(time.time() + 10) is False


def test_run_refresh_skips_without_embed_backend(cfg_tmp, monkeypatch):
    import skill_hub.embeddings as emb
    monkeypatch.setattr(emb, "embed_available", lambda: False)
    out = rs.run_refresh(store=object())
    assert "skipped" in out


def test_run_refresh_reindexes_wiki_when_stale(cfg_tmp, monkeypatch, tmp_path):
    import skill_hub.embeddings as emb
    import skill_hub.memory_index as mi
    import skill_hub.wiki as wiki

    monkeypatch.setattr(emb, "embed_available", lambda: True)
    calls: dict = {}

    def fake_wiki_reindex(store, root, dry_run=False):
        calls["wiki"] = str(root)
        return {"pages": 2, "edges": 0, "vectors": 4}

    monkeypatch.setattr(wiki, "reindex", fake_wiki_reindex)
    monkeypatch.setattr(mi, "index_user_memory", lambda store: 3)
    monkeypatch.setattr(mi, "index_plugin_memory", lambda store: {"p1": 1})

    (cfg_tmp / "pages" / "new.md").write_text("# page")
    state = tmp_path / "state.json"
    out = rs.run_refresh(store=object(), state_file=state)

    assert calls["wiki"] == str(cfg_tmp)
    assert out["wiki_pages"] == 2
    assert "user_memory_files" not in out
    assert "plugin_memory_files" not in out
    assert state.exists()   # last_run recorded

    # Second pass: nothing changed since last_run → wiki skipped; raw memory
    # remains outside the selected retrieval/indexing backend.
    calls.clear()
    out2 = rs.run_refresh(store=object(), state_file=state)
    assert "wiki" not in calls
    assert "user_memory_files" not in out2


def test_run_refresh_wiki_error_does_not_block_memory(cfg_tmp, monkeypatch, tmp_path):
    import skill_hub.embeddings as emb
    import skill_hub.memory_index as mi
    import skill_hub.wiki as wiki

    monkeypatch.setattr(emb, "embed_available", lambda: True)
    monkeypatch.setattr(wiki, "reindex", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("dim guard tripped")))
    monkeypatch.setattr(mi, "index_user_memory", lambda store: 2)
    monkeypatch.setattr(mi, "index_plugin_memory", lambda store: {})

    state_file = tmp_path / "s.json"
    out = rs.run_refresh(store=object(), wiki=True, state_file=state_file)
    assert "wiki_error" in out
    assert "user_memory_files" not in out
    assert rs._read_state(state_file) == {}


def test_run_refresh_does_not_advance_state_on_partial_wiki_reindex(
    cfg_tmp, monkeypatch, tmp_path
):
    import skill_hub.embeddings as emb
    import skill_hub.memory_index as mi
    import skill_hub.wiki as wiki

    monkeypatch.setattr(emb, "embed_available", lambda: True)
    monkeypatch.setattr(wiki, "reindex", lambda *a, **k: {
        "pages": 3, "edges": 2, "vectors": 4, "errors": 1,
    })
    monkeypatch.setattr(mi, "index_plugin_memory", lambda store: {})
    state_file = tmp_path / "partial.json"

    result = rs.run_refresh(store=object(), state_file=state_file)

    assert "wiki_error" in result
    assert rs._read_state(state_file) == {}


def test_run_refresh_wiki_false_skips_transition_refresh(cfg_tmp, monkeypatch, tmp_path):
    import skill_hub.embeddings as emb
    import skill_hub.memory_index as mi
    import skill_hub.wiki as wiki

    monkeypatch.setattr(emb, "embed_available", lambda: True)
    calls = []
    monkeypatch.setattr(wiki, "reindex", lambda *a, **k: calls.append("wiki") or {})
    monkeypatch.setattr(mi, "index_user_memory", lambda store: calls.append("raw") or 1)
    monkeypatch.setattr(mi, "index_plugin_memory", lambda store: {})

    state_file = tmp_path / "state.json"
    out = rs.run_refresh(store=object(), wiki=False, state_file=state_file)
    assert calls == []
    assert "wiki_pages" not in out
    assert rs._read_state(state_file) == {}


def test_periodic_sweep_allows_backend_transition_when_wiki_is_not_stale(
    cfg_tmp, monkeypatch, tmp_path
):
    from types import SimpleNamespace
    import skill_hub.resource_monitor as monitor
    import skill_hub.store as store_module

    state_file = tmp_path / "state.json"
    state_file.write_text(json.dumps({
        "last_run": time.time() - 2 * 24 * 60 * 60,
        "memory_retrieval_backend": "raw",
    }))
    calls = []
    monkeypatch.setattr(rs, "wiki_stale_since", lambda ts: False)
    monkeypatch.setattr(monitor, "snapshot", lambda: SimpleNamespace(pressure=monitor.Pressure.IDLE))
    monkeypatch.setattr(store_module, "get_store", lambda: object())
    monkeypatch.setattr(rs, "run_refresh", lambda store, **kwargs: calls.append(kwargs) or {})

    rs._run_sweep(state_file=state_file, _reschedule=False)

    assert calls == [{"state_file": state_file}]


def test_run_refresh_raw_skips_wiki_and_indexes_user_memory(cfg_tmp, monkeypatch, tmp_path):
    import skill_hub.config as cfg
    import skill_hub.embeddings as emb
    import skill_hub.memory_index as mi
    import skill_hub.wiki as wiki

    config_file = cfg.CONFIG_PATH
    config_file.write_text(json.dumps({"memory_retrieval_backend": "raw"}))
    monkeypatch.setattr(emb, "embed_available", lambda: True)
    calls = []
    monkeypatch.setattr(wiki, "reindex", lambda *a, **k: calls.append("wiki") or {})
    monkeypatch.setattr(mi, "index_user_memory", lambda store: calls.append("raw") or 2)
    monkeypatch.setattr(mi, "index_plugin_memory", lambda store: {"p": 1})

    out = rs.run_refresh(store=object(), wiki=True, state_file=tmp_path / "raw.json")
    assert calls == ["raw"]
    assert out["user_memory_files"] == 2
    assert out["plugin_memory_files"] == 1
    assert "wiki_pages" not in out


def test_run_refresh_invalid_backend_skips_both_memory_sources(cfg_tmp, monkeypatch, tmp_path):
    import skill_hub.config as cfg
    import skill_hub.embeddings as emb
    import skill_hub.memory_index as mi
    import skill_hub.wiki as wiki

    cfg.CONFIG_PATH.write_text(json.dumps({"memory_retrieval_backend": []}))
    monkeypatch.setattr(emb, "embed_available", lambda: True)
    calls = []
    monkeypatch.setattr(wiki, "reindex", lambda *a, **k: calls.append("wiki") or {})
    monkeypatch.setattr(mi, "index_user_memory", lambda store: calls.append("raw") or 2)
    monkeypatch.setattr(mi, "index_plugin_memory", lambda store: {"p": 1})

    out = rs.run_refresh(store=object(), wiki=True, state_file=tmp_path / "invalid.json")
    assert calls == []
    assert "memory_backend_error" in out
    assert "plugin_memory_files" not in out


def test_refresh_after_task_close_honours_flag(cfg_tmp, monkeypatch, tmp_path):
    import skill_hub.config as cfg

    ran: list[int] = []
    monkeypatch.setattr(rs, "run_refresh", lambda store, **k: ran.append(1) or {})

    rs.refresh_after_task_close(store=object(), task_id=7)
    deadline = time.time() + 5
    while not ran and time.time() < deadline:
        time.sleep(0.02)
    assert ran

    # Flag off → no thread, no refresh.
    p = tmp_path / "config2.json"
    p.write_text(json.dumps({"reindex_on_task_close": False}))
    monkeypatch.setattr(cfg, "CONFIG_PATH", p)
    ran.clear()
    rs.refresh_after_task_close(store=object(), task_id=8)
    time.sleep(0.1)
    assert not ran
