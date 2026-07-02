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
    assert out["user_memory_files"] == 3
    assert out["plugin_memory_files"] == 1
    assert state.exists()   # last_run recorded

    # Second pass: nothing changed since last_run → wiki skipped, memory still runs.
    calls.clear()
    out2 = rs.run_refresh(store=object(), state_file=state)
    assert "wiki" not in calls
    assert out2["user_memory_files"] == 3


def test_run_refresh_wiki_error_does_not_block_memory(cfg_tmp, monkeypatch, tmp_path):
    import skill_hub.embeddings as emb
    import skill_hub.memory_index as mi
    import skill_hub.wiki as wiki

    monkeypatch.setattr(emb, "embed_available", lambda: True)
    monkeypatch.setattr(wiki, "reindex", lambda *a, **k: (_ for _ in ()).throw(
        ValueError("dim guard tripped")))
    monkeypatch.setattr(mi, "index_user_memory", lambda store: 2)
    monkeypatch.setattr(mi, "index_plugin_memory", lambda store: {})

    out = rs.run_refresh(store=object(), wiki=True, state_file=tmp_path / "s.json")
    assert "wiki_error" in out
    assert out["user_memory_files"] == 2


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
