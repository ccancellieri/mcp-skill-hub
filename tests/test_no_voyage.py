"""Guard: Voyage embedding backend must be fully removed."""
import importlib


def test_embeddings_has_no_voyage():
    import skill_hub.embeddings as e
    importlib.reload(e)
    assert not hasattr(e, "_embed_voyage")
    src = open(e.__file__).read()
    assert "voyage" not in src.lower()


def test_default_priority_excludes_voyage(monkeypatch, tmp_path):
    p = tmp_path / "cfg.json"; p.write_text("{}")
    import skill_hub.config as c
    monkeypatch.setattr(c, "CONFIG_PATH", p)
    cfg = c.load_config()
    assert "voyage" not in (cfg.get("embedding_backend_priority") or [])
    assert "voyage_api_key" not in cfg
