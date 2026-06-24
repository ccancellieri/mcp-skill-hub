import os, json


def test_env_source(monkeypatch):
    from skill_hub.llm import credentials, registry
    monkeypatch.setenv("MY_KEY", "sk-123")
    monkeypatch.setenv("MY_KEY_BASE", "https://example/v1")
    p = registry.Provider(name="x", level="L3", kind="openai_compatible",
                           api_key={"source": "env", "ref": "MY_KEY"})
    base, key = credentials.resolve_credentials(p)
    assert (base, key) == ("https://example/v1", "sk-123")


def test_inline_source():
    from skill_hub.llm import credentials, registry
    p = registry.Provider(name="x", level="L3", kind="openai_compatible",
                          api_base="https://b/v1",
                          api_key={"source": "inline", "ref": "sk-inline"})
    assert credentials.resolve_credentials(p) == ("https://b/v1", "sk-inline")


def test_opencode_source(monkeypatch, tmp_path):
    from skill_hub.llm import credentials, registry
    cfg_dir = tmp_path / "opencode"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps({
        "provider": {"gw": {"options": {"baseURL": "https://gw/v1", "apiKey": "sk-gw"}}}
    }))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    p = registry.Provider(name="work", level="L3", kind="openai_compatible",
                          api_key={"source": "opencode", "ref": "gw"})
    base, key = credentials.resolve_credentials(p)
    assert base == "https://gw/v1" and key == "sk-gw"
