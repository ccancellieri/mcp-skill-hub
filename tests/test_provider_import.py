# tests/test_provider_import.py
from __future__ import annotations
import json
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from skill_hub.llm import importers as imp  # noqa: E402
from skill_hub.llm import registry as reg   # noqa: E402


OPENCODE = {
    "provider": {
        "agent-platform": {
            "npm": "@ai-sdk/openai-compatible",
            "name": "FAO Vibe Coding",
            "options": {"baseURL": "https://gw.example/v1", "apiKey": "SECRET"},
            "models": {
                "zai-org/glm-5-maas": {"name": "GLM-5"},
                "google/gemini-3.1-pro-preview": {"name": "Gemini 3.1"},
                "anthropic/claude-opus-4-8": {"name": "Opus"},
            },
        }
    }
}


def test_normalize_opencode_refs_credentials_without_copying_secret():
    provs = imp.normalize("opencode", OPENCODE)
    assert len(provs) == 1
    p = provs[0]
    assert p.name == "FAO Vibe Coding"
    assert p.kind == "openai_compatible"
    assert p.api_base == ""                       # resolved live, not copied
    assert p.api_key == {"source": "opencode", "ref": "agent-platform"}
    assert "SECRET" not in str(p.api_key)         # secret never copied in
    assert set(p.model_ids) == {
        "zai-org/glm-5-maas", "google/gemini-3.1-pro-preview",
        "anthropic/claude-opus-4-8"}
    assert p.match_ref == "agent-platform"


def test_normalize_openai_generic_inline_key():
    payload = {"name": "local", "baseURL": "http://h/v1", "apiKey": "k",
               "models": ["m1", "m2"]}
    p = imp.normalize("openai", payload)[0]
    assert p.kind == "openai_compatible"
    assert p.api_base == "http://h/v1"
    assert p.api_key == {"source": "inline", "ref": "k"}
    assert p.model_ids == ["m1", "m2"]


def test_normalize_litellm_groups_by_base():
    payload = {"model_list": [
        {"model_name": "a", "litellm_params": {"model": "vendor/a",
         "api_base": "http://x/v1", "api_key": "k"}},
        {"model_name": "b", "litellm_params": {"model": "vendor/b",
         "api_base": "http://x/v1"}},
        {"model_name": "c", "litellm_params": {"model": "other/c",
         "api_base": "http://y/v1"}},
    ]}
    provs = imp.normalize("litellm", payload)
    assert len(provs) == 2                         # two distinct endpoints
    by_base = {p.api_base: p for p in provs}
    assert by_base["http://x/v1"].model_ids == ["vendor/a", "vendor/b"]
    assert by_base["http://x/v1"].api_key == {"source": "inline", "ref": "k"}


def test_normalize_unsupported_format_raises():
    import pytest
    with pytest.raises(ValueError):
        imp.normalize("nope", {})


def test_merge_into_existing_preserves_tags_and_complexity():
    current = [{
        "name": "work-gateway", "level": "L3", "kind": "openai_compatible",
        "api_base": "", "api_key": {"source": "opencode", "ref": "agent-platform"},
        "enabled": True, "order": 30,
        "models": [
            {"id": "zai-org/glm-5-maas", "complexity": "heavy",
             "tags": ["code-review", "git"]},
            {"id": "removed/old-model", "complexity": "light", "tags": ["x"]},
        ],
    }]
    incoming = imp.normalize("opencode", OPENCODE)
    merged, diffs = imp.merge_registry(current, incoming)

    assert len(merged) == 1                        # matched by ref, not duplicated
    rec = merged[0]
    assert rec["name"] == "work-gateway"           # display name preserved
    assert rec["level"] == "L3" and rec["order"] == 30
    by_id = {m["id"]: m for m in rec["models"]}
    # surviving id keeps its tuned complexity + tags
    assert by_id["zai-org/glm-5-maas"]["complexity"] == "heavy"
    assert by_id["zai-org/glm-5-maas"]["tags"] == ["code-review", "git"]
    # new ids added with defaults
    assert by_id["anthropic/claude-opus-4-8"]["complexity"] == "light"
    # stale id dropped
    assert "removed/old-model" not in by_id

    d = diffs[0]
    assert d["status"] == "update"
    assert d["matched_name"] == "work-gateway"
    assert "zai-org/glm-5-maas" in d["models_kept"]
    assert "anthropic/claude-opus-4-8" in d["models_added"]
    assert "removed/old-model" in d["models_removed"]


def test_merge_new_provider_appended_with_default_level():
    merged, diffs = imp.merge_registry([], imp.normalize("opencode", OPENCODE))
    assert len(merged) == 1
    assert merged[0]["level"] == "L3"
    assert diffs[0]["status"] == "new"
    assert set(diffs[0]["models_added"]) == set(merged[0]["models"][i]["id"]
                                                for i in range(len(merged[0]["models"])))


def test_diff_is_secret_free():
    current = []
    incoming = imp.normalize("openai", {"name": "x", "baseURL": "http://h",
                                         "apiKey": "TOPSECRET", "models": ["m"]})
    diffs = imp.diff_registry(current, incoming)
    assert "TOPSECRET" not in str(diffs)           # label only, never the key
    assert diffs[0]["cred_label"] == "inline"


def test_merged_records_pass_registry_validation():
    merged, _ = imp.merge_registry([], imp.normalize("opencode", OPENCODE))
    for rec in merged:
        assert reg._parse_provider(rec) is not None


def test_empty_keyed_credential_does_not_wipe_existing():
    # A generic import without a key must not erase an existing credential.
    current = [{
        "name": "h", "level": "L3", "kind": "openai_compatible",
        "api_base": "http://h", "api_key": {"source": "env", "ref": "KEY"},
        "enabled": True, "order": 10, "models": [{"id": "m"}],
    }]
    incoming = imp.normalize("openai", {"name": "h", "baseURL": "http://h",
                                        "models": ["m", "m2"]})
    merged, _ = imp.merge_registry(current, incoming)
    assert merged[0]["api_key"] == {"source": "env", "ref": "KEY"}


def test_inline_key_does_not_downgrade_env_credential():
    # A re-sync that carries an inline key must NOT clobber a deliberate env cred.
    current = [{
        "name": "h", "level": "L3", "kind": "openai_compatible",
        "api_base": "http://h", "api_key": {"source": "env", "ref": "KEY"},
        "enabled": True, "order": 10, "models": [{"id": "m"}],
    }]
    incoming = imp.normalize("openai", {"name": "h", "baseURL": "http://h",
                                        "apiKey": "PLAINTEXT", "models": ["m"]})
    merged, _ = imp.merge_registry(current, incoming)
    assert merged[0]["api_key"] == {"source": "env", "ref": "KEY"}


def test_inline_key_can_be_rotated():
    # An existing *inline* key may still be replaced (rotation is allowed).
    current = [{
        "name": "h", "level": "L3", "kind": "openai_compatible",
        "api_base": "http://h", "api_key": {"source": "inline", "ref": "OLD"},
        "enabled": True, "order": 10, "models": [{"id": "m"}],
    }]
    incoming = imp.normalize("openai", {"name": "h", "baseURL": "http://h",
                                        "apiKey": "NEW", "models": ["m"]})
    merged, _ = imp.merge_registry(current, incoming)
    assert merged[0]["api_key"] == {"source": "inline", "ref": "NEW"}


# ── classification ────────────────────────────────────────────────────────────

def test_parse_classification_keeps_known_ids_and_coerces():
    text = ('noise {"opus": {"complexity": "heavy", "tags": ["reasoning", "code"]}, '
            '"mini": {"complexity": "light", "tags": ["fast"]}, '
            '"ghost": {"complexity": "heavy"}} trailing')
    out = imp._parse_classification(text, ["opus", "mini"])
    assert out == {
        "opus": {"complexity": "heavy", "tags": ["reasoning", "code"]},
        "mini": {"complexity": "light", "tags": ["fast"]},
    }                                              # unrequested "ghost" dropped


def test_parse_classification_unknown_complexity_defaults_light():
    out = imp._parse_classification('{"m": {"complexity": "medium", "tags": ["bogus"]}}',
                                    ["m"])
    assert out == {"m": {"complexity": "light", "tags": []}}  # bad tag filtered


def test_parse_classification_bad_json_returns_empty():
    assert imp._parse_classification("not json at all", ["m"]) == {}
    assert imp._parse_classification("", ["m"]) == {}


def test_classify_models_best_effort_on_llm_error(monkeypatch):
    class _Boom:
        def complete(self, *a, **k):
            raise RuntimeError("ladder exhausted")
    # get_provider is imported lazily via `from . import get_provider`, which
    # resolves through the skill_hub.llm package — patch it there.
    import skill_hub.llm as _llm
    monkeypatch.setattr(_llm, "get_provider", lambda: _Boom())
    assert imp.classify_models(["a", "b"]) == {}


def test_apply_classification_labels_only_added_models():
    merged = [{
        "name": "g", "level": "L3", "kind": "openai_compatible", "api_base": "",
        "api_key": {}, "enabled": True, "order": 30,
        "models": [
            {"id": "kept", "complexity": "heavy", "tags": ["git"]},   # surviving
            {"id": "new-opus", "complexity": "light", "tags": []},    # added
        ],
    }]
    diffs = [{"provider": "g", "status": "update",
              "models_added": ["new-opus"], "models_removed": [], "models_kept": ["kept"]}]

    def fake_classify(ids):
        assert ids == ["new-opus"]                 # only the added id is classified
        return {"new-opus": {"complexity": "heavy", "tags": ["reasoning"]}}

    applied = imp.apply_classification(merged, diffs, fake_classify)
    by_id = {m["id"]: m for m in merged[0]["models"]}
    assert by_id["new-opus"]["complexity"] == "heavy"
    assert by_id["new-opus"]["tags"] == ["reasoning"]
    assert by_id["kept"]["complexity"] == "heavy" and by_id["kept"]["tags"] == ["git"]
    assert applied == {"new-opus": {"complexity": "heavy", "tags": ["reasoning"]}}


def test_apply_classification_best_effort_leaves_light_on_failure():
    merged = [{
        "name": "g", "level": "L3", "kind": "openai_compatible", "api_base": "",
        "api_key": {}, "enabled": True, "order": 30,
        "models": [{"id": "new-x", "complexity": "light", "tags": []}],
    }]
    diffs = [{"provider": "g", "status": "new",
              "models_added": ["new-x"], "models_removed": [], "models_kept": []}]

    def boom(ids):
        raise RuntimeError("classifier down")

    # Both an empty result and a raising classifier must leave the safe default.
    assert imp.apply_classification(merged, diffs, lambda ids: {}) == {}
    assert imp.apply_classification(merged, diffs, boom) == {}
    assert merged[0]["models"][0]["complexity"] == "light"


def test_apply_classification_shared_id_across_providers_does_not_touch_survivor():
    # Two providers both expose "shared": provider A has it as a surviving
    # (tuned) model, provider B adds it. Only B's copy may be relabelled.
    merged = [
        {"name": "A", "level": "L3", "kind": "openai_compatible", "api_base": "",
         "api_key": {}, "enabled": True, "order": 10,
         "models": [{"id": "shared", "complexity": "heavy", "tags": ["git"]}]},
        {"name": "B", "level": "L3", "kind": "openai_compatible", "api_base": "",
         "api_key": {}, "enabled": True, "order": 20,
         "models": [{"id": "shared", "complexity": "light", "tags": []}]},
    ]
    diffs = [
        {"provider": "A", "status": "update", "models_added": [],
         "models_removed": [], "models_kept": ["shared"]},
        {"provider": "B", "status": "new", "models_added": ["shared"],
         "models_removed": [], "models_kept": []},
    ]
    imp.apply_classification(
        merged, diffs,
        lambda ids: {"shared": {"complexity": "heavy", "tags": ["reasoning"]}})
    a = merged[0]["models"][0]
    b = merged[1]["models"][0]
    assert a["complexity"] == "heavy" and a["tags"] == ["git"]   # survivor intact
    assert b["complexity"] == "heavy" and b["tags"] == ["reasoning"]  # added labelled


# ── inject_skill_hub_mcp ──────────────────────────────────────────────────────

def test_inject_skill_hub_mcp_creates_block(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    cfgdir = tmp_path / "opencode"
    cfgdir.mkdir(parents=True)
    (cfgdir / "config.json").write_text(
        '{"$schema":"x","provider":{"p":{"name":"P"}}}')
    result = imp.inject_skill_hub_mcp()
    assert result["replaced"] is False
    assert result["command"]                      # non-empty launch command
    doc = json.loads((cfgdir / "config.json").read_text())
    assert doc["mcp"]["skill-hub"]["type"] == "local"
    assert doc["mcp"]["skill-hub"]["enabled"] is True
    assert doc["mcp"]["skill-hub"]["command"] == result["command"]
    assert doc["provider"]["p"]["name"] == "P"     # untouched
    assert doc["$schema"] == "x"


def test_inject_skill_hub_mcp_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    (tmp_path / "opencode").mkdir(parents=True)
    imp.inject_skill_hub_mcp()
    result2 = imp.inject_skill_hub_mcp()
    assert result2["replaced"] is True
    doc = json.loads((tmp_path / "opencode" / "config.json").read_text())
    assert "skill-hub" in doc["mcp"]


def test_inject_skill_hub_mcp_creates_missing_file(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    result = imp.inject_skill_hub_mcp()
    assert result["replaced"] is False
    doc = json.loads(Path(result["path"]).read_text())
    assert doc["mcp"]["skill-hub"]["enabled"] is True
