"""Tests for the skill indexer (#126 — index pollution).

Covers:
- ``_skill_id_from_path``: literal "unknown" plugin-cache version dirs must
  not be mistaken for a plugin name (they used to produce ids like
  ``unknown:build-mcp-server``).
- ``SkillStore.dedupe_skills_by_content_hash``: byte-identical skills indexed
  under more than one id collapse to a single row, preferring the one whose
  file_path lives under ``plugins/cache/``.
- ``index_all``: end-to-end walk over a fake plugin tree exercises the
  marketplace/cache double-indexing dedup, the "unknown" version-dir fix, and
  confirms the existing orphan sweep (file_path gone -> row pruned) still
  self-heals a full reindex.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub import config as cfg
from skill_hub import indexer
from skill_hub.store import Skill, SkillStore


# ---------------------------------------------------------------------------
# _skill_id_from_path — "unknown" version dir
# ---------------------------------------------------------------------------

def test_unknown_version_dir_is_skipped_not_treated_as_plugin():
    path = Path(
        "/home/u/.claude/plugins/cache/claude-plugins-official/"
        "mcp-server-dev/unknown/skills/build-mcp-server/SKILL.md"
    )
    assert indexer._skill_id_from_path(path) == "mcp-server-dev:build-mcp-server"


def test_unknown_version_dir_skipped_for_other_plugins_too():
    path = Path(
        "/home/u/.claude/plugins/cache/claude-plugins-official/"
        "skill-creator/unknown/skills/skill-creator/SKILL.md"
    )
    assert indexer._skill_id_from_path(path) == "skill-creator:skill-creator"


def test_real_version_dir_still_skipped_as_before():
    path = Path(
        "/home/u/.claude/plugins/cache/claude-plugins-official/"
        "chrome-devtools-mcp/1.4.0/skills/debug-optimize-lcp/SKILL.md"
    )
    assert indexer._skill_id_from_path(path) == "chrome-devtools-mcp:debug-optimize-lcp"


# ---------------------------------------------------------------------------
# SkillStore.dedupe_skills_by_content_hash
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")
    return SkillStore(db_path=tmp_path / "skill_hub.db")


def _skill(id_, file_path, target="claude"):
    return Skill(
        id=id_, name=id_.split(":")[-1], description="dup content",
        content="same body every time", file_path=file_path,
        plugin=id_.split(":")[0], target=target,
    )


def test_dedupe_prefers_cache_row_over_marketplace_row(store):
    store.upsert_skill(
        _skill("document-skills:xlsx",
               "/home/u/.claude/plugins/cache/anthropic-agent-skills/"
               "document-skills/9d2f/skills/xlsx/SKILL.md"),
        content_hash="H1",
    )
    store.upsert_skill(
        _skill("plugins:xlsx",
               "/home/u/.claude/plugins/marketplaces/anthropic-agent-skills/"
               "skills/xlsx/SKILL.md"),
        content_hash="H1",
    )
    deleted = store.dedupe_skills_by_content_hash()
    assert deleted == ["plugins:xlsx"]
    remaining = {r["id"] for r in store._conn.execute("SELECT id FROM skills")}
    assert remaining == {"document-skills:xlsx"}


def test_dedupe_ties_break_on_smaller_id(store):
    """Neither row is a cache row (or both are) — tie-break deterministically."""
    store.upsert_skill(
        _skill("example-skills:skill-creator",
               "/home/u/.claude/plugins/cache/anthropic-agent-skills/"
               "example-skills/9d2f/skills/skill-creator/SKILL.md"),
        content_hash="H2",
    )
    store.upsert_skill(
        _skill("document-skills:skill-creator",
               "/home/u/.claude/plugins/cache/anthropic-agent-skills/"
               "document-skills/9d2f/skills/skill-creator/SKILL.md"),
        content_hash="H2",
    )
    deleted = store.dedupe_skills_by_content_hash()
    assert deleted == ["example-skills:skill-creator"]
    remaining = {r["id"] for r in store._conn.execute("SELECT id FROM skills")}
    assert remaining == {"document-skills:skill-creator"}


def test_dedupe_leaves_distinct_content_alone(store):
    store.upsert_skill(_skill("a:one", "/x/a/one/SKILL.md"), content_hash="HA")
    store.upsert_skill(_skill("b:two", "/x/b/two/SKILL.md"), content_hash="HB")
    assert store.dedupe_skills_by_content_hash() == []
    remaining = {r["id"] for r in store._conn.execute("SELECT id FROM skills")}
    assert remaining == {"a:one", "b:two"}


def test_dedupe_ignores_local_target_rows(store):
    store.upsert_skill(_skill("local:foo", "/x/foo.json", target="local"), content_hash="HL")
    store.upsert_skill(_skill("local:bar", "/x/bar.json", target="local"), content_hash="HL")
    assert store.dedupe_skills_by_content_hash() == []
    remaining = {r["id"] for r in store._conn.execute("SELECT id FROM skills")}
    assert remaining == {"local:foo", "local:bar"}


def test_dedupe_companion_tables_cleaned(store):
    """delete_skill cascades to embeddings — confirm dedupe doesn't leak vectors."""
    store.upsert_skill(
        _skill("document-skills:xlsx", "/x/plugins/cache/y/xlsx/SKILL.md"),
        content_hash="H1",
    )
    store.upsert_skill(
        _skill("plugins:xlsx", "/x/plugins/marketplaces/y/xlsx/SKILL.md"),
        content_hash="H1",
    )
    store.upsert_embedding("plugins:xlsx", "test-model", [0.1, 0.2, 0.3])
    store.dedupe_skills_by_content_hash()
    row = store._conn.execute(
        "SELECT COUNT(*) c FROM embeddings WHERE skill_id = 'plugins:xlsx'"
    ).fetchone()
    assert row["c"] == 0


# ---------------------------------------------------------------------------
# index_all — end-to-end over a fake plugin tree
# ---------------------------------------------------------------------------

SKILL_TEMPLATE = """---
name: {name}
description: {description}
---

# {name}

Body content for {name}.
"""


def _write_skill(base: Path, *segments: str, name: str, description: str = "desc") -> Path:
    skill_dir = base.joinpath(*segments)
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(SKILL_TEMPLATE.format(name=name, description=description))
    return skill_file


class _StubCfg:
    """Stand-in for skill_hub.config inside index_all — avoids any real
    config/HOME I/O for the two keys index_all reads directly."""

    def get(self, key):
        if key == "extra_skill_dirs":
            return []
        if key == "local_skills_dir":
            return "/nonexistent/local-skills"
        return None


@pytest.fixture()
def index_env(tmp_path, monkeypatch):
    """Isolated store + fake ~/.claude/plugins/{cache,marketplaces} tree.

    Layout mirrors production (.../plugins/cache/... and .../plugins/marketplaces/...)
    so path-walk-back plugin-name derivation behaves exactly as it does on a
    real install, including the "plugins:<skill>" bug pattern for marketplaces
    whose skills/ tree sits directly under the marketplace root.
    """
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr(indexer, "_cfg", _StubCfg())

    plugins_root = tmp_path / "plugins"
    cache_dir = plugins_root / "cache"
    marketplaces_dir = plugins_root / "marketplaces"
    monkeypatch.setattr(indexer, "PLUGIN_DIRS", [cache_dir, marketplaces_dir])

    # Stub embed() so no live Ollama/embedding backend is required.
    monkeypatch.setattr(indexer, "embed", lambda text, model=None: [0.1, 0.2, 0.3])

    # Stub the memory-index steps index_all() runs after the skill scan: they
    # import from memory_index lazily (so the indexer.embed stub doesn't reach
    # them) and would otherwise walk the real ~/.claude memory tree and hit the
    # live embedding backend.
    from skill_hub import memory_index as _memory_index

    monkeypatch.setattr(_memory_index, "index_plugin_memory", lambda store: {})
    monkeypatch.setattr(_memory_index, "index_user_memory", lambda store: 0)

    store = SkillStore(db_path=tmp_path / "skill_hub.db")
    return store, cache_dir, marketplaces_dir


def test_index_all_dedupes_cache_vs_marketplace_double_index(index_env):
    store, cache_dir, marketplaces_dir = index_env
    # Installed (cache) copy of document-skills' xlsx skill. The version
    # segment must look like Claude Code's real cache hash (8+ hex chars) so
    # the plugin-name walk-back skips it the same way it does in production.
    _write_skill(cache_dir, "anthropic-agent-skills", "document-skills", "9d2f1ae18723",
                 "skills", "xlsx", name="xlsx", description="Excel spreadsheets")
    # Git-checkout (marketplace) copy of the same skill, byte-identical.
    _write_skill(marketplaces_dir, "anthropic-agent-skills", "skills", "xlsx",
                 name="xlsx", description="Excel spreadsheets")

    indexed, errors = indexer.index_all(store)

    ids = {r["id"] for r in store._conn.execute("SELECT id FROM skills")}
    assert "document-skills:xlsx" in ids
    assert "plugins:xlsx" not in ids
    assert sum(1 for i in ids if i.endswith(":xlsx")) == 1
    assert any("deduped" in e for e in errors)


def test_index_all_fixes_unknown_version_dir_id(index_env):
    store, cache_dir, _ = index_env
    _write_skill(cache_dir, "claude-plugins-official", "mcp-server-dev", "unknown",
                 "skills", "build-mcp-server", name="build-mcp-server",
                 description="Scaffold an MCP server")

    indexer.index_all(store)

    ids = {r["id"] for r in store._conn.execute("SELECT id FROM skills")}
    assert "mcp-server-dev:build-mcp-server" in ids
    assert not any(i.startswith("unknown:") for i in ids)


def test_index_all_prunes_orphaned_row_on_reindex(index_env):
    store, cache_dir, _ = index_env
    skill_file = _write_skill(cache_dir, "claude-plugins-official", "superpowers", "6.0.0",
                               "skills", "using-git-worktrees", name="using-git-worktrees",
                               description="Use git worktrees for parallel branches")

    indexer.index_all(store)
    ids = {r["id"] for r in store._conn.execute("SELECT id FROM skills")}
    assert "superpowers:using-git-worktrees" in ids

    skill_file.unlink()
    indexed, errors = indexer.index_all(store)

    ids = {r["id"] for r in store._conn.execute("SELECT id FROM skills")}
    assert "superpowers:using-git-worktrees" not in ids
    assert any("pruned" in e for e in errors)
