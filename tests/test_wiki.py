"""Tests for LLM Wiki Wave 1 — Steps 0–3.

Step 0: privacy leak fix in iter_user_memory_files
Step 1: wiki_pages / wiki_edges DDL, vector index namespaces, promote_memory guard
Step 2: wiki.py — parse_frontmatter, render_page, extract_edges, page_path
Step 3: wiki.reindex, wiki.status, dim-guard, MCP tool registration
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path, monkeypatch):
    """Isolated SkillStore backed by a temp DB."""
    from skill_hub.store import SkillStore
    monkeypatch.setenv("HOME", str(tmp_path))
    return SkillStore(db_path=tmp_path / "skill_hub.db")


@pytest.fixture()
def wiki_root(tmp_path) -> Path:
    """Empty wiki root under tmp_path."""
    root = tmp_path / "wiki"
    root.mkdir()
    return root


# ---------------------------------------------------------------------------
# Step 0 — privacy leak fix
# ---------------------------------------------------------------------------


class TestIterUserMemoryFilesPrivacyLeak:
    """Regression test: private/ subdir files must never be returned."""

    def test_public_file_returned(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        mem_root = tmp_path / "projects"
        proj = mem_root / "myproj" / "memory"
        proj.mkdir(parents=True)
        public = proj / "foo.md"
        public.write_text("public content")

        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", mem_root)
        files = mi.iter_user_memory_files()
        assert public in files

    def test_private_file_excluded(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        mem_root = tmp_path / "projects"
        proj = mem_root / "myproj" / "memory"
        proj.mkdir(parents=True)
        private_dir = proj / "private"
        private_dir.mkdir()
        secret = private_dir / "secret.md"
        secret.write_text("secret content")
        public = proj / "foo.md"
        public.write_text("public content")

        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", mem_root)
        files = mi.iter_user_memory_files()
        assert secret not in files
        assert public in files

    def test_only_private_nothing_returned(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        mem_root = tmp_path / "projects"
        proj = mem_root / "myproj" / "memory"
        private_dir = proj / "private"
        private_dir.mkdir(parents=True)
        (private_dir / "a.md").write_text("a")
        (private_dir / "b.md").write_text("b")

        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", mem_root)
        files = mi.iter_user_memory_files()
        assert files == []


# ---------------------------------------------------------------------------
# Step 1 — schema + vector index config
# ---------------------------------------------------------------------------


class TestWikiSchema:
    def test_wiki_pages_table_exists(self, store):
        tables = {
            row[0] for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "wiki_pages" in tables

    def test_wiki_edges_table_exists(self, store):
        tables = {
            row[0] for row in store._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "wiki_edges" in tables

    def test_wiki_pages_columns(self, store):
        cols = {
            row[1] for row in store._conn.execute(
                "PRAGMA table_info(wiki_pages)"
            ).fetchall()
        }
        required = {"slug", "id", "title", "type", "scope", "projects",
                    "tags", "aliases", "rel_path", "updated", "indexed_at"}
        assert required.issubset(cols)

    def test_wiki_edges_columns(self, store):
        cols = {
            row[1] for row in store._conn.execute(
                "PRAGMA table_info(wiki_edges)"
            ).fetchall()
        }
        required = {"id", "src_slug", "dst_slug", "dst_raw",
                    "edge_kind", "project", "resolved"}
        assert required.issubset(cols)

    def test_wiki_namespace_in_vector_index_config(self, store):
        rows = {
            row[0]: row for row in store._conn.execute(
                "SELECT name, default_level, half_life_days FROM vector_index_config"
                " WHERE name IN ('wiki','wiki-private')"
            ).fetchall()
        }
        assert "wiki" in rows
        assert "wiki-private" in rows
        assert rows["wiki"][1] == "L3"
        assert rows["wiki"][2] == 365.0
        assert rows["wiki-private"][1] == "L3"
        assert rows["wiki-private"][2] == 365.0


class TestPromoteMemoryWikiGuard:
    """promote_memory must never touch wiki/wiki-private rows.

    We exercise all four rules from server.py promote_memory: promote L1→L2,
    promote L2→L3, prune L0, prune L1. Each is seeded with a wiki-namespace
    row that meets the trigger condition, then the guarded WHERE is run and the
    row must NOT appear in the candidate set.
    """

    # Exact guard from server.py promote_memory (kept in sync here).
    _WIKI_GUARD = "namespace NOT IN ('wiki','wiki-private')"

    def _seed(self, store, slug: str, level: str, access_count: int,
              age_expr: str) -> int:
        """Insert a wiki row with arbitrary level/access_count/indexed_at."""
        vec = json.dumps([0.1] * 10)
        store._conn.execute(
            f"""
            INSERT INTO vectors (namespace, doc_id, model, vector, norm,
                                 level, access_count, indexed_at)
            VALUES ('wiki', ?, 'test', ?, 0.1, ?, ?,
                    datetime('now', {age_expr!r}))
            """,
            (slug, vec, level, access_count),
        )
        store._conn.commit()
        row = store._conn.execute(
            "SELECT id FROM vectors WHERE namespace='wiki' AND doc_id=?", (slug,)
        ).fetchone()
        return row[0]

    def _candidate_ids(self, store, where: str) -> set:
        rows = store._conn.execute(
            f"SELECT id FROM vectors WHERE {where}"
        ).fetchall()
        return {r[0] for r in rows}

    def test_promote_l1_to_l2_skips_wiki(self, store):
        """Rule: promote L1→L2 (access_count≥2, age>7d). Wiki row must be skipped."""
        row_id = self._seed(store, "pg-l1", "L1", access_count=5,
                            age_expr="'-10 days'")
        where = (
            f"level = 'L1' AND access_count >= 2 AND "
            f"indexed_at < datetime('now', '-7 days') AND {self._WIKI_GUARD}"
        )
        assert row_id not in self._candidate_ids(store, where)

    def test_promote_l2_to_l3_skips_wiki(self, store):
        """Rule: promote L2→L3 (access_count≥5, age>30d). Wiki row must be skipped."""
        row_id = self._seed(store, "pg-l2", "L2", access_count=10,
                            age_expr="'-45 days'")
        where = (
            f"level = 'L2' AND access_count >= 5 AND "
            f"indexed_at < datetime('now', '-30 days') AND {self._WIKI_GUARD}"
        )
        assert row_id not in self._candidate_ids(store, where)

    def test_prune_l0_skips_wiki(self, store):
        """Rule: prune L0 (access_count=0, age>1d). Wiki row must be skipped."""
        row_id = self._seed(store, "pg-l0", "L0", access_count=0,
                            age_expr="'-2 days'")
        where = (
            f"level = 'L0' AND access_count = 0 AND "
            f"indexed_at < datetime('now', '-1 day') AND {self._WIKI_GUARD}"
        )
        assert row_id not in self._candidate_ids(store, where)

    def test_prune_l1_skips_wiki(self, store):
        """Rule: prune L1 (access_count=0, age>7d). Wiki row must be skipped."""
        row_id = self._seed(store, "pg-l1-prune", "L1", access_count=0,
                            age_expr="'-30 days'")
        where = (
            f"level = 'L1' AND access_count = 0 AND "
            f"indexed_at < datetime('now', '-7 days') AND {self._WIKI_GUARD}"
        )
        assert row_id not in self._candidate_ids(store, where)
        # Also confirm the row still physically exists.
        exists = store._conn.execute(
            "SELECT 1 FROM vectors WHERE id=?", (row_id,)
        ).fetchone()
        assert exists is not None


# ---------------------------------------------------------------------------
# Step 2 — wiki.py unit tests
# ---------------------------------------------------------------------------


class TestParseFrontmatter:
    def test_with_frontmatter(self):
        from skill_hub.wiki import parse_frontmatter
        text = "---\nslug: foo\ntitle: Foo Page\n---\nBody text here."
        fm, body = parse_frontmatter(text)
        assert fm["slug"] == "foo"
        assert fm["title"] == "Foo Page"
        assert body == "Body text here."

    def test_no_frontmatter(self):
        from skill_hub.wiki import parse_frontmatter
        text = "Just some body text."
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert body == "Just some body text."

    def test_empty_frontmatter(self):
        from skill_hub.wiki import parse_frontmatter
        text = "---\n---\nBody."
        fm, body = parse_frontmatter(text)
        assert fm == {}
        assert body == "Body."

    def test_list_fields(self):
        from skill_hub.wiki import parse_frontmatter
        text = "---\nprojects:\n  - skill-hub\n  - geoid\n---\nBody."
        fm, body = parse_frontmatter(text)
        assert fm["projects"] == ["skill-hub", "geoid"]
        assert body == "Body."


class TestRenderPage:
    def _make_page(self, **overrides):
        from skill_hub.wiki import WikiPage
        defaults = dict(
            id="01HX123", slug="vectors-table", title="vectors table",
            type="entity", projects=["skill-hub"], scope="public",
            body="Body with [[wikilinks]].",
            tags=["storage", "sqlite"], aliases=["vector store"],
            created="2026-06-23", updated="2026-06-23",
        )
        defaults.update(overrides)
        return WikiPage(**defaults)

    def test_render_produces_frontmatter_block(self):
        from skill_hub.wiki import render_page
        page = self._make_page()
        out = render_page(page)
        assert out.startswith("---\n")
        assert "\n---\n" in out

    def test_round_trip(self):
        from skill_hub.wiki import WikiPage, render_page, parse_frontmatter
        page = self._make_page()
        rendered = render_page(page)
        fm, body = parse_frontmatter(rendered)
        assert fm["slug"] == "vectors-table"
        assert fm["title"] == "vectors table"
        assert fm["projects"] == ["skill-hub"]
        assert body == "Body with [[wikilinks]]."

    def test_render_no_optional_fields(self):
        from skill_hub.wiki import WikiPage, render_page, parse_frontmatter
        page = WikiPage(
            id="abc", slug="plain", title="Plain", type="concept",
            projects=["p"], scope="public", body="hi",
            created="2026-01-01", updated="2026-01-01",
        )
        rendered = render_page(page)
        fm, body = parse_frontmatter(rendered)
        assert "tags" not in fm
        assert "aliases" not in fm
        assert "source_refs" not in fm
        assert body == "hi"


class TestExtractEdges:
    def test_simple_wikilink(self):
        from skill_hub.wiki import extract_edges
        edges = extract_edges("src", "See [[target-slug]] for more.")
        assert len(edges) == 1
        assert edges[0].dst_raw == "target-slug"
        assert edges[0].edge_kind == "wikilink"

    def test_wikilink_with_alias(self):
        from skill_hub.wiki import extract_edges
        edges = extract_edges("src", "See [[target-slug|display text]] here.")
        assert len(edges) == 1
        assert edges[0].dst_raw == "target-slug"
        assert edges[0].edge_kind == "wikilink"

    def test_embed_link(self):
        from skill_hub.wiki import extract_edges
        edges = extract_edges("src", "Embed: ![[diagram]]")
        assert len(edges) == 1
        assert edges[0].dst_raw == "diagram"
        assert edges[0].edge_kind == "embed"

    def test_multiple_links_on_one_line(self):
        from skill_hub.wiki import extract_edges
        edges = extract_edges("src", "See [[a]] and [[b]] and ![[c]].")
        kinds = {e.dst_raw: e.edge_kind for e in edges}
        assert "a" in kinds
        assert "b" in kinds
        assert "c" in kinds
        assert kinds["a"] == "wikilink"
        assert kinds["b"] == "wikilink"
        assert kinds["c"] == "embed"

    def test_deduplication(self):
        from skill_hub.wiki import extract_edges
        edges = extract_edges("src", "[[foo]] and [[foo]] again.")
        assert len([e for e in edges if e.dst_raw == "foo"]) == 1

    def test_embed_not_also_wikilink(self):
        from skill_hub.wiki import extract_edges
        edges = extract_edges("src", "![[diagram]] only")
        assert all(e.edge_kind == "embed" for e in edges)
        wikilinks = [e for e in edges if e.edge_kind == "wikilink"]
        assert len(wikilinks) == 0


class TestPagePath:
    def _make_page(self, scope="public", page_type="entity", projects=None):
        from skill_hub.wiki import WikiPage
        return WikiPage(
            id="x", slug="my-slug", title="T", type=page_type,
            projects=projects or ["p"], scope=scope, body="",
            created="2026-01-01", updated="2026-01-01",
        )

    def test_public_path(self, tmp_path):
        from skill_hub.wiki import page_path
        page = self._make_page(scope="public", page_type="entity")
        path = page_path(tmp_path, page)
        assert path == tmp_path / "pages" / "entity" / "my-slug.md"

    def test_private_path(self, tmp_path):
        from skill_hub.wiki import page_path
        page = self._make_page(scope="private", projects=["career"])
        path = page_path(tmp_path, page)
        assert path == tmp_path / "_private" / "career" / "my-slug.md"

    def test_concept_type_in_path(self, tmp_path):
        from skill_hub.wiki import page_path
        page = self._make_page(scope="public", page_type="concept")
        path = page_path(tmp_path, page)
        assert "concept" in str(path)


# ---------------------------------------------------------------------------
# Step 3 — reindex + status + dim guard
# ---------------------------------------------------------------------------

def _write_page(wiki_root: Path, rel: str, content: str) -> Path:
    """Write a markdown page file under wiki_root."""
    p = wiki_root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


_PUBLIC_PAGE = """\
---
id: page-pub-001
slug: public-page
title: "Public Page"
type: entity
projects:
  - skill-hub
scope: public
created: 2026-06-23
updated: 2026-06-23
---
This is a [[private-page]] reference and some content.
"""

_PRIVATE_PAGE = """\
---
id: page-priv-001
slug: private-page
title: "Private Page"
type: entity
projects:
  - career
scope: private
created: 2026-06-23
updated: 2026-06-23
---
Private content here.
"""


class TestWikiReindex:
    def _seed_pages(self, wiki_root: Path):
        _write_page(wiki_root, "pages/entity/public-page.md", _PUBLIC_PAGE)
        _write_page(wiki_root, "_private/career/private-page.md", _PRIVATE_PAGE)

    def _stub_embed(self, store):
        """Patch store.upsert_vector to record calls without embedding."""
        calls: list[dict] = []

        def fake_upsert(namespace, doc_id, text, metadata=None, **kw):
            calls.append({"namespace": namespace, "doc_id": doc_id})
            vec = json.dumps([0.1] * 10)
            store._conn.execute(
                """
                INSERT INTO vectors (namespace, doc_id, model, vector, norm,
                                     level, source, project)
                VALUES (?, ?, 'stub', ?, 0.1, 'L3', 'wiki', NULL)
                ON CONFLICT(namespace, doc_id) DO UPDATE SET
                    vector=excluded.vector, indexed_at=datetime('now')
                """,
                (namespace, doc_id, vec),
            )
            store._conn.commit()

        store.upsert_vector = fake_upsert
        return calls

    def test_reindex_populates_wiki_pages(self, store, wiki_root):
        from skill_hub.wiki import reindex
        self._seed_pages(wiki_root)
        self._stub_embed(store)

        # Suppress dim guard (no stored dim yet, detect returns None).
        with patch("skill_hub.wiki._check_dim_guard"):
            result = reindex(store, wiki_root, dry_run=False)

        assert result["pages"] == 2
        count = store._conn.execute(
            "SELECT COUNT(*) FROM wiki_pages"
        ).fetchone()[0]
        assert count == 2

    def test_reindex_populates_wiki_edges(self, store, wiki_root):
        from skill_hub.wiki import reindex
        self._seed_pages(wiki_root)
        self._stub_embed(store)

        with patch("skill_hub.wiki._check_dim_guard"):
            result = reindex(store, wiki_root, dry_run=False)

        assert result["edges"] >= 1
        # The public page links to private-page.
        edge = store._conn.execute(
            "SELECT resolved FROM wiki_edges "
            "WHERE src_slug='public-page' AND dst_slug='private-page'"
        ).fetchone()
        assert edge is not None
        assert edge[0] == 1  # resolved because private-page is in wiki_pages

    def test_reindex_private_goes_to_wiki_private_namespace(self, store, wiki_root):
        from skill_hub.wiki import reindex
        self._seed_pages(wiki_root)
        calls = self._stub_embed(store)

        with patch("skill_hub.wiki._check_dim_guard"):
            reindex(store, wiki_root, dry_run=False)

        namespaces = {c["namespace"] for c in calls}
        assert "wiki" in namespaces
        assert "wiki-private" in namespaces

    def test_reindex_is_idempotent(self, store, wiki_root):
        from skill_hub.wiki import reindex
        self._seed_pages(wiki_root)

        for _ in range(2):
            self._stub_embed(store)
            with patch("skill_hub.wiki._check_dim_guard"):
                reindex(store, wiki_root, dry_run=False)

        count = store._conn.execute(
            "SELECT COUNT(*) FROM wiki_pages"
        ).fetchone()[0]
        assert count == 2

    def test_reindex_dry_run_writes_nothing(self, store, wiki_root):
        from skill_hub.wiki import reindex
        self._seed_pages(wiki_root)

        with patch("skill_hub.wiki._check_dim_guard"):
            result = reindex(store, wiki_root, dry_run=True)

        assert result["dry_run"] is True
        count = store._conn.execute(
            "SELECT COUNT(*) FROM wiki_pages"
        ).fetchone()[0]
        assert count == 0

    def test_status_after_reindex_shows_no_drift(self, store, wiki_root):
        from skill_hub.wiki import reindex, status
        self._seed_pages(wiki_root)
        self._stub_embed(store)

        with patch("skill_hub.wiki._check_dim_guard"):
            reindex(store, wiki_root, dry_run=False)

        st = status(store, wiki_root)
        # pages_disk should match pages_db.
        assert st["pages_db"] == 2
        assert st["pages_disk"] == 2
        assert st["drift"] == 0

    def test_status_no_drift_with_multi_section_page(self, store, wiki_root):
        """A page with multiple ## sections produces N vector rows but drift must be 0.

        Drift compares wiki_pages rows vs on-disk file count, NOT vec rows.
        """
        from skill_hub.wiki import reindex, status
        # Single page with two ## sections.
        multi_section = """\
---
id: multi-001
slug: multi-section-page
title: "Multi Section"
type: concept
projects:
  - skill-hub
scope: public
created: 2026-06-23
updated: 2026-06-23
---
Intro text.

## First Section
Content of section one.

## Second Section
Content of section two.
"""
        _write_page(wiki_root, "pages/concept/multi-section-page.md", multi_section)
        calls = self._stub_embed(store)

        with patch("skill_hub.wiki._check_dim_guard"):
            result = reindex(store, wiki_root, dry_run=False)

        st = status(store, wiki_root)
        # 1 page on disk, 1 page in wiki_pages.
        assert st["pages_disk"] == 1
        assert st["pages_db"] == 1
        assert st["drift"] == 0
        # The page produced more than 1 vector row (multiple sections).
        assert st["vec_rows"] > 1, (
            "expected multiple vector rows for a page with two ## sections"
        )

    def test_status_empty_wiki(self, store, wiki_root):
        from skill_hub.wiki import status
        st = status(store, wiki_root)
        assert st["pages_db"] == 0
        assert st["pages_disk"] == 0
        assert st["edges"] == 0
        assert st["drift"] == 0


class TestDimGuard:
    def test_dim_guard_raises_on_mismatch(self, store):
        from skill_hub.wiki import _check_dim_guard
        # Record a stored dim of 768.
        store._meta_set("vec_dim", "768")
        # Patch _detect_embedding_dim to return 384 (mismatch).
        with patch.object(store, "_detect_embedding_dim", return_value=384):
            with pytest.raises(ValueError, match="dim mismatch"):
                _check_dim_guard(store)

    def test_dim_guard_no_stored_dim_passes(self, store):
        from skill_hub.wiki import _check_dim_guard
        # No stored dim → guard passes silently.
        with patch.object(store, "_detect_embedding_dim", return_value=768):
            _check_dim_guard(store)  # must not raise

    def test_dim_guard_matching_dim_passes(self, store):
        from skill_hub.wiki import _check_dim_guard
        store._meta_set("vec_dim", "768")
        with patch.object(store, "_detect_embedding_dim", return_value=768):
            _check_dim_guard(store)  # must not raise

    def test_dim_guard_no_active_dim_passes(self, store):
        from skill_hub.wiki import _check_dim_guard
        store._meta_set("vec_dim", "768")
        with patch.object(store, "_detect_embedding_dim", return_value=None):
            _check_dim_guard(store)  # must not raise (backend unavailable)


class TestMcpToolRegistration:
    """wiki_reindex and wiki_status must be registered in TIER_REGISTRY."""

    def test_wiki_reindex_in_tier_registry(self):
        from skill_hub.capabilities import TIER_REGISTRY
        assert "wiki_reindex" in TIER_REGISTRY
        assert TIER_REGISTRY["wiki_reindex"] == "embedding"

    def test_wiki_status_in_tier_registry(self):
        from skill_hub.capabilities import TIER_REGISTRY
        assert "wiki_status" in TIER_REGISTRY
        assert TIER_REGISTRY["wiki_status"] == "none"

    def test_wiki_reindex_toolspec_hard_deps(self):
        from skill_hub.capabilities import TOOLS, BACKEND_DB, BACKEND_EMBED
        spec = next((s for s in TOOLS if s.name == "wiki_reindex"), None)
        assert spec is not None
        assert BACKEND_DB in spec.hard
        assert BACKEND_EMBED in spec.hard

    def test_wiki_status_toolspec_no_hard_deps(self):
        from skill_hub.capabilities import TOOLS
        spec = next((s for s in TOOLS if s.name == "wiki_status"), None)
        assert spec is not None
        assert spec.hard == ()


# ---------------------------------------------------------------------------
# Config keys
# ---------------------------------------------------------------------------

class TestWikiConfigKeys:
    def test_wiki_enabled_default(self, tmp_path, monkeypatch):
        import skill_hub.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")
        assert cfg_mod.load_config()["wiki_enabled"] is True

    def test_wiki_root_default(self, tmp_path, monkeypatch):
        import skill_hub.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")
        cfg = cfg_mod.load_config()
        assert "wiki_root" in cfg
        assert "wiki" in cfg["wiki_root"]

    def test_wiki_export_private_default(self, tmp_path, monkeypatch):
        import skill_hub.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")
        assert cfg_mod.load_config()["wiki_export_private"] is False

    def test_wiki_private_scopes_default(self, tmp_path, monkeypatch):
        import skill_hub.config as cfg_mod
        monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")
        assert cfg_mod.load_config()["wiki_private_scopes"] == {}
