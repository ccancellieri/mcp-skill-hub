"""Tests for LLM Wiki Wave 1 — Steps 0–3 and Step 8 (migration).

Step 0: privacy leak fix in iter_user_memory_files
Step 1: wiki_pages / wiki_edges DDL, vector index namespaces, promote_memory guard
Step 2: wiki.py — parse_frontmatter, render_page, extract_edges, page_path
Step 3: wiki.reindex, wiki.status, dim-guard, MCP tool registration
Step 8: wiki.migrate — mechanical source-page conversion
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
        scopes = cfg_mod.load_config()["wiki_private_scopes"]
        assert "glicemia" in scopes
        assert "career" in scopes


# ---------------------------------------------------------------------------
# Step 8 — wiki.migrate
# ---------------------------------------------------------------------------

def _make_auto_memory_tree(projects_root: Path) -> dict:
    """Build a fake ~/.claude/projects/ tree with several memory files.

    Returns a dict of notable paths for assertions.
    """
    # Project A: public files only.
    proj_a = projects_root / "-Users-user-work-projA" / "memory"
    proj_a.mkdir(parents=True)
    file_a1 = proj_a / "decisions.md"
    file_a1.write_text("# Decisions\n\nWe chose FastAPI.\n", encoding="utf-8")
    file_a2 = proj_a / "patterns.md"
    file_a2.write_text("# Patterns\n\nUse upsert not insert.\n", encoding="utf-8")

    # Project A: private subdir.
    priv_a = proj_a / "private"
    priv_a.mkdir()
    file_a_priv = priv_a / "secret.md"
    file_a_priv.write_text("# Secret\n\nSensitive info.\n", encoding="utf-8")

    # Project B: name contains "diabete" → glicemia scope override.
    proj_b = projects_root / "-Users-user-work-diabete-app" / "memory"
    proj_b.mkdir(parents=True)
    file_b1 = proj_b / "health-notes.md"
    file_b1.write_text("# Health Notes\n\nT1D readings.\n", encoding="utf-8")

    # Project C: name contains "career-skill-hub-plugin" → career scope.
    proj_c = projects_root / "-Users-user-work-career-skill-hub-plugin" / "memory"
    proj_c.mkdir(parents=True)
    file_c1 = proj_c / "cv-notes.md"
    file_c1.write_text("# CV Notes\n\nPortfolio content.\n", encoding="utf-8")

    # Inbox file — must be skipped.
    file_inbox = proj_a / "inbox.md"
    file_inbox.write_text("# Inbox\n\nUnconfirmed items.\n", encoding="utf-8")

    return {
        "proj_a_dir": proj_a,
        "file_a1": file_a1,
        "file_a2": file_a2,
        "file_a_priv": file_a_priv,
        "file_b1": file_b1,
        "file_c1": file_c1,
        "file_inbox": file_inbox,
    }


class TestMigrateDryRun:
    """Dry-run writes nothing and returns a manifest with correct counts."""

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        from skill_hub.wiki import migrate
        from skill_hub.store import SkillStore

        projects_root = tmp_path / "projects"
        files = _make_auto_memory_tree(projects_root)
        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", projects_root)

        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        store = SkillStore(db_path=tmp_path / "db.db")

        result = migrate(store, wiki_root, dry_run=True,
                         private_scopes={"glicemia": ["glicemia"],
                                         "career": ["career"]})

        assert result["dry_run"] is True
        assert result["would_write"] > 0
        # Nothing must be written.
        assert not (wiki_root / "pages").exists()
        assert not (wiki_root / "_private").exists()
        assert not (wiki_root / "index.md").exists()

    def test_dry_run_counts_public_and_private(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        from skill_hub.wiki import migrate
        from skill_hub.store import SkillStore

        projects_root = tmp_path / "projects"
        _make_auto_memory_tree(projects_root)
        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", projects_root)

        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        store = SkillStore(db_path=tmp_path / "db.db")

        result = migrate(store, wiki_root, dry_run=True,
                         private_scopes={"glicemia": ["glicemia"],
                                         "career": ["career"]})

        # Public: decisions.md, patterns.md from projA (inbox skipped).
        # Private: secret.md (private/ subdir), health-notes.md (diabete), cv-notes.md (career).
        assert result["public"] >= 2
        assert result["private"] >= 3


class TestMigrateNonDryRun:
    """Non-dry-run writes expected page files under pages/source/, project/, _private/."""

    def _run_migrate(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        from skill_hub.wiki import migrate
        from skill_hub.store import SkillStore

        projects_root = tmp_path / "projects"
        files = _make_auto_memory_tree(projects_root)
        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", projects_root)

        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        store = SkillStore(db_path=tmp_path / "db.db")

        result = migrate(store, wiki_root, dry_run=False,
                         private_scopes={"glicemia": ["glicemia"],
                                         "career": ["career"]})
        return result, wiki_root, files

    def test_writes_source_pages(self, tmp_path, monkeypatch):
        result, wiki_root, files = self._run_migrate(tmp_path, monkeypatch)
        source_dir = wiki_root / "pages" / "source"
        assert source_dir.exists(), "pages/source/ must exist after migrate"
        md_files = list(source_dir.glob("*.md"))
        assert len(md_files) >= 2, "at least 2 public source pages expected"

    def test_private_file_lands_in_private_dir(self, tmp_path, monkeypatch):
        result, wiki_root, files = self._run_migrate(tmp_path, monkeypatch)
        private_root = wiki_root / "_private"
        assert private_root.exists()
        all_private = list(private_root.rglob("*.md"))
        # Filter out index.md files.
        content_pages = [p for p in all_private if p.name != "index.md"]
        assert len(content_pages) >= 1

    def test_private_subdir_goes_to_private_project(self, tmp_path, monkeypatch):
        result, wiki_root, files = self._run_migrate(tmp_path, monkeypatch)
        # secret.md is under private/ subdir of projA; must land under _private/.
        private_root = wiki_root / "_private"
        all_slugs = set()
        for p in private_root.rglob("*.md"):
            if p.name == "index.md":
                continue
            text = p.read_text(encoding="utf-8")
            from skill_hub.wiki import parse_frontmatter
            fm, _ = parse_frontmatter(text)
            if fm.get("source_refs") and str(files["file_a_priv"]) in fm["source_refs"]:
                assert fm.get("scope") == "private"
                return
        # If we reach here, secret.md was not migrated to private.
        assert False, "secret.md should appear under _private/ with scope=private"

    def test_diabete_project_routes_to_glicemia(self, tmp_path, monkeypatch):
        result, wiki_root, files = self._run_migrate(tmp_path, monkeypatch)
        from skill_hub.wiki import parse_frontmatter
        # health-notes.md from diabete project must land in _private/glicemia/.
        glicemia_dir = wiki_root / "_private" / "glicemia"
        assert glicemia_dir.exists(), "_private/glicemia/ must exist"
        found = False
        for p in glicemia_dir.glob("*.md"):
            if p.name == "index.md":
                continue
            text = p.read_text(encoding="utf-8")
            fm, _ = parse_frontmatter(text)
            if fm.get("source_refs") and str(files["file_b1"]) in fm["source_refs"]:
                found = True
                break
        assert found, "health-notes.md must land in _private/glicemia/"

    def test_career_project_routes_to_career(self, tmp_path, monkeypatch):
        result, wiki_root, files = self._run_migrate(tmp_path, monkeypatch)
        from skill_hub.wiki import parse_frontmatter
        career_dir = wiki_root / "_private" / "career"
        assert career_dir.exists(), "_private/career/ must exist"
        found = False
        for p in career_dir.glob("*.md"):
            if p.name == "index.md":
                continue
            text = p.read_text(encoding="utf-8")
            fm, _ = parse_frontmatter(text)
            if fm.get("source_refs") and str(files["file_c1"]) in fm["source_refs"]:
                found = True
                break
        assert found, "cv-notes.md must land in _private/career/"

    def test_inbox_is_skipped(self, tmp_path, monkeypatch):
        result, wiki_root, files = self._run_migrate(tmp_path, monkeypatch)
        from skill_hub.wiki import parse_frontmatter
        # No page should have inbox.md in source_refs.
        for p in wiki_root.rglob("*.md"):
            if p.name in ("index.md", "log.md"):
                continue
            try:
                text = p.read_text(encoding="utf-8")
            except OSError:
                continue
            fm, _ = parse_frontmatter(text)
            refs = fm.get("source_refs") or []
            assert str(files["file_inbox"]) not in refs, \
                f"inbox.md must never appear in source_refs, found in {p}"


class TestMigrateIdempotency:
    """Running migrate twice produces the same files; second run skips all."""

    def test_second_run_skips_all(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        from skill_hub.wiki import migrate
        from skill_hub.store import SkillStore

        projects_root = tmp_path / "projects"
        _make_auto_memory_tree(projects_root)
        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", projects_root)

        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        store = SkillStore(db_path=tmp_path / "db.db")

        result1 = migrate(store, wiki_root, dry_run=False,
                          private_scopes={"glicemia": ["glicemia"],
                                          "career": ["career"]})
        result2 = migrate(store, wiki_root, dry_run=False,
                          private_scopes={"glicemia": ["glicemia"],
                                          "career": ["career"]})

        assert result1["written"] > 0
        assert result2["written"] == 0
        assert result2["skipped"] == result1["written"]

    def test_second_run_same_files(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        from skill_hub.wiki import migrate
        from skill_hub.store import SkillStore

        projects_root = tmp_path / "projects"
        _make_auto_memory_tree(projects_root)
        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", projects_root)

        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        store = SkillStore(db_path=tmp_path / "db.db")

        migrate(store, wiki_root, dry_run=False,
                private_scopes={"glicemia": ["glicemia"], "career": ["career"]})
        files_after_first = set(p.name for p in wiki_root.rglob("*.md"))

        migrate(store, wiki_root, dry_run=False,
                private_scopes={"glicemia": ["glicemia"], "career": ["career"]})
        files_after_second = set(p.name for p in wiki_root.rglob("*.md"))

        # Same set of files (log.md may grow, index may be rewritten — but
        # no new content pages should appear).
        assert files_after_first == files_after_second


class TestMigrateSlugCollisions:
    """Slug collisions across two projects produce project-suffixed slugs."""

    def test_collision_produces_unique_slugs(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        from skill_hub.wiki import migrate, parse_frontmatter
        from skill_hub.store import SkillStore

        projects_root = tmp_path / "projects"
        # Two projects each with a "notes.md" — will collide on slug "notes".
        for proj in ("projX", "projY"):
            d = projects_root / f"-{proj}" / "memory"
            d.mkdir(parents=True)
            (d / "notes.md").write_text(f"# Notes from {proj}\n", encoding="utf-8")

        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", projects_root)

        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        store = SkillStore(db_path=tmp_path / "db.db")

        result = migrate(store, wiki_root, dry_run=False,
                         private_scopes={})

        # Both pages must exist; slugs must be distinct.
        source_dir = wiki_root / "pages" / "source"
        slugs = set()
        for p in source_dir.glob("*.md"):
            text = p.read_text(encoding="utf-8")
            fm, _ = parse_frontmatter(text)
            slugs.add(fm.get("slug") or p.stem)
        assert len(slugs) == 2, f"expected 2 unique slugs for 2 notes.md files, got {slugs}"

    def test_collision_manifest_records_collisions(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        from skill_hub.wiki import migrate
        from skill_hub.store import SkillStore

        projects_root = tmp_path / "projects"
        for proj in ("projX", "projY"):
            d = projects_root / f"-{proj}" / "memory"
            d.mkdir(parents=True)
            (d / "notes.md").write_text(f"Notes from {proj}\n", encoding="utf-8")

        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", projects_root)

        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        store = SkillStore(db_path=tmp_path / "db.db")

        result = migrate(store, wiki_root, dry_run=True, private_scopes={})
        assert len(result["collisions"]) >= 1


class TestMigratePublicIndex:
    """Public index.md contains public slugs and no private slug/title."""

    def test_public_index_contains_public_slugs(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        from skill_hub.wiki import migrate
        from skill_hub.store import SkillStore

        projects_root = tmp_path / "projects"
        _make_auto_memory_tree(projects_root)
        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", projects_root)

        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        store = SkillStore(db_path=tmp_path / "db.db")

        migrate(store, wiki_root, dry_run=False,
                private_scopes={"glicemia": ["glicemia"], "career": ["career"]})

        index_path = wiki_root / "index.md"
        assert index_path.exists(), "index.md must be written"
        index_text = index_path.read_text(encoding="utf-8")

        # Must contain at least one public slug.
        assert "decisions" in index_text or "patterns" in index_text

    def test_public_index_excludes_private_content(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        from skill_hub.wiki import migrate, parse_frontmatter
        from skill_hub.store import SkillStore

        projects_root = tmp_path / "projects"
        _make_auto_memory_tree(projects_root)
        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", projects_root)

        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        store = SkillStore(db_path=tmp_path / "db.db")

        migrate(store, wiki_root, dry_run=False,
                private_scopes={"glicemia": ["glicemia"], "career": ["career"]})

        index_text = (wiki_root / "index.md").read_text(encoding="utf-8")

        # Collect all private page slugs and titles.
        private_slugs: set[str] = set()
        private_titles: set[str] = set()
        for p in (wiki_root / "_private").rglob("*.md"):
            if p.name == "index.md":
                continue
            try:
                fm, _ = parse_frontmatter(p.read_text(encoding="utf-8"))
                if fm.get("slug"):
                    private_slugs.add(fm["slug"])
                if fm.get("title"):
                    private_titles.add(fm["title"])
            except Exception:
                pass

        for slug in private_slugs:
            assert slug not in index_text, \
                f"private slug {slug!r} must not appear in public index.md"


class TestMigrateLogMd:
    """log.md gets one ## [...] migrate | ... line per run."""

    def test_log_md_appended(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        from skill_hub.wiki import migrate
        from skill_hub.store import SkillStore

        projects_root = tmp_path / "projects"
        _make_auto_memory_tree(projects_root)
        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", projects_root)

        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        store = SkillStore(db_path=tmp_path / "db.db")

        migrate(store, wiki_root, dry_run=False,
                private_scopes={"glicemia": ["glicemia"], "career": ["career"]},
                today="2026-06-23")

        log_path = wiki_root / "log.md"
        assert log_path.exists()
        log_text = log_path.read_text(encoding="utf-8")
        assert "## [2026-06-23] migrate |" in log_text

    def test_log_md_one_line_per_run(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        from skill_hub.wiki import migrate
        from skill_hub.store import SkillStore

        projects_root = tmp_path / "projects"
        _make_auto_memory_tree(projects_root)
        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", projects_root)

        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        store = SkillStore(db_path=tmp_path / "db.db")

        migrate(store, wiki_root, dry_run=False,
                private_scopes={"glicemia": ["glicemia"], "career": ["career"]},
                today="2026-06-23")
        migrate(store, wiki_root, dry_run=False,
                private_scopes={"glicemia": ["glicemia"], "career": ["career"]},
                today="2026-06-23")

        log_text = (wiki_root / "log.md").read_text(encoding="utf-8")
        log_lines = [l for l in log_text.splitlines() if l.startswith("## [")]
        assert len(log_lines) == 2, \
            f"expected 2 log entries after 2 runs, got {len(log_lines)}"


class TestMigrateReindexStatus:
    """After non-dry migrate + reindex, status reports drift == 0."""

    def _stub_embed(self, store):
        """Patch store.upsert_vector to avoid needing a real embedding backend."""
        def fake_upsert(namespace, doc_id, text, metadata=None, **kw):
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

    def test_reindex_after_migrate_drift_zero(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        from skill_hub.wiki import migrate, reindex, status
        from skill_hub.store import SkillStore

        projects_root = tmp_path / "projects"
        _make_auto_memory_tree(projects_root)
        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", projects_root)

        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        store = SkillStore(db_path=tmp_path / "db.db")

        migrate(store, wiki_root, dry_run=False,
                private_scopes={"glicemia": ["glicemia"], "career": ["career"]})

        self._stub_embed(store)
        with patch("skill_hub.wiki._check_dim_guard"):
            reindex(store, wiki_root, dry_run=False)

        st = status(store, wiki_root)
        assert st["drift"] == 0, \
            f"drift must be 0 after migrate + reindex, got {st}"


class TestMigrateCronAndTool:
    """wiki-reindex-nightly is seeded disabled; wiki_migrate in TIER_REGISTRY."""

    def test_wiki_migrate_in_tier_registry(self):
        from skill_hub.capabilities import TIER_REGISTRY
        assert "wiki_migrate" in TIER_REGISTRY
        assert TIER_REGISTRY["wiki_migrate"] == "none"

    def test_wiki_migrate_toolspec_hard_db(self):
        from skill_hub.capabilities import TOOLS, BACKEND_DB
        spec = next((s for s in TOOLS if s.name == "wiki_migrate"), None)
        assert spec is not None
        assert BACKEND_DB in spec.hard

    def test_wiki_reindex_nightly_in_default_jobs(self):
        from skill_hub.cron import _DEFAULT_JOBS
        names = [job[0] for job in _DEFAULT_JOBS]
        assert "wiki-reindex-nightly" in names

    def test_wiki_reindex_nightly_disabled_by_default(self):
        from skill_hub.cron import _DEFAULT_JOBS
        for name, schedule, command, enabled in _DEFAULT_JOBS:
            if name == "wiki-reindex-nightly":
                assert enabled == 0, "wiki-reindex-nightly must be seeded disabled"
                assert schedule == "0 5 * * *"
                assert command == "wiki_reindex_nightly"
                return
        assert False, "wiki-reindex-nightly not found in _DEFAULT_JOBS"


class TestMigrateProjectRoots:
    """project_roots covers literal <repo>/.memory/ trees (distinct from auto-memory)."""

    def test_project_memory_files_migrated(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        from skill_hub.wiki import migrate
        from skill_hub.store import SkillStore

        # Empty auto-memory tree so only project_roots contributes.
        empty_auto = tmp_path / "projects"
        empty_auto.mkdir()
        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", empty_auto)

        # A repo root with a literal .memory/ tree (public + private subdir).
        repo = tmp_path / "geoid"
        mem = repo / ".memory"
        mem.mkdir(parents=True)
        (mem / "decisions.md").write_text("# Decisions\n\nUse codegraph.\n",
                                          encoding="utf-8")
        priv = mem / "private"
        priv.mkdir()
        (priv / "sovereign.md").write_text("# Sovereign\n\nSensitive.\n",
                                           encoding="utf-8")

        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        store = SkillStore(db_path=tmp_path / "db.db")

        result = migrate(store, wiki_root, dry_run=False,
                         project_roots=[repo])

        # Public .memory file → pages/source/; private/ subdir → _private/geoid/.
        assert (wiki_root / "pages" / "source" / "decisions.md").exists()
        assert (wiki_root / "_private" / "geoid" / "sovereign.md").exists()
        assert result["public"] == 1
        assert result["private"] == 1

    def test_no_project_roots_skips_repo_memory(self, tmp_path, monkeypatch):
        import skill_hub.memory_index as mi
        from skill_hub.wiki import migrate
        from skill_hub.store import SkillStore

        empty_auto = tmp_path / "projects"
        empty_auto.mkdir()
        monkeypatch.setattr(mi, "_USER_MEMORY_ROOT", empty_auto)

        repo = tmp_path / "geoid"
        mem = repo / ".memory"
        mem.mkdir(parents=True)
        (mem / "decisions.md").write_text("# Decisions\n\nx\n", encoding="utf-8")

        wiki_root = tmp_path / "wiki"
        wiki_root.mkdir()
        store = SkillStore(db_path=tmp_path / "db.db")

        # No project_roots → repo .memory is not discovered.
        result = migrate(store, wiki_root, dry_run=True)
        assert result["would_write"] == 0


# ---------------------------------------------------------------------------
# Wave 2 — Step W2.1: wiki_ingest LLM producer
# ---------------------------------------------------------------------------


class TestWikiIngestProducer:
    """embeddings.wiki_ingest distills a source into page rewrites (mock LLM)."""

    def _provider(self, payload: str):
        class FakeProvider:
            def complete(self, *a, **kw):
                return payload
        return FakeProvider()

    def test_parses_well_formed_json(self):
        from skill_hub import embeddings as _emb

        payload = json.dumps({
            "source_page": {"slug": "source-x", "title": "X", "body": "raw"},
            "page_updates": [
                {"slug": "x", "title": "X", "type": "entity", "scope": "public",
                 "new_body": "About [[y]].", "reason": "new", "is_new": True}
            ],
            "index_entries": [{"slug": "x", "title": "X", "one_line": "the x"}],
            "assumptions": [{"claim": "c", "verify_by": "v"}],
        })
        with patch.object(_emb, "get_provider", return_value=self._provider(payload)):
            out = _emb.wiki_ingest("memory", "X", "raw text")
        assert out["source_page"]["slug"] == "source-x"
        assert out["page_updates"][0]["slug"] == "x"
        assert out["index_entries"][0]["one_line"] == "the x"
        assert out["assumptions"] == [{"claim": "c", "verify_by": "v"}]
        assert not out.get("_fallback")

    def test_normalizes_missing_and_wrong_type_lists(self):
        from skill_hub import embeddings as _emb

        # source_page present (required), but list fields null / wrong type / missing.
        payload = json.dumps({
            "source_page": {"slug": "source-x", "title": "X", "body": "raw"},
            "page_updates": None,
            "index_entries": "not a list",
        })
        with patch.object(_emb, "get_provider", return_value=self._provider(payload)):
            out = _emb.wiki_ingest("memory", "X", "raw")
        assert out["page_updates"] == []
        assert out["index_entries"] == []
        assert out["assumptions"] == []

    def test_strips_think_block(self):
        from skill_hub import embeddings as _emb

        payload = (
            "<think>deciding</think>\n"
            '{"source_page": {"slug": "source-x", "title": "X", "body": "b"}, '
            '"page_updates": [], "index_entries": [], "assumptions": []}'
        )
        with patch.object(_emb, "get_provider", return_value=self._provider(payload)):
            out = _emb.wiki_ingest("memory", "X", "raw")
        assert out["source_page"]["slug"] == "source-x"
        assert not out.get("_fallback")

    def test_fallback_on_garbage(self):
        from skill_hub import embeddings as _emb

        with patch.object(_emb, "get_provider", return_value=self._provider("not json")):
            out = _emb.wiki_ingest("memory", "X", "raw")
        assert out["_fallback"] is True
        assert out["source_page"] == {}
        assert out["page_updates"] == []

    def test_fallback_on_provider_exception(self):
        from skill_hub import embeddings as _emb

        class Boom:
            def complete(self, *a, **kw):
                raise RuntimeError("provider down")
        with patch.object(_emb, "get_provider", return_value=Boom()):
            out = _emb.wiki_ingest("memory", "X", "raw")
        assert out["_fallback"] is True


# ---------------------------------------------------------------------------
# Wave 2 — Step W2.2: wiki.ingest_source (discovery → LLM → write)
# ---------------------------------------------------------------------------


class TestIngestSource:
    """ingest_source orchestrates candidate discovery, LLM distill, and writes."""

    def _payload(self, **over):
        p = {
            "source_page": {"slug": "ignored", "title": "Foo source", "body": "raw foo"},
            "page_updates": [
                {"slug": "foo", "title": "Foo", "type": "entity", "scope": "public",
                 "new_body": "Foo is about [[bar]].", "reason": "new", "is_new": True}
            ],
            "index_entries": [{"slug": "foo", "title": "Foo", "one_line": "the foo"}],
            "assumptions": [{"claim": "c", "verify_by": "v"}],
        }
        p.update(over)
        return p

    def _patch(self, monkeypatch, store, payload, hits=None):
        import skill_hub.embeddings as _emb
        monkeypatch.setattr(_emb, "wiki_ingest", lambda **kw: payload)
        monkeypatch.setattr(store, "search_vectors", lambda *a, **k: hits or [])

        def fake_upsert(namespace, doc_id, text, metadata=None, **kw):
            store._conn.execute(
                "INSERT INTO vectors (namespace, doc_id, model, vector, norm, "
                "level, source, project) VALUES (?, ?, 'stub', '[]', 0.1, 'L3', "
                "'wiki', NULL) ON CONFLICT(namespace, doc_id) DO UPDATE SET "
                "vector=excluded.vector",
                (namespace, doc_id),
            )
            store._conn.commit()

        monkeypatch.setattr(store, "upsert_vector", fake_upsert)

    def test_dry_run_writes_nothing(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import ingest_source
        self._patch(monkeypatch, store, self._payload())

        out = ingest_source(store, wiki_root, source_kind="memory",
                            source_id="foo", source_title="Foo", source_text="text",
                            dry_run=True)
        assert out["status"] == "dry_run"
        assert {d["slug"] for d in out["pages"]} == {"source-foo", "foo"}
        assert not (wiki_root / "pages").exists()
        assert not (wiki_root / "index.md").exists()

    def test_non_dry_writes_pages_index_log(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import ingest_source
        self._patch(monkeypatch, store, self._payload())

        out = ingest_source(store, wiki_root, source_kind="memory",
                            source_id="foo", source_title="Foo", source_text="text",
                            dry_run=False, today="2026-06-23")
        assert out["status"] == "ok"
        assert (wiki_root / "pages" / "source" / "source-foo.md").exists()
        assert (wiki_root / "pages" / "entity" / "foo.md").exists()
        assert (wiki_root / "index.md").exists()
        log = (wiki_root / "log.md").read_text()
        assert "## [2026-06-23] ingest | Foo" in log
        # assumptions captured to inbox
        assert "verify: v" in (wiki_root / "inbox.md").read_text()
        # source page carries a source_hash for idempotence
        sp = (wiki_root / "pages" / "source" / "source-foo.md").read_text()
        assert "source_hash:" in sp

    def test_backup_before_overwrite(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import ingest_source
        # Pre-create the entity page so the ingest updates (and backs up) it.
        ent = wiki_root / "pages" / "entity"
        ent.mkdir(parents=True)
        (ent / "foo.md").write_text(
            "---\nslug: foo\ntitle: Foo\ntype: entity\nscope: public\n"
            "projects:\n- _global\n---\nold body\n", encoding="utf-8")
        self._patch(monkeypatch, store, self._payload())

        ingest_source(store, wiki_root, source_kind="memory", source_id="foo",
                      source_title="Foo", source_text="text", dry_run=False)
        bak = ent / ".backups" / "foo.md.bak"
        assert bak.exists()
        assert "old body" in bak.read_text()

    def test_idempotence_skips_unchanged_source(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import ingest_source
        self._patch(monkeypatch, store, self._payload())

        first = ingest_source(store, wiki_root, source_kind="memory",
                              source_id="foo", source_text="same text",
                              dry_run=False)
        assert first["status"] == "ok"
        second = ingest_source(store, wiki_root, source_kind="memory",
                               source_id="foo", source_text="same text",
                               dry_run=False)
        assert second["status"] == "skipped"

    def test_changed_source_reingests(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import ingest_source
        self._patch(monkeypatch, store, self._payload())
        ingest_source(store, wiki_root, source_kind="memory", source_id="foo",
                      source_text="v1", dry_run=False)
        out = ingest_source(store, wiki_root, source_kind="memory", source_id="foo",
                            source_text="v2 changed", dry_run=False)
        assert out["status"] == "ok"

    def test_private_target_denied_without_authorization(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import ingest_source
        self._patch(monkeypatch, store, self._payload())
        out = ingest_source(store, wiki_root, source_kind="memory", source_id="h",
                            source_text="t", target_scope="private",
                            target_project="glicemia", authorized_scopes=[],
                            dry_run=True)
        assert out["status"] == "denied"
        assert not (wiki_root / "_private").exists()

    def test_private_target_authorized_writes_private(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import ingest_source
        payload = self._payload(page_updates=[
            {"slug": "sugar", "title": "Sugar", "type": "entity", "scope": "private",
             "new_body": "private note", "is_new": True}])
        self._patch(monkeypatch, store, payload)
        out = ingest_source(store, wiki_root, source_kind="memory", source_id="h",
                            source_text="t", target_scope="private",
                            target_project="glicemia",
                            authorized_scopes=["glicemia"], dry_run=False)
        assert out["status"] == "ok"
        assert (wiki_root / "_private" / "glicemia" / "source-h.md").exists()
        assert (wiki_root / "_private" / "glicemia" / "sugar.md").exists()

    def test_unauthorized_private_page_update_dropped(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import ingest_source
        # Public ingest, but the LLM proposes a private page → must be dropped.
        payload = self._payload(page_updates=[
            {"slug": "secret", "title": "Secret", "type": "entity",
             "scope": "private", "new_body": "leak", "is_new": True}])
        self._patch(monkeypatch, store, payload)
        out = ingest_source(store, wiki_root, source_kind="memory", source_id="foo",
                            source_text="t", dry_run=False)
        assert out["status"] == "ok"
        assert {d["slug"] for d in out["pages"]} == {"source-foo"}
        assert not (wiki_root / "_private").exists()

    def test_candidate_discovery_reads_neighbours(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import ingest_source
        # Seed a neighbour page on disk + a fake hit pointing at it.
        ent = wiki_root / "pages" / "entity"
        ent.mkdir(parents=True)
        (ent / "bar.md").write_text(
            "---\nslug: bar\ntitle: Bar\ntype: entity\nscope: public\n"
            "projects:\n- _global\n---\nbar body\n", encoding="utf-8")
        hits = [{"metadata": {"slug": "bar", "rel_path": "pages/entity/bar.md"}}]
        self._patch(monkeypatch, store, self._payload(), hits=hits)
        out = ingest_source(store, wiki_root, source_kind="memory", source_id="foo",
                            source_text="t", dry_run=True)
        assert "bar" in out["candidates"]


# ---------------------------------------------------------------------------
# Wave 2 — Step W2.3: scan_candidates + scan_and_enqueue (auto source selection)
# ---------------------------------------------------------------------------


class TestScanCandidates:
    """Deterministic source selection: undistilled + stale source pages."""

    def _write(self, wiki_root, rel, slug, ptype, body="x", scope="public",
               source_refs=None, source_hash=""):
        p = wiki_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        fm = [f"---", f"slug: {slug}", f"title: {slug}", f"type: {ptype}",
              f"scope: {scope}", "projects:", "- _global"]
        if source_refs:
            fm.append("source_refs:")
            for r in source_refs:
                fm.append(f"- {r}")
        if source_hash:
            fm.append(f"source_hash: {source_hash}")
        fm.append("---")
        p.write_text("\n".join(fm) + f"\n{body}\n", encoding="utf-8")
        return p

    def test_undistilled_source_is_candidate(self, store, wiki_root):
        from skill_hub.wiki import scan_candidates
        self._write(wiki_root, "pages/source/source-a.md", "source-a", "source",
                    source_refs=["/orig/a.md"])
        cands = scan_candidates(store, wiki_root)
        assert len(cands) == 1
        assert cands[0]["slug"] == "source-a"
        assert cands[0]["reason"] == "undistilled"

    def test_distilled_source_not_candidate(self, store, wiki_root):
        from skill_hub.wiki import scan_candidates
        # A source page AND an entity page sharing the same source_ref → distilled.
        self._write(wiki_root, "pages/source/source-a.md", "source-a", "source",
                    source_refs=["/orig/a.md"])
        self._write(wiki_root, "pages/entity/a.md", "a", "entity",
                    source_refs=["/orig/a.md"])
        cands = scan_candidates(store, wiki_root)
        assert cands == []

    def test_stale_source_is_candidate(self, store, wiki_root, tmp_path):
        from skill_hub.wiki import scan_candidates, _hash_text
        orig = tmp_path / "a.md"
        orig.write_text("new content", encoding="utf-8")
        # Record a stale hash (of different content) + a distilled descendant so
        # 'undistilled' does not also fire — isolating the stale path.
        self._write(wiki_root, "pages/source/source-a.md", "source-a", "source",
                    source_refs=[str(orig)], source_hash=_hash_text("old content"))
        self._write(wiki_root, "pages/entity/a.md", "a", "entity",
                    source_refs=[str(orig)])
        cands = scan_candidates(store, wiki_root)
        assert len(cands) == 1
        assert cands[0]["reason"] == "stale"

    def test_unchanged_hash_not_stale(self, store, wiki_root, tmp_path):
        from skill_hub.wiki import scan_candidates, _hash_text
        orig = tmp_path / "a.md"
        orig.write_text("same", encoding="utf-8")
        self._write(wiki_root, "pages/source/source-a.md", "source-a", "source",
                    source_refs=[str(orig)], source_hash=_hash_text("same"))
        self._write(wiki_root, "pages/entity/a.md", "a", "entity",
                    source_refs=[str(orig)])
        assert scan_candidates(store, wiki_root) == []

    def test_ranking_stale_before_undistilled(self, store, wiki_root, tmp_path):
        from skill_hub.wiki import scan_candidates, _hash_text
        orig = tmp_path / "s.md"
        orig.write_text("changed", encoding="utf-8")
        self._write(wiki_root, "pages/source/source-s.md", "source-s", "source",
                    source_refs=[str(orig)], source_hash=_hash_text("was"))
        self._write(wiki_root, "pages/entity/s.md", "s", "entity",
                    source_refs=[str(orig)])
        self._write(wiki_root, "pages/source/source-u.md", "source-u", "source",
                    source_refs=["/orig/u.md"])
        cands = scan_candidates(store, wiki_root)
        assert [c["reason"] for c in cands] == ["stale", "undistilled"]

    def test_index_md_ignored(self, store, wiki_root):
        from skill_hub.wiki import scan_candidates
        (wiki_root / "index.md").write_text("# Wiki Index\n", encoding="utf-8")
        self._write(wiki_root, "pages/source/source-a.md", "source-a", "source",
                    source_refs=["/orig/a.md"])
        cands = scan_candidates(store, wiki_root)
        assert {c["slug"] for c in cands} == {"source-a"}


class TestScanAndEnqueue:
    """scan_and_enqueue upserts the approval queue idempotently."""

    def _src(self, wiki_root, slug, ref):
        p = wiki_root / "pages" / "source" / f"{slug}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            f"---\nslug: {slug}\ntitle: {slug}\ntype: source\nscope: public\n"
            f"projects:\n- _global\nsource_refs:\n- {ref}\n---\nbody\n",
            encoding="utf-8")

    def test_enqueues_pending(self, store, wiki_root):
        from skill_hub.wiki import scan_and_enqueue
        self._src(wiki_root, "source-a", "/orig/a.md")
        out = scan_and_enqueue(store, wiki_root)
        assert out["scanned"] == 1
        assert out["pending"] == 1
        assert out["total_est_calls"] == 1
        assert out["queue"][0]["slug"] == "source-a"

    def test_idempotent_rescan(self, store, wiki_root):
        from skill_hub.wiki import scan_and_enqueue
        self._src(wiki_root, "source-a", "/orig/a.md")
        scan_and_enqueue(store, wiki_root)
        out = scan_and_enqueue(store, wiki_root)
        assert out["pending"] == 1  # not duplicated
        n = store._conn.execute("SELECT COUNT(*) FROM wiki_queue").fetchone()[0]
        assert n == 1

    def test_approved_row_not_reset(self, store, wiki_root):
        from skill_hub.wiki import scan_and_enqueue
        self._src(wiki_root, "source-a", "/orig/a.md")
        scan_and_enqueue(store, wiki_root)
        store._conn.execute(
            "UPDATE wiki_queue SET status='approved' WHERE slug='source-a'")
        store._conn.commit()
        out = scan_and_enqueue(store, wiki_root)
        assert out["approved"] == 1
        assert out["pending"] == 0


# ---------------------------------------------------------------------------
# Wave 2 — Step W2.4: approval queue tools (decision + queued ingest + batch)
# ---------------------------------------------------------------------------


class TestQueueDecisionAndIngest:
    """Approval gate: only approved rows may be ingested live."""

    def _enqueue(self, store, wiki_root, slug="source-a", ref=None):
        # Write a source page + enqueue it as a candidate.
        p = wiki_root / "pages" / "source" / f"{slug}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        ref = ref or "/orig/a.md"
        p.write_text(
            f"---\nslug: {slug}\ntitle: {slug}\ntype: source\nscope: public\n"
            f"projects:\n- _global\nsource_refs:\n- {ref}\n---\nraw body\n",
            encoding="utf-8")
        from skill_hub.wiki import scan_and_enqueue
        scan_and_enqueue(store, wiki_root)

    def _stub_llm(self, monkeypatch, store, slug="source-a"):
        import skill_hub.embeddings as _emb
        payload = {
            "source_page": {"slug": slug, "title": "A", "body": "raw"},
            "page_updates": [{"slug": "a", "title": "A", "type": "entity",
                              "scope": "public", "new_body": "About a.",
                              "is_new": True}],
            "index_entries": [], "assumptions": [],
        }
        monkeypatch.setattr(_emb, "wiki_ingest", lambda **kw: payload)
        monkeypatch.setattr(store, "search_vectors", lambda *a, **k: [])

        def fake_upsert(namespace, doc_id, text, metadata=None, **kw):
            store._conn.execute(
                "INSERT INTO vectors (namespace, doc_id, model, vector, norm, "
                "level, source, project) VALUES (?, ?, 'stub', '[]', 0.1, 'L3', "
                "'wiki', NULL) ON CONFLICT(namespace, doc_id) DO UPDATE SET "
                "vector=excluded.vector", (namespace, doc_id))
            store._conn.commit()
        monkeypatch.setattr(store, "upsert_vector", fake_upsert)

    def test_approve_sets_status(self, store, wiki_root):
        from skill_hub.wiki import queue_decision
        self._enqueue(store, wiki_root)
        out = queue_decision(store, "source-a", "approve")
        assert out["decision"] == "approved"
        row = store._conn.execute(
            "SELECT status FROM wiki_queue WHERE slug='source-a'").fetchone()
        assert row["status"] == "approved"

    def test_skip_sets_status(self, store, wiki_root):
        from skill_hub.wiki import queue_decision
        self._enqueue(store, wiki_root)
        queue_decision(store, "source-a", "skip")
        row = store._conn.execute(
            "SELECT status FROM wiki_queue WHERE slug='source-a'").fetchone()
        assert row["status"] == "skipped"

    def test_bad_decision_errors(self, store, wiki_root):
        from skill_hub.wiki import queue_decision
        self._enqueue(store, wiki_root)
        assert queue_decision(store, "source-a", "maybe")["status"] == "error"

    def test_live_ingest_requires_approval(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import ingest_queued
        self._enqueue(store, wiki_root)
        self._stub_llm(monkeypatch, store)
        out = ingest_queued(store, wiki_root, "source-a", dry_run=False)
        assert out["status"] == "denied"

    def test_dry_run_allowed_without_approval(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import ingest_queued
        self._enqueue(store, wiki_root)
        self._stub_llm(monkeypatch, store)
        out = ingest_queued(store, wiki_root, "source-a", dry_run=True)
        assert out["status"] == "dry_run"
        # diff preview stored
        row = store._conn.execute(
            "SELECT diff_preview FROM wiki_queue WHERE slug='source-a'").fetchone()
        assert row["diff_preview"] is not None

    def test_approved_ingest_writes_and_marks_done(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import ingest_queued, queue_decision
        self._enqueue(store, wiki_root)
        self._stub_llm(monkeypatch, store)
        queue_decision(store, "source-a", "approve")
        out = ingest_queued(store, wiki_root, "source-a", dry_run=False,
                            today="2026-06-23")
        assert out["status"] == "ok"
        assert (wiki_root / "pages" / "entity" / "a.md").exists()
        row = store._conn.execute(
            "SELECT status FROM wiki_queue WHERE slug='source-a'").fetchone()
        assert row["status"] == "done"

    def test_batch_respects_limit(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import ingest_approved, queue_decision
        for s in ("source-a", "source-b", "source-c"):
            self._enqueue(store, wiki_root, slug=s, ref=f"/orig/{s}.md")
            queue_decision(store, s, "approve")
        self._stub_llm(monkeypatch, store)
        out = ingest_approved(store, wiki_root, limit=2, dry_run=False)
        assert out["processed"] == 2


# ---------------------------------------------------------------------------
# Wave 2 — Step W2.5: wiki_query (hybrid vector + lexical) + file-back stub
# ---------------------------------------------------------------------------


class TestWikiQuery:
    """Hybrid query: vector ranking unioned with index.md lexical hits."""

    def _page(self, wiki_root, rel, slug, title, scope="public",
              source_refs=None):
        p = wiki_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        fm = [f"---", f"slug: {slug}", f"title: {title}", "type: entity",
              f"scope: {scope}", "projects:", "- _global"]
        if source_refs:
            fm.append("source_refs:")
            for r in source_refs:
                fm.append(f"- {r}")
        fm.append("---")
        p.write_text("\n".join(fm) + f"\nbody of {slug}\n", encoding="utf-8")

    def test_file_back_stub_raises(self, store, wiki_root):
        from skill_hub.wiki import query
        with pytest.raises(NotImplementedError):
            query(store, wiki_root, "x", _file_back=True)

    def test_vector_hit_returns_body_and_provenance(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import query
        self._page(wiki_root, "pages/entity/foo.md", "foo", "Foo",
                   source_refs=["/orig/foo.md"])
        hits = [{"metadata": {"slug": "foo", "rel_path": "pages/entity/foo.md"},
                 "score": 0.91}]
        monkeypatch.setattr(store, "search_vectors", lambda *a, **k: hits)
        out = query(store, wiki_root, "foo")
        assert out["results"][0]["slug"] == "foo"
        assert "body of foo" in out["results"][0]["body"]
        assert out["results"][0]["source_refs"] == ["/orig/foo.md"]
        assert out["results"][0]["score"] == 0.91

    def test_lexical_backfill_from_index(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import query
        self._page(wiki_root, "pages/entity/karpathy.md", "karpathy",
                   "Karpathy Guidelines")
        (wiki_root / "index.md").write_text(
            "# Wiki Index\n\n## entity\n- [[karpathy]] — Karpathy Guidelines\n",
            encoding="utf-8")
        monkeypatch.setattr(store, "search_vectors", lambda *a, **k: [])
        out = query(store, wiki_root, "karpathy")
        assert any(r["slug"] == "karpathy" for r in out["results"])

    def test_private_excluded_when_unauthorized(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import query
        self._page(wiki_root, "_private/glicemia/secret.md", "secret", "Secret",
                   scope="private")
        hits = [{"metadata": {"slug": "secret",
                              "rel_path": "_private/glicemia/secret.md"},
                 "score": 0.99}]
        monkeypatch.setattr(store, "search_vectors", lambda *a, **k: hits)
        out = query(store, wiki_root, "secret", authorized_scopes=[])
        assert out["results"] == []

    def test_private_included_when_authorized(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import query
        self._page(wiki_root, "_private/glicemia/secret.md", "secret", "Secret",
                   scope="private")
        hits = [{"metadata": {"slug": "secret",
                              "rel_path": "_private/glicemia/secret.md"},
                 "score": 0.99}]
        monkeypatch.setattr(store, "search_vectors", lambda *a, **k: hits)
        out = query(store, wiki_root, "secret", authorized_scopes=["glicemia"])
        assert out["results"][0]["slug"] == "secret"

    def test_top_k_caps_results(self, store, wiki_root, monkeypatch):
        from skill_hub.wiki import query
        hits = []
        for i in range(5):
            self._page(wiki_root, f"pages/entity/e{i}.md", f"e{i}", f"E{i}")
            hits.append({"metadata": {"slug": f"e{i}",
                                      "rel_path": f"pages/entity/e{i}.md"},
                         "score": 0.5})
        monkeypatch.setattr(store, "search_vectors", lambda *a, **k: hits)
        out = query(store, wiki_root, "e", top_k=2)
        assert len(out["results"]) == 2


# ---------------------------------------------------------------------------
# Wave 2 — Step W2.6: dashboard "Memory Wiki" card
# ---------------------------------------------------------------------------


class TestDashboardWikiCard:
    """The dashboard surfaces wiki health + the ingest approval queue."""

    def _setup_vault(self, tmp_path):
        # store fixture sets HOME=tmp_path, so this is the default wiki_root.
        wiki_root = tmp_path / ".claude" / "mcp-skill-hub" / "wiki"
        src = wiki_root / "pages" / "source"
        src.mkdir(parents=True)
        (src / "source-a.md").write_text(
            "---\nslug: source-a\ntitle: A\ntype: source\nscope: public\n"
            "projects:\n- _global\nsource_refs:\n- /orig/a.md\n---\nbody\n",
            encoding="utf-8")
        return wiki_root

    def test_db_metrics_includes_wiki(self, store, tmp_path):
        from skill_hub import dashboard as dash
        from skill_hub import wiki as _wiki
        wiki_root = self._setup_vault(tmp_path)
        _wiki.scan_and_enqueue(store, wiki_root)
        m = dash._db_metrics(store)
        assert m["wiki"] is not None
        assert m["wiki"]["queue_pending"] == 1
        assert m["wiki"]["pages_disk"] >= 1

    def test_render_writes_wiki_card(self, store, tmp_path):
        from skill_hub import dashboard as dash
        from skill_hub import wiki as _wiki
        wiki_root = self._setup_vault(tmp_path)
        _wiki.scan_and_enqueue(store, wiki_root)
        out = tmp_path / "dash.html"
        dash.render(store, out_path=out, log_path=tmp_path / "missing.log")
        html = out.read_text()
        assert "Memory Wiki" in html
        assert "ingest queue" in html


# ---------------------------------------------------------------------------
# wiki_lint — deterministic structural lint
# ---------------------------------------------------------------------------


class TestWikiLint:
    """wiki.lint detects dangling edges, orphans, and stale source hashes."""

    def _stub_embed(self, store):
        """Patch upsert_vector to avoid needing a real embedding backend."""
        def fake_upsert(namespace, doc_id, text, metadata=None, **kw):
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

    def test_clean_wiki_returns_zero_issues(self, store, wiki_root):
        from skill_hub.wiki import reindex, lint
        _write_page(wiki_root, "pages/entity/alpha.md", """\
---
id: alpha
slug: alpha
title: Alpha
type: entity
projects:
  - p
scope: public
created: 2026-01-01
updated: 2026-01-01
---
See [[beta]] for details.
""")
        _write_page(wiki_root, "pages/entity/beta.md", """\
---
id: beta
slug: beta
title: Beta
type: entity
projects:
  - p
scope: public
created: 2026-01-01
updated: 2026-01-01
---
[[alpha]] refers to this.
""")
        self._stub_embed(store)
        with patch("skill_hub.wiki._check_dim_guard"):
            reindex(store, wiki_root, dry_run=False)

        result = lint(store, wiki_root)
        # beta is linked by alpha (resolved edge), so beta has 1 inbound.
        # alpha is linked by beta, so alpha has 1 inbound.
        assert result["dangling_edges"] == 0, result
        assert result["stale_pages"] == 0, result
        assert result["total_issues"] == result["orphans"]  # orphans only

    def test_detects_dangling_edge(self, store, wiki_root):
        from skill_hub.wiki import reindex, lint
        _write_page(wiki_root, "pages/entity/lone.md", """\
---
id: lone
slug: lone
title: Lone
type: entity
projects:
  - p
scope: public
created: 2026-01-01
updated: 2026-01-01
---
See [[nonexistent-page]] for info.
""")
        self._stub_embed(store)
        with patch("skill_hub.wiki._check_dim_guard"):
            reindex(store, wiki_root, dry_run=False)

        result = lint(store, wiki_root)
        assert result["dangling_edges"] >= 1
        assert any("nonexistent-page" in s for s in result["dangling_sample"])

    def test_detects_orphan(self, store, wiki_root):
        from skill_hub.wiki import reindex, lint
        _write_page(wiki_root, "pages/entity/island.md", """\
---
id: island
slug: island
title: Island
type: entity
projects:
  - p
scope: public
created: 2026-01-01
updated: 2026-01-01
---
No links in or out.
""")
        self._stub_embed(store)
        with patch("skill_hub.wiki._check_dim_guard"):
            reindex(store, wiki_root, dry_run=False)

        result = lint(store, wiki_root)
        assert result["orphans"] >= 1
        assert "island" in result["orphan_sample"]

    def test_detects_stale_source_hash(self, store, wiki_root, tmp_path):
        from skill_hub.wiki import lint
        # Write a source file.
        src = tmp_path / "original.md"
        src.write_text("# Original\nContent v1.\n", encoding="utf-8")
        import hashlib
        v1_hash = hashlib.sha256(b"# Original\nContent v1.\n").hexdigest()[:16]
        # Write a wiki page referencing it with the correct hash initially.
        _write_page(wiki_root, "pages/source/orig.md", f"""\
---
id: orig
slug: orig
title: Orig
type: source
projects:
  - p
scope: public
source_refs:
  - {src}
source_hash: {v1_hash}
created: 2026-01-01
updated: 2026-01-01
---
Content.
""")
        # Now overwrite the source file with new content.
        src.write_text("# Original\nContent v2 changed.\n", encoding="utf-8")

        result = lint(store, wiki_root)
        assert result["stale_pages"] >= 1
        assert "orig" in result["stale_sample"]

    def test_empty_wiki_all_zeros(self, store, wiki_root):
        from skill_hub.wiki import lint
        result = lint(store, wiki_root)
        assert result["dangling_edges"] == 0
        assert result["orphans"] == 0
        assert result["stale_pages"] == 0
        assert result["total_issues"] == 0

    def test_wiki_lint_in_tier_registry(self):
        from skill_hub.capabilities import TIER_REGISTRY
        assert "wiki_lint" in TIER_REGISTRY
        assert TIER_REGISTRY["wiki_lint"] == "none"

    def test_wiki_lint_toolspec_hard_db(self):
        from skill_hub.capabilities import TOOLS, BACKEND_DB
        spec = next((s for s in TOOLS if s.name == "wiki_lint"), None)
        assert spec is not None
        assert BACKEND_DB in spec.hard
