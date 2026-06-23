"""Tests for the /control/activity endpoint.

Mirrors test_control_graphcode_route.py: TestClient + monkeypatched store so
the real database and GitHub CLI are never touched.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest
from fastapi.testclient import TestClient

from skill_hub import config as cfg_mod
from skill_hub.services import registry as reg_mod


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class _FakeStore:
    """Minimal store stub returning predictable data."""

    def list_all_issue_links(self, repo: str = "") -> list[dict]:
        return [
            {
                "id": 1,
                "task_id": 10,
                "repo": "owner/repo",
                "issue_number": 42,
                "url": "https://github.com/owner/repo/issues/42",
                "state": "open",
                "last_synced_at": "2024-01-15T10:00:00+00:00",
                "writeback_done": 0,
            }
        ]

    def get_task(self, task_id: int):
        import sqlite3
        if task_id == 10:
            # Return a minimal row-like object.
            return {"id": 10, "title": "Fix the bug", "status": "open"}
        return None

    def get_events(self, session_id: str = "", since: float = 0.0,
                   kind: str = "", limit: int = 200) -> list[dict]:
        return [
            {
                "id": 1,
                "ts": 1700000000.0,
                "kind": "task.created",
                "tool_name": "save_task",
                "session_id": "abc123xyz",
                "payload": '{"task_id": 10}',
                "node_id": "local",
            }
        ]


class _FakeStoreWithDrift:
    """Store returning a drifted link (issue closed, task open)."""

    def list_all_issue_links(self, repo: str = "") -> list[dict]:
        return [
            {
                "id": 2,
                "task_id": 20,
                "repo": "owner/repo",
                "issue_number": 7,
                "url": "https://github.com/owner/repo/issues/7",
                "state": "closed",
                "last_synced_at": None,
                "writeback_done": 0,
            }
        ]

    def get_task(self, task_id: int):
        return {"id": 20, "title": "Stale task", "status": "open"}

    def get_events(self, **kwargs) -> list[dict]:
        return []


class _FakeStoreEmpty:
    """Store returning no links and no events."""

    def list_all_issue_links(self, repo: str = "") -> list[dict]:
        return []

    def get_task(self, task_id: int):
        return None

    def get_events(self, **kwargs) -> list[dict]:
        return []


@pytest.fixture
def _make_client(tmp_path, monkeypatch):
    """Factory: returns a function that builds a TestClient with a given store."""

    def _build(store=None):
        monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")

        reg = reg_mod.ServiceRegistry([])
        reg_mod.set_registry(reg)

        class _FakePressure:
            def sample(self):
                from skill_hub.services.monitor import ResourceSample
                return ResourceSample(8000, 16000, 1.0, 8, False, 0.0)
            def sustained_seconds(self): return 0.0
            def last_sample(self): return self.sample()

        reg_mod.set_pressure(_FakePressure())

        from skill_hub.webapp.main import create_app
        app = create_app(store=store)
        return TestClient(app)

    return _build


@pytest.fixture
def client(_make_client):
    return _make_client(_FakeStore())


@pytest.fixture
def client_drift(_make_client):
    return _make_client(_FakeStoreWithDrift())


@pytest.fixture
def client_empty(_make_client):
    return _make_client(_FakeStoreEmpty())


# ---------------------------------------------------------------------------
# GET /control/activity — panel renders
# ---------------------------------------------------------------------------

class TestActivityPanelRenders:
    def test_returns_200(self, client):
        r = client.get("/control/activity")
        assert r.status_code == 200

    def test_contains_issue_links_heading(self, client):
        r = client.get("/control/activity")
        assert "Task" in r.text and "Issue" in r.text

    def test_shows_link_from_store(self, client):
        r = client.get("/control/activity")
        # Issue number 42 and repo are rendered.
        assert "42" in r.text
        assert "owner/repo" in r.text

    def test_shows_task_title(self, client):
        r = client.get("/control/activity")
        assert "Fix the bug" in r.text

    def test_shows_event_kind(self, client):
        r = client.get("/control/activity")
        assert "task.created" in r.text

    def test_shows_events_section(self, client):
        r = client.get("/control/activity")
        assert "Recent Events" in r.text


# ---------------------------------------------------------------------------
# Drift warnings
# ---------------------------------------------------------------------------

class TestDriftWarnings:
    def test_drift_banner_shown_when_drift_present(self, client_drift):
        r = client_drift.get("/control/activity")
        assert r.status_code == 200
        # The drift section should appear when issue is closed but task is open.
        assert "drift" in r.text.lower() or "warning" in r.text.lower()

    def test_no_drift_banner_when_clean(self, client):
        r = client.get("/control/activity")
        assert r.status_code == 200
        # No drift expected (issue open, task open — no mismatch).
        assert "No sync drift detected" in r.text

    def test_drift_direction_shown(self, client_drift):
        r = client_drift.get("/control/activity")
        # issue closed, task open → direction is issue→task
        assert "issue" in r.text.lower()


# ---------------------------------------------------------------------------
# Empty store
# ---------------------------------------------------------------------------

class TestEmptyStore:
    def test_empty_store_returns_200(self, client_empty):
        r = client_empty.get("/control/activity")
        assert r.status_code == 200

    def test_empty_store_no_links_message(self, client_empty):
        r = client_empty.get("/control/activity")
        assert "No task" in r.text or "0 link" in r.text

    def test_empty_store_no_events_message(self, client_empty):
        r = client_empty.get("/control/activity")
        assert "No events" in r.text


# ---------------------------------------------------------------------------
# Degraded data sources — panel must not 500
# ---------------------------------------------------------------------------

class TestDegradedDataSources:
    def test_none_store_returns_200(self, _make_client):
        client = _make_client(None)
        r = client.get("/control/activity")
        assert r.status_code == 200

    def test_none_store_shows_error(self, _make_client):
        client = _make_client(None)
        r = client.get("/control/activity")
        assert "store unavailable" in r.text or "unavailable" in r.text.lower()

    def test_broken_list_all_issue_links_returns_200(self, _make_client, monkeypatch):
        class _BrokenStore(_FakeStoreEmpty):
            def list_all_issue_links(self, repo=""):
                raise RuntimeError("db exploded")

        client = _make_client(_BrokenStore())
        r = client.get("/control/activity")
        assert r.status_code == 200

    def test_broken_get_events_returns_200(self, _make_client):
        class _BrokenEvents(_FakeStoreEmpty):
            def get_events(self, **kwargs):
                raise RuntimeError("events table missing")

        client = _make_client(_BrokenEvents())
        r = client.get("/control/activity")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# HTML escaping (XSS prevention)
# ---------------------------------------------------------------------------

class TestHtmlEscaping:
    def test_malicious_repo_name_is_escaped(self, _make_client):
        class _XssStore(_FakeStoreEmpty):
            def list_all_issue_links(self, repo=""):
                return [{
                    "id": 99,
                    "task_id": 1,
                    "repo": "<script>alert(1)</script>",
                    "issue_number": 1,
                    "url": "",
                    "state": "open",
                    "last_synced_at": None,
                    "writeback_done": 0,
                }]

            def get_task(self, task_id):
                return {"id": 1, "title": "Normal title", "status": "open"}

        client = _make_client(_XssStore())
        r = client.get("/control/activity")
        assert r.status_code == 200
        assert "<script>alert(1)</script>" not in r.text
        assert "&lt;script&gt;" in r.text

    def test_malicious_task_title_is_escaped(self, _make_client):
        class _XssStore2(_FakeStoreEmpty):
            def list_all_issue_links(self, repo=""):
                return [{
                    "id": 98,
                    "task_id": 2,
                    "repo": "safe/repo",
                    "issue_number": 5,
                    "url": "",
                    "state": "",
                    "last_synced_at": None,
                    "writeback_done": 0,
                }]

            def get_task(self, task_id):
                return {"id": 2,
                        "title": '<img src=x onerror=alert(1)>',
                        "status": "open"}

        client = _make_client(_XssStore2())
        r = client.get("/control/activity")
        assert r.status_code == 200
        assert '<img src=x' not in r.text
        assert '&lt;img' in r.text


# ---------------------------------------------------------------------------
# Control page tab integration
# ---------------------------------------------------------------------------

class TestControlPageTab:
    def test_control_page_contains_activity_tab_button(self, client):
        r = client.get("/control")
        assert r.status_code == 200
        assert "Activity Sync" in r.text

    def test_control_page_contains_activity_htmx_target(self, client):
        r = client.get("/control")
        assert "/control/activity" in r.text


# ---------------------------------------------------------------------------
# Sync-now trigger + auto-refresh
# ---------------------------------------------------------------------------

class TestActivitySyncNow:
    def test_panel_renders_sync_now_button(self, client):
        r = client.get("/control/activity")
        assert r.status_code == 200
        assert "Sync now" in r.text
        assert 'hx-post="/control/activity/sync"' in r.text

    def test_control_page_panel_auto_refreshes(self, client):
        r = client.get("/control")
        assert r.status_code == 200
        # The activity tab panel should poll, not just load once.
        assert "every 30s" in r.text

    def test_post_sync_invokes_reconcile_and_returns_panel(self, client, monkeypatch):
        calls = {}

        def _fake_reconcile(store, **kwargs):
            calls["kwargs"] = kwargs
            return {"checked": 1, "closed": 0, "commented": 0}

        from skill_hub import issue_sync as _issue_sync
        monkeypatch.setattr(_issue_sync, "reconcile", _fake_reconcile)

        r = client.post("/control/activity/sync")
        assert r.status_code == 200
        # Re-renders the panel (so HTMX can swap it in).
        assert "Task ↔ Issue Links" in r.text
        # Did not write back to GitHub.
        assert calls["kwargs"].get("writeback") == "off"
        assert calls["kwargs"].get("dry_run") is False

    def test_post_sync_surfaces_error_inline(self, client, monkeypatch):
        def _boom(store, **kwargs):
            raise RuntimeError("github unreachable")

        from skill_hub import issue_sync as _issue_sync
        monkeypatch.setattr(_issue_sync, "reconcile", _boom)

        r = client.post("/control/activity/sync")
        # Endpoint must not 500 — it surfaces the error inline in the panel.
        assert r.status_code == 200
        assert "github unreachable" in r.text
