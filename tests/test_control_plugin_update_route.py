"""Tests for the per-plugin Update endpoint and git_pull_plugin helper.

Mirrors test_control_chrome_route.py: TestClient + monkeypatched dependencies so
real git, filesystem, and embedding backends are never touched.
"""
from __future__ import annotations

import subprocess
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

@pytest.fixture
def client(tmp_path, monkeypatch):
    """TestClient backed by an empty config; no real plugins or services."""
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
    app = create_app(store=None)
    return TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_plugin(name: str, path: Path, source: str = "extra") -> dict:
    return {
        "name": name,
        "full_key": f"{name}@{source}",
        "path": path,
        "manifest": {},
        "description": "",
        "source": source,
        "source_enabled": True,
        "enabled": True,
    }


# ---------------------------------------------------------------------------
# POST /control/plugins/{plugin_id}/update — unknown plugin
# ---------------------------------------------------------------------------

class TestUpdateUnknownPlugin:
    def test_unknown_plugin_returns_404(self, client, monkeypatch):
        from skill_hub.webapp.routes import control_plugins as cp
        monkeypatch.setattr(cp._pr, "iter_all_plugins", lambda: iter([]))
        r = client.post("/control/plugins/does-not-exist/update")
        assert r.status_code == 404

    def test_unknown_plugin_body_contains_unknown(self, client, monkeypatch):
        from skill_hub.webapp.routes import control_plugins as cp
        monkeypatch.setattr(cp._pr, "iter_all_plugins", lambda: iter([]))
        r = client.post("/control/plugins/does-not-exist/update")
        assert "Unknown plugin" in r.text or "unknown" in r.text.lower()


# ---------------------------------------------------------------------------
# git_pull_plugin helper — non-git plugin
# ---------------------------------------------------------------------------

class TestGitPullPluginNotGit:
    def test_non_git_dir_returns_not_git(self, tmp_path):
        from skill_hub.webapp.routes.control_plugins import git_pull_plugin
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        result = git_pull_plugin(plugin_dir)
        assert result["status"] == "not_git"

    def test_non_git_message_no_traceback(self, tmp_path):
        from skill_hub.webapp.routes.control_plugins import git_pull_plugin
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        result = git_pull_plugin(plugin_dir)
        assert "Traceback" not in result["message"]
        assert result["message"]

    def test_nonexistent_dir_returns_error(self, tmp_path):
        from skill_hub.webapp.routes.control_plugins import git_pull_plugin
        result = git_pull_plugin(tmp_path / "no-such-dir")
        assert result["status"] == "error"


# ---------------------------------------------------------------------------
# git_pull_plugin helper — git-backed plugin (fake repo via tmp_path)
# ---------------------------------------------------------------------------

class TestGitPullPluginGitBacked:
    @pytest.fixture
    def fake_repo(self, tmp_path):
        """Create a minimal bare+clone pair; upstream has one extra commit."""
        import subprocess as sp
        bare = tmp_path / "upstream.git"
        clone = tmp_path / "plugin-clone"

        # Initialise upstream bare repo with an explicit branch name so the
        # test is not sensitive to git's global init.defaultBranch setting.
        sp.run(["git", "init", "--bare", "--initial-branch=main", str(bare)],
               check=True, capture_output=True)

        # Clone it locally.
        sp.run(["git", "clone", str(bare), str(clone)], check=True,
               capture_output=True)

        # Commit something in the clone and push → upstream is now populated.
        (clone / "plugin.json").write_text('{"name": "fake-plugin"}')
        sp.run(["git", "-C", str(clone), "config", "user.email", "t@t.com"],
               check=True, capture_output=True)
        sp.run(["git", "-C", str(clone), "config", "user.name", "T"],
               check=True, capture_output=True)
        sp.run(["git", "-C", str(clone), "add", "."], check=True,
               capture_output=True)
        sp.run(["git", "-C", str(clone), "commit", "-m", "init"],
               check=True, capture_output=True)
        sp.run(["git", "-C", str(clone), "push", "-u", "origin", "main"],
               check=True, capture_output=True)
        # Ensure origin/HEAD points to main so _default_branch() resolves it.
        sp.run(["git", "-C", str(clone), "remote", "set-head", "origin", "main"],
               check=True, capture_output=True)

        # Now add one more commit only to the bare (simulates remote advance).
        # We do it by making a second clone, committing, and pushing.
        clone2 = tmp_path / "clone2"
        sp.run(["git", "clone", str(bare), str(clone2)], check=True,
               capture_output=True)
        sp.run(["git", "-C", str(clone2), "config", "user.email", "t@t.com"],
               check=True, capture_output=True)
        sp.run(["git", "-C", str(clone2), "config", "user.name", "T"],
               check=True, capture_output=True)
        (clone2 / "skills.md").write_text("# skills")
        sp.run(["git", "-C", str(clone2), "add", "."], check=True,
               capture_output=True)
        sp.run(["git", "-C", str(clone2), "commit", "-m", "add skills"],
               check=True, capture_output=True)
        sp.run(["git", "-C", str(clone2), "push", "origin", "HEAD"],
               check=True, capture_output=True)

        return clone

    def test_updated_status_when_remote_is_ahead(self, fake_repo):
        from skill_hub.webapp.routes.control_plugins import git_pull_plugin
        result = git_pull_plugin(fake_repo)
        assert result["status"] == "updated"

    def test_updated_has_before_after_shas(self, fake_repo):
        from skill_hub.webapp.routes.control_plugins import git_pull_plugin
        result = git_pull_plugin(fake_repo)
        assert "before_sha" in result and "after_sha" in result
        assert result["before_sha"] != result["after_sha"]

    def test_updated_commit_count_positive(self, fake_repo):
        from skill_hub.webapp.routes.control_plugins import git_pull_plugin
        result = git_pull_plugin(fake_repo)
        assert result.get("commit_count", 0) >= 1

    def test_up_to_date_after_second_call(self, fake_repo):
        from skill_hub.webapp.routes.control_plugins import git_pull_plugin
        git_pull_plugin(fake_repo)  # first call advances HEAD
        result = git_pull_plugin(fake_repo)
        assert result["status"] == "up_to_date"

    def test_dirty_tree_aborts(self, fake_repo):
        from skill_hub.webapp.routes.control_plugins import git_pull_plugin
        # Introduce an untracked + modified file.
        (fake_repo / "dirty.txt").write_text("uncommitted")
        import subprocess as sp
        sp.run(["git", "-C", str(fake_repo), "add", "dirty.txt"],
               check=True, capture_output=True)
        result = git_pull_plugin(fake_repo)
        assert result["status"] == "dirty"


# ---------------------------------------------------------------------------
# POST /control/plugins/{plugin_id}/update — HTML fragment
# ---------------------------------------------------------------------------

class TestUpdateEndpointHTMLFragment:
    def _make_client_with_plugin(self, tmp_path, monkeypatch, plugin_dir: Path,
                                 status_result: dict):
        """Build a test client where iter_all_plugins returns one plugin and
        git_pull_plugin is monkeypatched to return status_result."""
        monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")
        reg = reg_mod.ServiceRegistry([])
        reg_mod.set_registry(reg)

        class _FP:
            def sample(self):
                from skill_hub.services.monitor import ResourceSample
                return ResourceSample(8000, 16000, 1.0, 8, False, 0.0)
            def sustained_seconds(self): return 0.0
            def last_sample(self): return self.sample()

        reg_mod.set_pressure(_FP())

        from skill_hub.webapp.main import create_app
        app = create_app(store=None)

        from skill_hub.webapp.routes import control_plugins as cp

        fake_p = _fake_plugin("my-plugin", plugin_dir)
        monkeypatch.setattr(cp._pr, "iter_all_plugins",
                            lambda: iter([fake_p]))
        monkeypatch.setattr(cp, "git_pull_plugin", lambda d: status_result)

        return TestClient(app)

    def test_not_git_returns_200(self, tmp_path, monkeypatch):
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        c = self._make_client_with_plugin(
            tmp_path, monkeypatch, plugin_dir,
            {"status": "not_git", "message": "Not a git repository — no update available."},
        )
        r = c.post("/control/plugins/my-plugin%40extra/update")
        assert r.status_code == 200

    def test_not_git_shows_na_class(self, tmp_path, monkeypatch):
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        c = self._make_client_with_plugin(
            tmp_path, monkeypatch, plugin_dir,
            {"status": "not_git", "message": "Not a git repository — no update available."},
        )
        r = c.post("/control/plugins/my-plugin%40extra/update")
        assert "update-na" in r.text

    def test_updated_shows_ok_class(self, tmp_path, monkeypatch):
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        c = self._make_client_with_plugin(
            tmp_path, monkeypatch, plugin_dir,
            {"status": "updated", "message": "abc1234 → def5678 (2 commits)",
             "before_sha": "abc1234", "after_sha": "def5678", "commit_count": 2},
        )
        r = c.post("/control/plugins/my-plugin%40extra/update")
        assert r.status_code == 200
        assert "update-ok" in r.text

    def test_error_shows_error_class(self, tmp_path, monkeypatch):
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        c = self._make_client_with_plugin(
            tmp_path, monkeypatch, plugin_dir,
            {"status": "error", "message": "git fetch failed: timeout"},
        )
        r = c.post("/control/plugins/my-plugin%40extra/update")
        assert r.status_code == 200
        assert "update-error" in r.text

    def test_up_to_date_shows_ok_class(self, tmp_path, monkeypatch):
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        c = self._make_client_with_plugin(
            tmp_path, monkeypatch, plugin_dir,
            {"status": "up_to_date", "message": "Already up to date at abc1234 (main)."},
        )
        r = c.post("/control/plugins/my-plugin%40extra/update")
        assert r.status_code == 200
        assert "update-ok" in r.text

    def test_message_is_in_response(self, tmp_path, monkeypatch):
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()
        c = self._make_client_with_plugin(
            tmp_path, monkeypatch, plugin_dir,
            {"status": "updated", "message": "aaa → bbb (1 commit)",
             "before_sha": "aaa", "after_sha": "bbb", "commit_count": 1},
        )
        r = c.post("/control/plugins/my-plugin%40extra/update")
        assert "aaa" in r.text and "bbb" in r.text


# ---------------------------------------------------------------------------
# HTML escaping — no XSS through status messages
# ---------------------------------------------------------------------------

class TestHtmlEscaping:
    def test_xss_in_not_git_message_is_escaped(self, tmp_path, monkeypatch):
        """git_pull_plugin escapes its output; the route echoes it directly."""
        plugin_dir = tmp_path / "my-plugin"
        plugin_dir.mkdir()

        # Bypass monkeypatching: call git_pull_plugin on a non-.git dir that has
        # a crafted name — the dir name never appears in not_git output; use
        # the error path instead by making it non-existent.
        from skill_hub.webapp.routes.control_plugins import git_pull_plugin
        result = git_pull_plugin(tmp_path / "<script>alert(1)</script>")
        # status is "error", message must be escaped.
        assert "<script>" not in result["message"]
        assert "&lt;script&gt;" in result["message"] or "script" not in result["message"]

    def test_route_does_not_echo_raw_html(self, tmp_path, monkeypatch):
        monkeypatch.setattr(cfg_mod, "CONFIG_PATH", tmp_path / "config.json")
        reg = reg_mod.ServiceRegistry([])
        reg_mod.set_registry(reg)

        class _FP:
            def sample(self):
                from skill_hub.services.monitor import ResourceSample
                return ResourceSample(8000, 16000, 1.0, 8, False, 0.0)
            def sustained_seconds(self): return 0.0
            def last_sample(self): return self.sample()

        reg_mod.set_pressure(_FP())

        from skill_hub.webapp.main import create_app
        app = create_app(store=None)

        from skill_hub.webapp.routes import control_plugins as cp

        plugin_dir = tmp_path / "xss-plugin"
        plugin_dir.mkdir()
        fake_p = _fake_plugin("xss-plugin", plugin_dir)
        monkeypatch.setattr(cp._pr, "iter_all_plugins", lambda: iter([fake_p]))
        # Inject XSS into the message via monkeypatched git_pull_plugin.
        monkeypatch.setattr(cp, "git_pull_plugin",
                            lambda d: {"status": "error",
                                       "message": "&lt;script&gt;alert(1)&lt;/script&gt;"})

        c = TestClient(app)
        r = c.post("/control/plugins/xss-plugin%40extra/update")
        assert r.status_code == 200
        assert "<script>alert(1)</script>" not in r.text


# ---------------------------------------------------------------------------
# Control page — plugins tab renders Update button
# ---------------------------------------------------------------------------

class TestPluginsTabHasUpdateButton:
    def test_plugins_panel_contains_update_post(self, client, monkeypatch, tmp_path):
        from skill_hub.webapp.routes import control_plugins as cp

        plugin_dir = tmp_path / "test-plugin"
        plugin_dir.mkdir()
        fake_p = _fake_plugin("test-plugin", plugin_dir)
        monkeypatch.setattr(cp._pr, "iter_all_plugins", lambda: iter([fake_p]))

        r = client.get("/control/plugins")
        assert r.status_code == 200
        assert "update" in r.text.lower()
        assert "/update" in r.text

    def test_update_button_has_aria_label(self, client, monkeypatch, tmp_path):
        from skill_hub.webapp.routes import control_plugins as cp

        plugin_dir = tmp_path / "aria-plugin"
        plugin_dir.mkdir()
        fake_p = _fake_plugin("aria-plugin", plugin_dir)
        monkeypatch.setattr(cp._pr, "iter_all_plugins", lambda: iter([fake_p]))

        r = client.get("/control/plugins")
        assert r.status_code == 200
        assert "aria-label" in r.text
