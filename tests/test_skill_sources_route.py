from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from skill_hub import config as cfg_mod  # noqa: E402
from skill_hub.services import registry as reg_mod  # noqa: E402
from skill_hub.services.base import Service, Status  # noqa: E402


class _FakeService(Service):
    name = "noop"
    label = "Noop"
    description = ""

    def status(self) -> Status:
        return "stopped"

    def is_available(self):
        return True, ""

    def start(self):
        return True, ""

    def stop(self):
        return True, ""


def _client(tmp_path: Path, monkeypatch, config: dict | None = None) -> TestClient:
    cfg_path = tmp_path / "config.json"
    monkeypatch.setattr(cfg_mod, "CONFIG_PATH", cfg_path)
    cfg_path.write_text(json.dumps(config or {}))
    reg_mod.set_registry(reg_mod.ServiceRegistry([_FakeService()]))

    from skill_hub.webapp.main import create_app

    return TestClient(create_app(store=None))


def _write_skill(root: Path, name: str = "sample") -> None:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"""---
name: {name}
description: Use for {name} tests.
---

# {name}
""",
        encoding="utf-8",
    )


def _write_json_skill(root: Path, name: str = "local") -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.json").write_text(
        json.dumps({
            "name": name,
            "description": f"Use for {name} tests.",
            "triggers": [name],
            "steps": ["return ok"],
        }),
        encoding="utf-8",
    )


def test_skill_sources_page_renders_configured_sources(tmp_path, monkeypatch):
    source_dir = tmp_path / "skills"
    source_dir.mkdir()
    live_dir = tmp_path / "live"
    live_dir.mkdir()
    client = _client(
        tmp_path,
        monkeypatch,
        {
            "skill_import_sources": [
                {"path": str(source_dir), "source": "custom", "enabled": True}
            ],
            "extra_skill_dirs": [
                {"path": str(live_dir), "source": "live", "enabled": True}
            ],
        },
    )

    response = client.get("/skill-sources")

    assert response.status_code == 200
    assert "Skill Sources" in response.text
    assert "custom" in response.text
    assert str(source_dir) in response.text
    assert 'for="source-path-0"' in response.text
    assert 'for="source-label-0"' in response.text
    assert "Live indexed roots" in response.text
    assert str(live_dir) in response.text


def test_skill_sources_save_persists_multiple_sources(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/skill-sources/save",
        data={
            "path": [str(first), "", str(second)],
            "source": ["first", "blank", "second"],
            "enabled": ["0", "2"],
        },
    )

    assert response.status_code == 200
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["skill_import_sources"] == [
        {"path": str(first), "source": "first", "enabled": True},
        {"path": str(second), "source": "second", "enabled": True},
    ]


def test_skill_sources_audit_uses_configured_sources(tmp_path, monkeypatch):
    source_dir = tmp_path / "skills"
    skill_dir = source_dir / "sample"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: sample-skill
description: Use for sample audits.
---

# Sample
""",
        encoding="utf-8",
    )
    client = _client(
        tmp_path,
        monkeypatch,
        {
            "skill_import_sources": [
                {"path": str(source_dir), "source": "sample", "enabled": True}
            ]
        },
    )

    response = client.post("/skill-sources/audit")

    assert response.status_code == 200
    assert "sample-skill" in response.text
    assert "skill_md" in response.text


def test_skill_sources_audit_endpoint_saves_form_before_refresh(tmp_path, monkeypatch):
    source_dir = tmp_path / "fresh"
    skill_dir = source_dir / "sample"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: fresh-skill
description: Use for fresh audits.
---

# Fresh
""",
        encoding="utf-8",
    )
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/skill-sources/audit",
        data={
            "path": [str(source_dir)],
            "source": ["fresh"],
            "enabled": ["0"],
        },
    )

    assert response.status_code == 200
    assert "fresh-skill" in response.text
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["skill_import_sources"] == [
        {"path": str(source_dir), "source": "fresh", "enabled": True}
    ]


def test_skill_sources_import_promotes_enabled_sources_and_reindexes(
    tmp_path, monkeypatch
):
    source_dir = tmp_path / "skills"
    source_dir.mkdir()
    _write_skill(source_dir)
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    calls = []

    def fake_index_all(store):
        calls.append(store)
        return 3, ["info: skipped unchanged skills"]

    import skill_hub.webapp.routes.skill_sources as route_mod

    monkeypatch.setattr(route_mod, "index_all", fake_index_all)
    client = _client(
        tmp_path,
        monkeypatch,
        {
            "skill_import_sources": [
                {"path": str(source_dir), "source": "custom", "enabled": True},
                {"path": str(tmp_path / "disabled"), "source": "off", "enabled": False},
            ],
            "extra_skill_dirs": [
                {"path": str(existing_dir), "source": "existing", "enabled": True}
            ],
        },
    )

    response = client.post("/skill-sources/import")

    assert response.status_code == 200
    assert "Imported 1 source(s); indexed 3 item(s)." in response.text
    assert calls == [None]
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["extra_skill_dirs"] == [
        {"path": str(existing_dir), "source": "existing", "enabled": True},
        {"path": str(source_dir), "source": "custom", "enabled": True},
    ]


def test_skill_sources_import_endpoint_saves_form_before_reindex(tmp_path, monkeypatch):
    source_dir = tmp_path / "fresh"
    source_dir.mkdir()
    _write_skill(source_dir)
    calls = []

    import skill_hub.webapp.routes.skill_sources as route_mod

    monkeypatch.setattr(route_mod, "index_all", lambda store: calls.append(store) or (2, []))
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/skill-sources/import",
        data={
            "path": [str(source_dir)],
            "source": ["fresh"],
            "enabled": ["0"],
        },
    )

    assert response.status_code == 200
    assert "Imported 1 source(s); indexed 2 item(s)." in response.text
    assert calls == [None]
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["skill_import_sources"] == [
        {"path": str(source_dir), "source": "fresh", "enabled": True}
    ]
    assert {"path": str(source_dir), "source": "fresh", "enabled": True} in data[
        "extra_skill_dirs"
    ]


def test_skill_sources_import_reuses_existing_live_source(tmp_path, monkeypatch):
    source_dir = tmp_path / "skills"
    source_dir.mkdir()
    _write_skill(source_dir)

    import skill_hub.webapp.routes.skill_sources as route_mod

    monkeypatch.setattr(route_mod, "index_all", lambda store: (0, []))
    client = _client(
        tmp_path,
        monkeypatch,
        {
            "skill_import_sources": [
                {"path": str(source_dir), "source": "new-label", "enabled": True}
            ],
            "extra_skill_dirs": [
                {"path": str(source_dir), "source": "old-label", "enabled": False}
            ],
        },
    )

    response = client.post("/skill-sources/import")

    assert response.status_code == 200
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["extra_skill_dirs"] == [
        {"path": str(source_dir), "source": "new-label", "enabled": True}
    ]


def test_skill_sources_apply_audit_saves_rows_before_refresh(tmp_path, monkeypatch):
    source_dir = tmp_path / "fresh"
    skill_dir = source_dir / "sample"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: fresh-skill
description: Use for fresh audits.
---

# Fresh
""",
        encoding="utf-8",
    )
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/skill-sources/apply",
        data={
            "action": "audit",
            "path": [str(source_dir)],
            "source": ["fresh"],
            "enabled": ["0"],
        },
    )

    assert response.status_code == 200
    assert "fresh-skill" in response.text
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["skill_import_sources"] == [
        {"path": str(source_dir), "source": "fresh", "enabled": True}
    ]


def test_skill_sources_apply_import_saves_rows_before_reindex(tmp_path, monkeypatch):
    source_dir = tmp_path / "fresh"
    source_dir.mkdir()
    _write_skill(source_dir)
    calls = []

    import skill_hub.webapp.routes.skill_sources as route_mod

    monkeypatch.setattr(route_mod, "index_all", lambda store: calls.append(store) or (2, []))
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/skill-sources/apply",
        data={
            "action": "import",
            "path": [str(source_dir)],
            "source": ["fresh"],
            "enabled": ["0"],
        },
    )

    assert response.status_code == 200
    assert "Imported 1 source(s); indexed 2 item(s)." in response.text
    assert calls == [None]
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["skill_import_sources"] == [
        {"path": str(source_dir), "source": "fresh", "enabled": True}
    ]
    assert {"path": str(source_dir), "source": "fresh", "enabled": True} in data[
        "extra_skill_dirs"
    ]


def test_skill_sources_import_disables_previously_managed_source(tmp_path, monkeypatch):
    source_dir = tmp_path / "skills"
    source_dir.mkdir()
    _write_skill(source_dir)

    import skill_hub.webapp.routes.skill_sources as route_mod

    monkeypatch.setattr(route_mod, "index_all", lambda store: (0, []))
    client = _client(
        tmp_path,
        monkeypatch,
        {
            "skill_import_sources": [
                {"path": str(source_dir), "source": "custom", "enabled": False}
            ],
            "skill_import_managed_paths": [str(source_dir)],
            "extra_skill_dirs": [
                {"path": str(source_dir), "source": "custom", "enabled": True}
            ],
        },
    )

    response = client.post("/skill-sources/import")

    assert response.status_code == 200
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["extra_skill_dirs"] == []


def test_skill_sources_import_removes_deleted_managed_source(tmp_path, monkeypatch):
    source_dir = tmp_path / "skills"
    source_dir.mkdir()
    _write_skill(source_dir)
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()

    import skill_hub.webapp.routes.skill_sources as route_mod

    monkeypatch.setattr(route_mod, "index_all", lambda store: (0, []))
    client = _client(
        tmp_path,
        monkeypatch,
        {
            "skill_import_sources": [],
            "skill_import_managed_paths": [str(source_dir)],
            "extra_skill_dirs": [
                {"path": str(source_dir), "source": "custom", "enabled": True},
                {"path": str(manual_dir), "source": "manual", "enabled": True},
            ],
        },
    )

    response = client.post(
        "/skill-sources/apply",
        data={"action": "import", "path": [""], "source": [""], "enabled": []},
    )

    assert response.status_code == 200
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["extra_skill_dirs"] == [
        {"path": str(manual_dir), "source": "manual", "enabled": True}
    ]


def test_skill_sources_save_rejects_duplicate_labels(tmp_path, monkeypatch):
    first = tmp_path / "first"
    second = tmp_path / "second"
    client = _client(tmp_path, monkeypatch)

    response = client.post(
        "/skill-sources/save",
        data={
            "path": [str(first), str(second)],
            "source": ["dup", "dup"],
            "enabled": ["0", "1"],
        },
    )

    assert response.status_code == 200
    assert "Duplicate source label: dup" in response.text
    data = json.loads((tmp_path / "config.json").read_text())
    assert "skill_import_sources" not in data


def test_skill_sources_import_blocks_missing_source(tmp_path, monkeypatch):
    missing = tmp_path / "missing"
    calls = []

    import skill_hub.webapp.routes.skill_sources as route_mod

    monkeypatch.setattr(route_mod, "index_all", lambda store: calls.append(store) or (0, []))
    client = _client(
        tmp_path,
        monkeypatch,
        {
            "skill_import_sources": [
                {"path": str(missing), "source": "missing", "enabled": True}
            ]
        },
    )

    response = client.post("/skill-sources/import")

    assert response.status_code == 200
    assert "source not found" in response.text
    assert calls == []


def test_skill_sources_import_blocks_empty_source_even_with_existing_live_root(
    tmp_path, monkeypatch
):
    empty = tmp_path / "empty"
    empty.mkdir()
    live = tmp_path / "live"
    live.mkdir()
    _write_skill(live)
    calls = []

    import skill_hub.webapp.routes.skill_sources as route_mod

    monkeypatch.setattr(route_mod, "index_all", lambda store: calls.append(store) or (0, []))
    client = _client(
        tmp_path,
        monkeypatch,
        {
            "skill_import_sources": [
                {"path": str(empty), "source": "empty", "enabled": True}
            ],
            "extra_skill_dirs": [
                {"path": str(live), "source": "live", "enabled": True}
            ],
        },
    )

    response = client.post("/skill-sources/import")

    assert response.status_code == 200
    assert f"No importable SKILL.md or local JSON skill candidates found in source: {empty}" in response.text
    assert calls == []


def test_skill_sources_reindex_action_reports_failures(tmp_path, monkeypatch):
    source_dir = tmp_path / "skills"
    source_dir.mkdir()
    _write_skill(source_dir)

    import skill_hub.webapp.routes.skill_sources as route_mod

    def boom(store):
        raise RuntimeError("database locked")

    monkeypatch.setattr(route_mod, "index_all", boom)
    client = _client(
        tmp_path,
        monkeypatch,
        {
            "skill_import_sources": [
                {"path": str(source_dir), "source": "custom", "enabled": True}
            ],
            "extra_skill_dirs": [
                {"path": str(source_dir), "source": "custom", "enabled": True}
            ],
        },
    )

    response = client.post(
        "/skill-sources/apply",
        data={
            "action": "reindex",
            "path": [str(source_dir)],
            "source": ["custom"],
            "enabled": ["0"],
        },
    )

    assert response.status_code == 200
    assert "reindex failed: database locked" in response.text


def test_skill_sources_import_sets_and_removes_managed_local_json_dir(
    tmp_path, monkeypatch
):
    json_dir = tmp_path / "json-skills"
    _write_json_skill(json_dir)

    import skill_hub.webapp.routes.skill_sources as route_mod

    monkeypatch.setattr(route_mod, "index_all", lambda store: (0, []))
    client = _client(
        tmp_path,
        monkeypatch,
        {
            "skill_import_sources": [
                {"path": str(json_dir), "source": "json", "enabled": True}
            ],
        },
    )

    response = client.post("/skill-sources/import")

    assert response.status_code == 200
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["local_skills_dir"] == str(json_dir)
    assert data["skill_import_managed_local_json_dir"] == str(json_dir)

    response = client.post(
        "/skill-sources/apply",
        data={
            "action": "import",
            "path": [str(json_dir)],
            "source": ["json"],
            "enabled": [],
        },
    )

    assert response.status_code == 200
    data = json.loads((tmp_path / "config.json").read_text())
    assert data["local_skills_dir"] == ""
    assert data["skill_import_managed_local_json_dir"] is None


def test_skill_sources_import_rejects_multiple_enabled_json_sources(
    tmp_path, monkeypatch
):
    first = tmp_path / "json-one"
    second = tmp_path / "json-two"
    _write_json_skill(first, "one")
    _write_json_skill(second, "two")
    calls = []

    import skill_hub.webapp.routes.skill_sources as route_mod

    monkeypatch.setattr(route_mod, "index_all", lambda store: calls.append(store) or (0, []))
    client = _client(
        tmp_path,
        monkeypatch,
        {
            "skill_import_sources": [
                {"path": str(first), "source": "one", "enabled": True},
                {"path": str(second), "source": "two", "enabled": True},
            ],
        },
    )

    response = client.post("/skill-sources/import")

    assert response.status_code == 200
    assert "Only one enabled local JSON source is supported" in response.text
    assert calls == []
