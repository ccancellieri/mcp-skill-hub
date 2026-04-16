"""Round-trip tests for memexp.snapshot — covers skip / override / llm modes."""
from __future__ import annotations

import json
import sqlite3
import tarfile
from pathlib import Path

import pytest

from memexp import snapshot


def _make_db(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Create a fresh ``skill_hub`` test DB with a single ``skills`` table."""
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE skills (id TEXT PRIMARY KEY, name TEXT, content TEXT)"
    )
    conn.executemany("INSERT INTO skills VALUES (?, ?, ?)", rows)
    # an ephemeral table that must be excluded
    conn.execute("CREATE TABLE session_log (id INTEGER PRIMARY KEY)")
    conn.execute("INSERT INTO session_log (id) VALUES (1)")
    conn.commit()
    conn.close()


def _read_skills(path: Path) -> dict[str, tuple[str, str]]:
    conn = sqlite3.connect(path)
    cur = conn.execute("SELECT id, name, content FROM skills ORDER BY id")
    out = {row[0]: (row[1], row[2]) for row in cur.fetchall()}
    conn.close()
    return out


def _make_project(root: Path, key: str, files: dict[str, str]) -> None:
    proj = root / key / "memory"
    proj.mkdir(parents=True)
    for name, body in files.items():
        (proj / name).write_text(body, encoding="utf-8")


def test_build_snapshot_excludes_ephemeral_and_includes_projects(tmp_path):
    db = tmp_path / "src.db"
    _make_db(db, [("a", "AAA", "aaa-body"), ("b", "BBB", "bbb-body")])

    projects_root = tmp_path / "projects"
    _make_project(projects_root, "proj-a", {"MEMORY.md": "# index", "n.md": "note"})

    out_dir = tmp_path / "out"
    opts = snapshot.ExportOptions(
        project_keys=["proj-a"],
        db_path=db,
        config_path=tmp_path / "nope-config.json",
        settings_path=tmp_path / "nope-settings.json",
        local_skills_dir=tmp_path / "nope-skills",
        projects_root=projects_root,
        output_dir=out_dir,
    )
    tar_path = snapshot.build_snapshot(opts)

    assert tar_path.exists()
    with tarfile.open(tar_path, "r:gz") as t:
        names = sorted(m.name for m in t.getmembers())
    # ephemeral table is gone, project files are present
    assert any(n.endswith("hub/tables/skills.jsonl") for n in names)
    assert not any("session_log" in n for n in names)
    assert any(n.endswith("projects/proj-a/MEMORY.md") for n in names)
    assert any(n.endswith("projects/proj-a/n.md") for n in names)


def _round_trip_setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Build a snapshot from src DB, then return (tar, dest_db, projects_root)."""
    src = tmp_path / "src.db"
    _make_db(src, [
        ("a", "AAA-incoming", "aaa-incoming-body"),
        ("b", "BBB-incoming", "bbb-incoming-body"),
    ])
    projects_root = tmp_path / "projects"
    _make_project(projects_root, "proj-a", {"n.md": "incoming-note\n"})

    out_dir = tmp_path / "out"
    opts = snapshot.ExportOptions(
        project_keys=["proj-a"],
        db_path=src,
        config_path=tmp_path / "x-config.json",
        settings_path=tmp_path / "x-settings.json",
        local_skills_dir=tmp_path / "x-skills",
        projects_root=projects_root,
        output_dir=out_dir,
    )
    tar = snapshot.build_snapshot(opts)
    # Now create a *different* destination DB (with a colliding row 'a') and
    # a different projects root.
    dest = tmp_path / "dest.db"
    _make_db(dest, [("a", "AAA-LOCAL", "aaa-local-body")])  # row 'a' differs
    dest_projects = tmp_path / "dest_projects"
    _make_project(dest_projects, "proj-a", {"n.md": "local-note\n"})
    return tar, dest, dest_projects


def test_restore_skip_preserves_local_rows_and_files(tmp_path):
    tar, dest, dest_projects = _round_trip_setup(tmp_path)
    plan = snapshot.RestorePlan(
        default_mode="skip",
        db_path=dest,
        projects_root=dest_projects,
    )
    report = snapshot.restore_snapshot(tar, plan)
    skills = _read_skills(dest)
    # 'a' kept local, 'b' freshly inserted
    assert skills["a"] == ("AAA-LOCAL", "aaa-local-body")
    assert skills["b"] == ("BBB-incoming", "bbb-incoming-body")
    # project file existed -> skip leaves it alone
    assert (dest_projects / "proj-a" / "memory" / "n.md").read_text() == "local-note\n"
    assert report.tables["skills"].inserted == 1
    assert report.tables["skills"].skipped == 1
    assert report.files.skipped == 1


def test_restore_override_replaces_local(tmp_path):
    tar, dest, dest_projects = _round_trip_setup(tmp_path)
    plan = snapshot.RestorePlan(
        hub_modes={"skills": "override"},
        project_modes={"proj-a": {"mode": "override"}},
        db_path=dest,
        projects_root=dest_projects,
    )
    report = snapshot.restore_snapshot(tar, plan)
    skills = _read_skills(dest)
    assert skills["a"] == ("AAA-incoming", "aaa-incoming-body")
    assert skills["b"] == ("BBB-incoming", "bbb-incoming-body")
    assert (dest_projects / "proj-a" / "memory" / "n.md").read_text() == "incoming-note\n"
    assert report.tables["skills"].replaced == 2
    assert report.files.overwritten == 1


def test_restore_llm_uses_merger_for_conflicts_only(tmp_path):
    tar, dest, dest_projects = _round_trip_setup(tmp_path)

    # Track invocations and return a predictable merged value.
    calls = []
    def fake_row_merger(table, pk, local, incoming, tier):
        calls.append((table, dict(pk), tier))
        return {"id": pk["id"], "name": "MERGED-NAME", "content": "MERGED-BODY"}

    file_calls = []
    def fake_file_merger(rel_path, local_text, incoming_text, tier):
        file_calls.append((rel_path, tier))
        return f"merged({local_text.strip()},{incoming_text.strip()})\n"

    plan = snapshot.RestorePlan(
        hub_modes={"skills": "llm"},
        project_modes={"proj-a": {"mode": "llm", "llm_tier": "tier_cheap"}},
        llm_per_target={"skills": "tier_smart"},
        db_path=dest,
        projects_root=dest_projects,
    )
    report = snapshot.restore_snapshot(
        tar, plan, row_merger=fake_row_merger, file_merger=fake_file_merger,
    )
    skills = _read_skills(dest)
    # row 'a' had a conflict → merger called, value updated
    assert skills["a"] == ("MERGED-NAME", "MERGED-BODY")
    # row 'b' was new → no merger call, plain insert
    assert skills["b"] == ("BBB-incoming", "bbb-incoming-body")
    assert calls == [("skills", {"id": "a"}, "tier_smart")]
    assert report.tables["skills"].llm_merged == 1
    assert report.tables["skills"].inserted == 1
    assert report.llm_calls == 2  # 1 row + 1 file
    # file conflict went through file_merger
    assert file_calls == [("n.md", "tier_cheap")]
    assert report.files.llm_merged == 1


def test_restore_rejects_private_paths(tmp_path):
    """Defence in depth — even if a malicious tar contains private/, it must be skipped."""
    src = tmp_path / "src.db"
    _make_db(src, [("a", "X", "x")])

    out = tmp_path / "out"
    opts = snapshot.ExportOptions(
        project_keys=[],
        db_path=src,
        config_path=tmp_path / "nope-c",
        settings_path=tmp_path / "nope-s",
        local_skills_dir=tmp_path / "nope-l",
        projects_root=tmp_path / "nope-p",
        output_dir=out,
    )
    tar_path = snapshot.build_snapshot(opts)

    # Surgically inject a private/ entry into the snapshot.
    inject_dir = tmp_path / "inject" / "private"
    inject_dir.mkdir(parents=True)
    bad = inject_dir / "secret.md"
    bad.write_text("this should be rejected", encoding="utf-8")
    with tarfile.open(tar_path, "a:") if False else tarfile.open(tar_path, "r:gz") as orig:
        members = orig.getmembers()
        files = {m.name: orig.extractfile(m).read() if orig.extractfile(m) else None for m in members}
    # Rebuild a poisoned tar.
    poisoned = tmp_path / "poisoned.tar.gz"
    with tarfile.open(poisoned, "w:gz") as t:
        for m in members:
            data = files.get(m.name)
            if data is None:
                t.addfile(m)
            else:
                ti = tarfile.TarInfo(name=m.name)
                ti.size = len(data)
                t.addfile(ti, fileobj=__import__("io").BytesIO(data))
        # Now the malicious one
        ti = tarfile.TarInfo(name="projects/proj-x/private/secret.md")
        body = b"private!"
        ti.size = len(body)
        t.addfile(ti, fileobj=__import__("io").BytesIO(body))

    dest = tmp_path / "dest.db"
    _make_db(dest, [])
    plan = snapshot.RestorePlan(db_path=dest, projects_root=tmp_path / "dest-p")
    report = snapshot.restore_snapshot(poisoned, plan)
    # The error list must mention the rejected private entry.
    assert any("private" in e for e in report.errors)
    # And no file under any private/ path must have been written.
    assert not any(p.name == "private" for p in (tmp_path / "dest-p").rglob("*"))
