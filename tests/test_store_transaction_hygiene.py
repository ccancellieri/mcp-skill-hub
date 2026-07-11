"""Regression tests for #142 — long-lived implicit transactions on SkillStore.

``SkillStore._conn`` uses sqlite3's default ``isolation_level`` (``''``), so
any INSERT/UPDATE/DELETE/REPLACE statement opens an implicit transaction —
even one that matches zero rows — and it stays open until an explicit
``commit()``/``rollback()``. A write method that calls ``commit()`` only on
some branches (e.g. ``if rowcount: commit()``) silently leaves the shared
daemon connection sitting inside an open transaction whenever the guard is
false, which pins the WAL read mark and freezes the connection's snapshot
(see the #139 follow-up incident that #142 is about).

Two concrete offenders were found by this audit and are fixed alongside
this test:

* ``prune_tool_examples`` only called ``commit()`` when rows were actually
  deleted (``if pruned: commit()``), even though both DELETE statements run
  unconditionally every call.
* ``dequeue_job`` only called ``commit()`` when a job was claimed
  (``if row is not None: commit()``), even though the UPDATE runs
  unconditionally (and opens a transaction) even against an empty queue.

This file has two dedicated regression tests reproducing those exact
call shapes, plus a broader data-driven sweep that walks a large set of
mutating public ``SkillStore`` methods (including the "nothing matched" /
early-return / branchy shapes most likely to hide this bug) and asserts
``store._conn.in_transaction is False`` after every single call. It is not
a claim of 100% method coverage, but it covers the write surface broadly
enough to catch a regression of this class anywhere it recurs.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import pytest


@pytest.fixture()
def store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore

    monkeypatch.setenv("HOME", str(tmp_path))
    return SkillStore(db_path=tmp_path / "skill_hub.db")


def _check(store, label: str) -> None:
    assert store._conn.in_transaction is False, (
        f"{label} left store._conn.in_transaction = True — an implicit "
        f"transaction is still open after the call returned (see #142)"
    )


# ---------------------------------------------------------------------------
# Isolation level sanity check — the premise of #142
# ---------------------------------------------------------------------------

def test_isolation_level_is_default_implicit_mode(store):
    """SkillStore relies on sqlite3's legacy implicit-transaction mode.

    A DELETE/UPDATE that matches zero rows still opens a transaction under
    this mode — that's the whole reason a conditional ``commit()`` is
    dangerous. If this ever changes (e.g. a future move to
    ``isolation_level=None`` + explicit ``BEGIN``), this test documents the
    behavior this whole file is guarding against.
    """
    assert store._conn.isolation_level == ""


# ---------------------------------------------------------------------------
# Dedicated regressions for the two offenders fixed by this issue
# ---------------------------------------------------------------------------

def test_prune_tool_examples_noop_still_commits(store):
    """#142: pruning an empty/fresh table must not leave a dangling txn.

    Both DELETE statements in prune_tool_examples run unconditionally, so
    even when nothing is old enough (or over the row cap) to delete, the
    connection was left mid-transaction because commit() was gated on
    ``if pruned:``.
    """
    pruned = store.prune_tool_examples(max_age_days=30, max_rows=5000)
    assert pruned == 0
    _check(store, "prune_tool_examples (no-op)")


def test_dequeue_job_empty_queue_still_commits(store):
    """#142: polling an empty job queue must not leave a dangling txn.

    The UPDATE...WHERE id = (subquery) always runs, even when the subquery
    matches nothing, but commit() was gated on ``if row is not None:``. A
    worker polling loop would leave the connection mid-transaction on every
    empty-queue tick.
    """
    row = store.dequeue_job()
    assert row is None
    _check(store, "dequeue_job (empty queue)")


# ---------------------------------------------------------------------------
# Broad data-driven sweep across the mutating public API
# ---------------------------------------------------------------------------

def test_write_methods_never_leave_open_transaction(store, tmp_path, monkeypatch):
    """Walk a large slice of SkillStore's write surface end to end.

    After every single mutating call, assert the connection isn't sitting
    in an open transaction. Deliberately includes empty-result / no-op /
    early-return branches (the exact shape that hid #142's two bugs)
    alongside the "normal" happy-path branches.
    """
    from skill_hub.store import Skill
    import skill_hub.embeddings as _emb_mod

    # upsert_vector() calls embeddings.embed() — stub it so the sweep is
    # fast/hermetic and doesn't depend on Ollama/sentence-transformers.
    monkeypatch.setattr(
        _emb_mod, "embed",
        lambda text, model=None, timeout=15.0: [0.1] * 8,
    )

    vec = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    # --- skills / embeddings / feedback -----------------------------------
    skill_path = tmp_path / "main_skill.md"
    skill_path.write_text("main skill content", encoding="utf-8")
    skill_id = "test:hygiene-skill"
    store.upsert_skill(
        Skill(id=skill_id, name="Hygiene Skill", description="desc",
              content="content", file_path=str(skill_path),
              plugin="test-plugin", target="claude"),
        content_hash="hash-main",
    )
    _check(store, "upsert_skill")

    store.upsert_embedding(skill_id, "test-model", vec)
    _check(store, "upsert_embedding")

    store.record_feedback(skill_id, "test query", vec, helpful=True)
    _check(store, "record_feedback (helpful)")
    store.record_feedback(skill_id, "test query", vec, helpful=False)
    _check(store, "record_feedback (unhelpful)")

    injection_id = store.log_skill_injection(skill_id, query="q", session_id="sess1")
    _check(store, "log_skill_injection")

    store.record_skill_used(skill_id, "sess1", injection_id=injection_id)
    _check(store, "record_skill_used")

    # --- teachings -----------------------------------------------------
    teaching_id = store.add_teaching(
        "when I give a URL", vec, "suggest chrome-devtools", "plugin",
        "chrome-devtools-mcp",
    )
    _check(store, "add_teaching")

    # --- plugins ---------------------------------------------------------
    store.upsert_plugin("test-plugin", "test-plugin", "desc")
    _check(store, "upsert_plugin")
    store.upsert_plugin_embedding("test-plugin", "test-model", vec)
    _check(store, "upsert_plugin_embedding")

    # --- session log -------------------------------------------------------
    store.log_session_tool("sess1", "query", vec, "search_skills", "test-plugin")
    _check(store, "log_session_tool")
    store.log_session_subagent("sess1", "agent1", "general-purpose", "SubagentStart")
    _check(store, "log_session_subagent")

    # --- tasks ---------------------------------------------------------
    task_id = store.save_task(
        title="hygiene task", summary="summary", vector=vec,
        session_id="sess1", cwd=str(tmp_path), branch="main", repo="test-repo",
    )
    _check(store, "save_task")

    store.set_task_worktree(task_id, '{"path": "/tmp/wt"}')
    _check(store, "set_task_worktree")
    store.update_task(task_id, summary="updated summary", vector=vec)
    _check(store, "update_task")
    store.bind_task_to_session(task_id, "sess1")
    _check(store, "bind_task_to_session (no session_context row yet)")
    store.touch_task_activity(task_id)
    _check(store, "touch_task_activity")
    store.set_task_auto_approve(task_id, True)
    _check(store, "set_task_auto_approve")
    store.set_task_options(task_id, {"routing_disabled": True})
    _check(store, "set_task_options")

    link_id = None
    store.link_task_issue(task_id, 142, repo="test-repo", url="https://example/142")
    _check(store, "link_task_issue")
    links = store.get_issue_links(task_id)
    assert links
    link_id = links[0]["id"]
    store.update_link_state(link_id, "open", writeback_done=0)
    _check(store, "update_link_state")

    store.reopen_task(task_id)
    _check(store, "reopen_task")

    # project_claude_task: exercise created / updated / closed / noop.
    r = store.project_claude_task(key="claude-key-1", title="ct", status="in_progress",
                                   session_id="sess1", cwd=str(tmp_path), branch="main")
    _check(store, "project_claude_task (created)")
    assert r["action"] == "created"
    r = store.project_claude_task(key="claude-key-1", title="ct2", status="in_progress",
                                   session_id="sess1")
    _check(store, "project_claude_task (updated)")
    assert r["action"] == "updated"
    r = store.project_claude_task(key="claude-key-1", title="ct2", status="completed",
                                   session_id="sess1")
    _check(store, "project_claude_task (closed)")
    assert r["action"] == "closed"
    r = store.project_claude_task(key="claude-key-1", title="x", status="completed",
                                   session_id="sess1")
    _check(store, "project_claude_task (noop)")
    assert r["action"] == "noop"

    # project_memory_task: exercise created / updated / closed / noop.
    # Uses a vector orthogonal to `vec` (dot product 0) so its Tier-2
    # cosine-similarity fallback can't accidentally adopt the still-open
    # `vec`-tagged tasks created earlier in this sweep (e.g. the hygiene
    # task's save_task call) at similarity 1.0.
    mem_vec = [0.8, -0.7, 0.6, -0.5, 0.4, -0.3, 0.2, -0.1]
    m = store.project_memory_task(key="mem-key-1", title="mem", vector=mem_vec)
    _check(store, "project_memory_task (created)")
    assert m["action"] == "created"
    m = store.project_memory_task(key="mem-key-1", title="mem2", vector=mem_vec)
    _check(store, "project_memory_task (updated)")
    assert m["action"] == "updated"
    m = store.project_memory_task(key="mem-key-1", title="mem2", vector=mem_vec, close=True)
    _check(store, "project_memory_task (closed)")
    assert m["action"] == "closed"
    m = store.project_memory_task(key="mem-key-1", title="x", close=True)
    _check(store, "project_memory_task (noop)")
    assert m["action"] == "noop"

    # merge_tasks: empty list (no-op/early-return), then a real merge.
    store.merge_tasks([])
    _check(store, "merge_tasks (empty list, no-op)")
    tid_a = store.save_task(title="A", summary="sa", vector=[])
    _check(store, "save_task (A)")
    tid_b = store.save_task(title="B", summary="sb", vector=[])
    _check(store, "save_task (B)")
    merged_id = store.merge_tasks([tid_a, tid_b])
    _check(store, "merge_tasks (real merge)")
    assert merged_id

    store.rename_task_title(merged_id, "renamed title")
    _check(store, "rename_task_title")
    store.delete_task(tid_a)
    _check(store, "delete_task")

    # cleanup_junk_memory_tasks: no junk (no-op), then a real junk row.
    result = store.cleanup_junk_memory_tasks(dry_run=False)
    _check(store, "cleanup_junk_memory_tasks (no junk, no-op)")
    assert result["count"] == 0
    store.save_task(title="raw_filename_slug_2026", summary="s", vector=[],
                     tags="src:memory", session_id="")
    _check(store, "save_task (junk seed)")
    result = store.cleanup_junk_memory_tasks(dry_run=False)
    _check(store, "cleanup_junk_memory_tasks (real delete)")
    assert result["count"] == 1

    # --- prune_skills / dedupe_skills_by_content_hash ----------------------
    stale_path = str(tmp_path / "stale_skill_file_never_created.md")
    store.upsert_skill(
        Skill(id="test:stale", name="Stale", description="d", content="c",
              file_path=stale_path, plugin="test-plugin", target="claude"),
        content_hash="stale-hash",
    )
    _check(store, "upsert_skill (stale)")
    pruned_ids = store.prune_skills()
    _check(store, "prune_skills (real prune)")
    assert "test:stale" in pruned_ids
    store.prune_skills()
    _check(store, "prune_skills (no-op, nothing left)")

    store.upsert_skill(
        Skill(id="test:dup1", name="Dup1", description="d", content="c",
              file_path=str(tmp_path / "dup1.md"), plugin="p", target="claude"),
        content_hash="dup-hash",
    )
    _check(store, "upsert_skill (dup1)")
    store.upsert_skill(
        Skill(id="test:dup2", name="Dup2", description="d", content="c",
              file_path=str(tmp_path / "dup2.md"), plugin="p", target="claude"),
        content_hash="dup-hash",
    )
    _check(store, "upsert_skill (dup2)")
    deleted = store.dedupe_skills_by_content_hash()
    _check(store, "dedupe_skills_by_content_hash (real dedupe)")
    assert len(deleted) == 1
    store.dedupe_skills_by_content_hash()
    _check(store, "dedupe_skills_by_content_hash (no-op)")

    # --- pipeline / telemetry / logging ------------------------------------
    store.record_pipeline_run(
        "sess1", task_id, {"tier1": 10, "tier2": 20, "tier3": 30, "tier4": None},
        ["tier4"], top_similarity=0.5, token_cost_usd=0.01,
    )
    _check(store, "record_pipeline_run")
    store.log_interception("save_task", "preview text", 100)
    _check(store, "log_interception")
    store.log_context_injection("preview", skills=1, tasks=1, teachings=1,
                                 memory=0, precompacted=False, chars=100)
    _check(store, "log_context_injection")
    store.log_triage("preview", "local_answer", 0.9, 50)
    _check(store, "log_triage")

    # --- session context / tool examples ------------------------------------
    store.save_session_context("sess1", [skill_id], "summary", 5,
                                recent_messages=["hi"])
    _check(store, "save_session_context")
    store.save_tool_example("sess1", "Bash", "ls -la", output_summary="out",
                             context_hint="hint", repo_path=str(tmp_path),
                             category="shell")
    _check(store, "save_tool_example")
    store.save_tool_examples_batch([
        {"session_id": "sess1", "tool_name": "Read", "tool_input": "a.py",
         "output_summary": "", "context_hint": "", "repo_path": str(tmp_path),
         "category": "file"},
    ])
    _check(store, "save_tool_examples_batch")
    store.update_transcript_offset("sess1", 100)
    _check(store, "update_transcript_offset")

    # The two known #142 offenders, exercised again in-flow for good measure.
    pruned = store.prune_tool_examples(max_age_days=30, max_rows=5000)
    _check(store, "prune_tool_examples (in-flow, fresh rows so nothing prunable)")
    assert pruned == 0

    store.upsert_repo_context(str(tmp_path), commit_style="conventional",
                               common_commands="git,ls", project_summary="test",
                               tool_stats="{}")
    _check(store, "upsert_repo_context")

    store.save_skill_version("hygiene-skill", "{}", change_reason="test")
    _check(store, "save_skill_version")
    store.save_conversation_state("sess1", 5, "{}", "[]", None)
    _check(store, "save_conversation_state")

    # --- response / error caches -------------------------------------------
    cache_id = store.cache_response("query text", vec, "response text", session_id="sess1")
    _check(store, "cache_response")
    store.hit_response_cache(cache_id)
    _check(store, "hit_response_cache")
    store.invalidate_response_cache(cache_id)
    _check(store, "invalidate_response_cache")

    err_id = store.cache_error("error text", vec, "fix hint", session_id="sess1")
    _check(store, "cache_error")
    store.confirm_error_fix(err_id)
    _check(store, "confirm_error_fix")

    # --- message / prompt patterns ------------------------------------------
    pattern = store.record_message_pattern("hello world message", vec)
    _check(store, "record_message_pattern (new)")
    store.record_message_pattern("hello world message", vec)
    _check(store, "record_message_pattern (existing, increments count)")
    store.mark_pattern_skill_generated(pattern["id"])
    _check(store, "mark_pattern_skill_generated")
    store.save_prompt_pattern("trigger text", vec, "pattern text", context_type="refactor")
    _check(store, "save_prompt_pattern (new)")
    store.save_prompt_pattern("trigger text", vec, "pattern text", context_type="refactor")
    _check(store, "save_prompt_pattern (existing, increments count)")

    store.record_implicit_feedback("sess1", [skill_id], ["Bash: ls -la"])
    _check(store, "record_implicit_feedback")

    # --- background job queue: the second #142 offender, full lifecycle ----
    job_id = store.enqueue_job("test_kind", {"a": 1}, priority=5)
    _check(store, "enqueue_job")
    claimed = store.dequeue_job()
    _check(store, "dequeue_job (non-empty queue)")
    assert claimed is not None and claimed["id"] == job_id
    store.complete_job(job_id, result={"ok": True}, worker="test")
    _check(store, "complete_job")

    empty_claim = store.dequeue_job()
    _check(store, "dequeue_job (empty queue, in-flow)")
    assert empty_claim is None

    job_id2 = store.enqueue_job("test_kind2", {}, priority=5)
    _check(store, "enqueue_job (2)")
    store.dequeue_job()
    _check(store, "dequeue_job (claim job 2)")
    store.fail_job(job_id2, "boom", max_attempts=1)
    _check(store, "fail_job")
    store.reset_deferred_jobs()
    _check(store, "reset_deferred_jobs")

    # --- cron jobs -----------------------------------------------------
    cron_id = store.upsert_cron_job("test-cron", "desc", "* * * * *",
                                     "noop_command", {}, enabled=True)
    _check(store, "upsert_cron_job")
    store.toggle_cron_job(cron_id, False)
    _check(store, "toggle_cron_job")
    store.update_cron_job_status(cron_id, "success", duration_ms=10, error=None)
    _check(store, "update_cron_job_status")
    store.reset_cron_job_for_run(cron_id)
    _check(store, "reset_cron_job_for_run")
    store.update_cron_job(cron_id, description="new desc")
    _check(store, "update_cron_job")
    store.delete_cron_job(cron_id)
    _check(store, "delete_cron_job")

    # --- vectors / events ------------------------------------------------
    store.upsert_vector("test-namespace", "doc1", "some text")
    _check(store, "upsert_vector")
    store.append_event("sess1", "test.event", {"k": "v"})
    _check(store, "append_event")
    store.events_prune(dry_run=True)
    _check(store, "events_prune (dry_run, no candidates)")
    store.events_prune(dry_run=False)
    _check(store, "events_prune (real run, no candidates)")

    store.record_memory_audit(action="promote", namespace="skills", doc_id=skill_id,
                               from_level="L2", to_level="L3")
    _check(store, "record_memory_audit")

    # --- experiments / presets -------------------------------------------
    preset_id = store.save_preset("test-preset", "desc", {"a": 1})
    _check(store, "save_preset")
    exp_id = store.create_experiment("test-exp", preset_id, preset_id, target_runs=5)
    _check(store, "create_experiment")
    store.rate_experiment_run(1, 1)
    _check(store, "rate_experiment_run (nonexistent run id)")
    store.cancel_experiment(exp_id)
    _check(store, "cancel_experiment")
    store.delete_preset(preset_id)
    _check(store, "delete_preset")

    # --- teardown of remaining state ----------------------------------
    removed = store.remove_teaching(teaching_id)
    _check(store, "remove_teaching")
    assert removed is True
    store.close_task(task_id, compact="done")
    _check(store, "close_task")
    store.delete_skill(skill_id)
    _check(store, "delete_skill")
