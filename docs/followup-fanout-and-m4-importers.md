# Followup — Fanout dispatch + M4 ruflo importers

Captures what shipped, what's held, and what's left after the first parallel-issue fanout exercise. Tracks issues #23, #24, #25.

## What shipped (committable)

| Component | Files | Status |
|---|---|---|
| Fanout MCP feature | `src/skill_hub/fanout/`, `src/skill_hub/server.py` (3 tools), `src/skill_hub/store.py` (tag filter), `src/skill_hub/config.py` (defaults), `tests/fanout/` (23 tests) | Green; ready |
| Slash command | `commands/hub-fanout.md` | Installed; tested live |
| #25 docs flip markers | `docs/comparison-ruflo.md` (M4-FLIP inline comments) | Markers in place; full flip deferred until M4-1..M4-5 land |

## What's held (license gate)

Issues #23 and #24 produced working code that **is not yet committed** pending a license-attribution decision:

- `scripts/import_ruflo_agents.py` + `tests/test_import_ruflo_agents.py` + fixtures
- `scripts/import_ruflo_skills.py` + `tests/test_import_ruflo_skills.py` + fixtures

**Why held:** ruflo (claude-flow) is MIT-licensed. MIT §2 requires the copyright + permission notice in "copies or substantial portions". The importers produce derivative output (verbatim agent prompts, skill bodies) without injecting that notice. Until the importers emit attribution into each output file (and/or a sibling `LICENSE-ruflo` in the output dir), shipping them risks downstream users producing un-attributed copies of ruflo content.

**Note on inbound compatibility:** skill-hub is already Apache-2.0. Inbound MIT into outbound Apache-2.0 is fine — MIT notices belong in `NOTICE` for any redistributed MIT-derived material.

## Known gaps in the held importers

| # | Gap | Fix |
|---|---|---|
| G1 | `import_ruflo_skills.py` assumes `<root>/skills/<name>/skill.yaml`. Real layout: `<root>/<plugin>/<version>/skills/<name>/`. Result: runs empty against a real install. | Update discovery to walk `<root>/<plugin>/*/skills/<name>/`. Mirror the agent importer's traversal. |
| G2 | Neither importer injects MIT/copyright into output files. | Emit a `# Source: ruflo <plugin>:<name>, MIT © 2024-2026 ruvnet` header per output file, plus one `LICENSE-ruflo.md` at the output root. |
| G3 | Fixtures for both importers landed in `tests/fixtures/ruflo-fake/` from different agent worktrees — distinct subtrees but shared parent. Merge needs both. | Trivial — let `git merge` resolve; no overlap. |

## Fanout design wrinkles (P2)

| # | Issue | Notes |
|---|---|---|
| F1 | Skill-hub creates a per-fanout bookmark worktree for each task, but the Agent tool's `isolation: "worktree"` creates its own harness-managed worktree alongside. The bookmarks end up empty duplicates. | Pick one: (a) drop `isolation: "worktree"` from the directive and have agents work in the bookmark; (b) drop the bookmark creation and just record `worktree_path` after the agent finishes; (c) keep both — bookmark = state, harness wt = execution — and add a cleanup pass. (c) is least invasive. |
| F2 | After a real fanout run, the 3 bookmark worktrees stay on disk indefinitely. | Add `fanout_cleanup(group_id)` MCP tool that runs `git worktree remove` on every bookmark associated with a closed group. |
| F3 | MCP tool registration requires server restart to expose newly added `@mcp.tool()` functions. No in-session reload. | Document in `docs/installation.md` troubleshooting; consider a `mcp_reload` MCP tool that exec's a clean reimport. |

## Stale state to clean up

After a fanout run, the following residue is expected:

- N skill-hub tasks open (one per issue), tagged `fanout:<group-id>`
- N empty bookmark worktrees created by the coordinator (one per task)
- N harness-managed worktrees (one per dispatched agent) — carry the actual work
- Incidental `uv.lock` modifications in each harness worktree (from `uv pip install pytest`) — discard

After the safe pile commits, the cleanup sequence is:

1. Cherry-pick / merge the #25 markers into main.
2. `fanout_close <group-id> …` to close the per-issue skill-hub tasks.
3. Either complete the importer work per G1/G2 above and merge the importer worktrees, or `git worktree remove` them and re-do later.
4. `git worktree remove` for the bookmark worktrees once their branches are merged or dropped.

## Followup plan — prioritized

**P0 (this session, on user approval)**
- [x] Commit fanout feature + slash command + tests
- [x] Commit #25 M4-FLIP markers
- [x] Commit this followup plan
- [ ] Close the live skill-hub fanout group (run `fanout_status` then `fanout_close <group-id>`)

**P1 (license + importers)**
- [ ] Decide attribution mechanism: header-in-each-output + `LICENSE-ruflo.md` in output dir
- [ ] Fix G1 (skills importer path discovery)
- [ ] Fix G2 (MIT attribution injection)
- [ ] Re-run importers against real ruflo cache; verify outputs carry attribution
- [ ] Commit importers + fixtures together
- [ ] Add `NOTICE` and `CITATION.cff` at repo root; add `## License & Citation` section to README

**P2 (fanout cleanup)**
- [ ] Resolve F1 bookmark-vs-harness worktree duplication
- [x] Implement F2 `fanout_cleanup(group_id)` — bulk inverse of `fanout_issues`; closes tasks + removes worktrees + deletes branches; idempotent
- [x] Document F3 MCP-reload workflow (see `docs/installation.md` Troubleshooting)

**P3 (smoke + polish)**
- [ ] End-to-end smoke: fresh fanout against a different repo with real GH issues
- [ ] Add cross-repo fanout (currently single-project per call)
- [ ] Pluggable adapters: Linear, Jira (interfaces already exist in `sources.py`)
