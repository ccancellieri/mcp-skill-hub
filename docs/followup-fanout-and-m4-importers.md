# Followup — Fanout dispatch + M4 ruflo importers

Captures what shipped, what's held, and what's left after the first parallel-issue fanout exercise. Tracks issues #23, #24, #25.

## What shipped (committable)

| Component | Files | Status |
|---|---|---|
| Fanout MCP feature | `src/skill_hub/fanout/`, `src/skill_hub/server.py` (3 tools), `src/skill_hub/store.py` (tag filter), `src/skill_hub/config.py` (defaults), `tests/fanout/` (23 tests) | Green; ready |
| Slash command | `commands/hub-fanout.md` | Installed; tested live |
| #25 docs flip markers | `docs/comparison-ruflo.md` (M4-FLIP inline comments) | Markers in place; full flip deferred until M4-1..M4-5 land |

## What shipped (importers, commit `43d9e8a`)

- `scripts/import_ruflo_agents.py` + `tests/test_import_ruflo_agents.py` + fixtures
- `scripts/import_ruflo_skills.py` + `tests/test_import_ruflo_skills.py` + fixtures
- `.github/workflows/no-ruflo-dep.yml` + `tests/test_no_ruflo_dep.py` (CI grep gates)

Closes #23 (skills importer), #24 (agents importer), #25 (CI gates portion only).
Skills importer fixed G1 by walking `root.rglob("SKILL.md")` rather than the old `<root>/skills/<name>/skill.yaml` assumption.

## Outstanding gaps

| # | Gap | Status | Fix |
|---|---|---|---|
| G1 | Skills importer assumed `<root>/skills/<name>/skill.yaml`. Real layout: `<root>/<plugin>/[<version>/]skills/<name>/`. | **Fixed** in `43d9e8a` — uses `rglob("SKILL.md")` + path inference. | — |
| G2 | Neither importer injects MIT/copyright into output files. | **Open** — legal risk. | Emit `# Source: ruflo <plugin>:<name>, MIT © 2024-2026 ruvnet` header per output file, plus one `LICENSE-ruflo.md` at the output root. Add `NOTICE` + `CITATION.cff` at repo root. |
| G3 | Fixture merge across worktrees. | **Resolved** — single curated fixture set in `tests/fixtures/ruflo-fake/`. | — |
| G4 | `docs/comparison-ruflo.md` still carries 6 `M4-FLIP` markers awaiting #20-#22 (swarm-lite, autopilot-lite, federation-lite). #25's doc-flip ask was **not** fully delivered — only the CI grep gates portion shipped. | **Deferred** until #20-#22 land. | Rewrite per the M4-FLIP markers when the three remaining M4 issues close. |
| G5 | Importer scripts have no CLI entrypoint in `[project.scripts]`. Invocation is `python scripts/import_ruflo_*.py`. | **Intentional** per `docs/comparison-ruflo.md:84` — one-shot scripts, never imported as Python. | None unless the user wants `skill-hub import-ruflo-{skills,agents}` ergonomics. |

**License posture (G2 context):** ruflo is MIT-licensed; skill-hub is Apache-2.0. Inbound MIT into outbound Apache-2.0 is fine — MIT notices belong in `NOTICE` for any redistributed MIT-derived material. The importers currently *produce* derivative output (canonical SKILL.md + subagent YAML) without injecting that notice. Downstream users running the importers will produce un-attributed copies of ruflo content.

## Fanout design wrinkles (P2)

| # | Issue | Notes |
|---|---|---|
| F1 | Skill-hub creates a per-fanout bookmark worktree for each task, but the Agent tool's `isolation: "worktree"` creates its own harness-managed worktree alongside. The bookmarks end up empty duplicates. | Pick one: (a) drop `isolation: "worktree"` from the directive and have agents work in the bookmark; (b) drop the bookmark creation and just record `worktree_path` after the agent finishes; (c) keep both — bookmark = state, harness wt = execution — and add a cleanup pass. (c) is least invasive. |
| F2 | After a real fanout run, the 3 bookmark worktrees stay on disk indefinitely. | Add `fanout_cleanup(group_id)` MCP tool that runs `git worktree remove` on every bookmark associated with a closed group. |
| F3 | MCP tool registration requires server restart to expose newly added `@mcp.tool()` functions. No in-session reload. | Document in `docs/installation.md` troubleshooting; consider a `mcp_reload` MCP tool that exec's a clean reimport. |

## Stale state cleaned up (2026-05-17 session)

After three back-to-back fanout runs against #23/#24/#25 (9 agents total, 3 groups):

- ✅ Picks consolidated into a single commit `43d9e8a` on `main` (skills importer, agents importer, CI gates).
- ✅ All three fanout groups closed (`fd92ed4b`, `782dc3cc`, `94c821c9` — 9 tasks total).
- ✅ 9 `worktree-agent-*` worktrees + 3 `cc/issue-*` bookmark worktrees removed; 13 stale local branches deleted.
- ✅ Pushed to `origin/main`; #23 auto-closed, #24/#25 manually closed referencing `43d9e8a`.

Residue pattern (carries forward — fold into a future `fanout-postmortem` skill):

- N skill-hub tasks open (one per issue), tagged `fanout:<group-id>`
- N bookmark worktrees from the coordinator (empty after dispatch — agents work in harness-managed worktrees)
- Incidental `uv.lock` modifications in each harness worktree (from `uv pip install pytest`) — discard

## Followup plan — prioritized

**P0 — shipped this session**
- [x] Commit fanout feature + slash command + tests (`c2dfa5b`)
- [x] Commit #25 M4-FLIP markers (`fb6502f`)
- [x] Commit this followup plan
- [x] Implement `fanout_cleanup(group_id)` (`ca24810`)
- [x] Document MCP-reload workflow (`aee0ae9`)
- [x] Ship importers + CI gates (`43d9e8a`); close #23/#24/#25
- [x] Cleanup all stale fanout worktrees + branches

**P1 — license attribution (G2; blocks re-running importers against real ruflo cache)**
- [ ] Inject `# Source: ruflo <plugin>:<name>, MIT © 2024-2026 ruvnet` header into every output file
- [ ] Emit `LICENSE-ruflo.md` at the output root of both importers
- [ ] Add `NOTICE` + `CITATION.cff` at repo root
- [ ] Add `## License & Citation` section to README
- [ ] Re-run importers against real ruflo cache; verify outputs carry attribution

**P2 — remaining M4 work (blocks G4 doc flip)**
- [ ] #20 swarm-lite: launch N Claude subprocesses, each on a distinct worktree+claim (effort:L)
- [ ] #21 autopilot-lite: pick-next-stealable loop, runs continuously (effort:M)
- [ ] #22 federation-lite: WAL-mode + node_id for multi-host shared state (effort:M)
- [ ] G4 — rewrite `docs/comparison-ruflo.md` per the 6 M4-FLIP markers once #20-#22 land

**P3 — fanout polish**
- [ ] Resolve F1 bookmark-vs-harness worktree duplication (currently every fanout leaves N empty bookmark worktrees because Agent's `isolation: "worktree"` creates its own)
- [ ] End-to-end smoke: fresh fanout against a different repo
- [ ] Add cross-repo fanout (currently single-project per call)
- [ ] Pluggable adapters: Linear, Jira (interfaces already exist in `sources.py`)

**P4 — process lessons from this exercise**
- [ ] Codify "compare & pick" consolidation pattern as a skill (3 parallel fanout groups for the same 3 issues produced 9 distinct attempts; the consolidation step manually compared all 9 and picked one per issue — repeatable but not yet a documented workflow)
- [ ] When `Closes #N` references in a commit message use parenthetical descriptions (`Closes #23 (skills importer), #24 ...`), GitHub auto-closes only the first. Either drop parens or close manually — observed on `43d9e8a`.
