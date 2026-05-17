# Followup — Fanout dispatch + M4 ruflo absorption

Captures what shipped, what's held, and what's left after the parallel-issue fanout exercises that cleared M4. Tracks issues #20-#25.

**M4 status (2026-05-17):** all 6 issues CLOSED on GitHub. Real residual work below.

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

## What shipped (swarm + autopilot + federation, commit `0647ac0`)

- `src/skill_hub/swarm.py` — `swarm_launch` / `swarm_reap` MCP tools; per-claim `subprocess.Popen`, per-claim log files under `~/.claude/mcp-skill-hub/swarm/<group>/<claim>.log`, non-blocking reap.
- `src/skill_hub/autopilot/` — `claims.py` + `loop.py` + MCP tools `autopilot_run` / `autopilot_stop` + matching CLI subcommands. Foreground loop drains the claims board with SIGINT-clean exit and a SQLite stop flag for cross-process stop.
- `src/skill_hub/store.py` — SQLite WAL mode (idempotent), `node_id` column on `tasks`, new `events` table per M2 design, `federation_view(remote_db_path)` ATTACHes peer DB read-only; MCP tool of the same name.
- Tests: `tests/test_swarm.py` (12), `tests/autopilot/test_loop.py` (13), `tests/test_federation_lite.py` (13). Full suite green: 665 passed, 1 skipped.

Closes #20 (swarm-lite), #21 (autopilot-lite), #22 (federation-lite).

## Outstanding gaps

| # | Gap | Status | Fix |
|---|---|---|---|
| G1 | Skills importer assumed `<root>/skills/<name>/skill.yaml`. Real layout: `<root>/<plugin>/[<version>/]skills/<name>/`. | **Fixed** in `43d9e8a` — uses `rglob("SKILL.md")` + path inference. | — |
| G2 | Neither importer injects MIT/copyright into output files. | **Open** — legal risk. | Emit `# Source: ruflo <plugin>:<name>, MIT © 2024-2026 ruvnet` header per output file, plus one `LICENSE-ruflo.md` at the output root. Add `NOTICE` + `CITATION.cff` at repo root. |
| G3 | Fixture merge across worktrees. | **Resolved** — single curated fixture set in `tests/fixtures/ruflo-fake/`. | — |
| G4 | `docs/comparison-ruflo.md` still carries 6 `M4-FLIP` markers. #25 was closed shipping the CI-gate portion only; the doc rewrite itself was **never** delivered. | **Open — now unblocked** (all of #20-#24 closed). | Rewrite per the 6 M4-FLIP markers: swap "planned M4-X" cells for the shipped entrypoint names; flip "Verdict" column to "Native in skill-hub vX.Y"; update Migration path to point at `swarm_launch` / `autopilot_run` / `federation_view`. |
| G5 | Importer scripts have no CLI entrypoint in `[project.scripts]`. | **Intentional** per `docs/comparison-ruflo.md:84` — one-shot scripts. | None unless ergonomics wanted. |
| G6 | `src/skill_hub/autopilot/loop.py:52 default_launcher` is a placeholder shell-cmd runner. Comment explicitly says "until issue #20 wires the real swarm_launch subprocess." #20 closed without that wiring. | **Open** — drift will accumulate. | Replace `default_launcher` with `swarm.swarm_launch([Claim(...)])`; pass back `swarm_reap` result so the loop marks `done`/`failed` from the actual subprocess exit. |
| G7 | Autopilot defined its own `claims` table (`src/skill_hub/autopilot/claims.py`); swarm has its own `Claim` dataclass. Both are forward-compat scaffolding for #9 (m1 claims-board). | **Tracked** — will need alignment when #9 lands. | When #9 ships the canonical claims-board, port autopilot.claims to the SSOT schema and route swarm's `Claim` dataclass through `claims_load()`. |

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

**P2 — M4 lite stack (shipped 2026-05-17)**
- [x] #20 swarm-lite: launch N Claude subprocesses, each on a distinct worktree+claim (effort:L) — `0647ac0`
- [x] #21 autopilot-lite: pick-next-stealable loop, runs continuously (effort:M) — `0647ac0`
- [x] #22 federation-lite: WAL-mode + node_id for multi-host shared state (effort:M) — `0647ac0`
- [x] Single-attempt fanout — no compare/pick needed since each issue had one agent
- [x] Bundled consolidation commit (`server.py` 3-way merge done by hand) + ff-merge + push
- [x] `fanout_cleanup` ran on group `82b9822c`; 3 bookmark worktrees + 3 harness worktrees + 7 branches removed

**P2a — owed wiring after M4 closed (G4/G6/G7)**
- [ ] **G4** rewrite `docs/comparison-ruflo.md` per the 6 M4-FLIP markers. Final entrypoint names confirmed: `swarm_launch` / `swarm_reap`, `autopilot_run` / `autopilot_stop`, `federation_view`, `python scripts/import_ruflo_{skills,agents}.py`. This is the ask of #25 that was never actually delivered.
- [ ] **G6** replace `autopilot.loop.default_launcher` placeholder with `swarm.swarm_launch` + `swarm_reap` integration. Without this, the autopilot loop runs shell commands rather than spawning real `claude` subprocesses — the whole point of the swarm.
- [ ] **G7** when #9 (m1 claims-board) lands, port `autopilot.claims` to the SSOT schema and route swarm's `Claim` dataclass through it.

**P3 — fanout polish**
- [ ] Resolve F1 bookmark-vs-harness worktree duplication (currently every fanout leaves N empty bookmark worktrees because Agent's `isolation: "worktree"` creates its own)
- [ ] End-to-end smoke: fresh fanout against a different repo
- [ ] Add cross-repo fanout (currently single-project per call)
- [ ] Pluggable adapters: Linear, Jira (interfaces already exist in `sources.py`)

**P4 — process lessons (carry forward into a fanout-postmortem skill)**
- [ ] Codify "compare & pick" consolidation pattern as a skill (3 parallel fanout groups for the same 3 issues produced 9 distinct attempts; the consolidation step manually compared all 9 and picked one per issue)
- [x] **Lesson logged:** parenthetical `Closes #N` only closes the first ref. Use bare `Closes #N` per line. Confirmed working on `0647ac0` (all 3 closed automatically).
- [x] **Lesson logged:** single-attempt fanout (one fanout invocation, one agent per issue) skips the compare-and-pick burden but requires careful `server.py` merge by hand when multiple agents touch it. Auto-3-way patch apply failed; manual splice via Edit was the workable path.
- [ ] Implement `mcp_reload` MCP tool so new `@mcp.tool()` registrations are discoverable mid-session without restarting the server. Currently the autopilot/swarm/federation tools added in `0647ac0` aren't visible until the user restarts skill-hub.

**P5 — next milestone candidates** (no current commitment)
- M3 worktree-policy bundle: #15-#19 (cross-repo stale-import detector, memory-rule export, lint-canary, worktree-policy pre-flight)
- M1 foundation: #9 claims-board (unblocks G7), #11 worktree-aware tasks, #10 witness-log
- M2: #14 Managed-Agents design phase (effort:L, meta)
