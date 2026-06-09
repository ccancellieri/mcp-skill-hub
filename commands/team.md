---
description: Drive specialized multi-agent orchestration across four task shapes — review, arch, issues, implement — with model·effort policy and upfront prompt refactoring.
---

Arguments: `$ARGUMENTS`

Parse `$ARGUMENTS`. If empty or `help`, print the **Help** block at the bottom and stop.

## Argument form

```
/team <review|arch|issues|implement> <target> [--effort low|medium|high|xhigh] [--estimate]
```

- `<kind>` — one of `review`, `arch`, `issues`, `implement`
- `<target>` — a PR number, issue number, file path, directory, or label filter; everything up to the first flag
- `--effort` — `low`, `medium`, `high`, or `xhigh` (default `xhigh` when omitted)
- `--estimate` — print the plan + cost estimate and STOP; no agents are spawned

Extract these four values from `$ARGUMENTS` before calling any tool.

---

## Agent roster (inline — dispatch is explicit before `team_plan` runs)

| Role | Agent name | Default model | One-line purpose |
|---|---|---|---|
| arch_analyst | `team-arch-analyst` | opus | read-only deep architecture + code analysis; cite `file:line`; never edits |
| reviewer | `team-reviewer` | opus (xhigh) / sonnet (high) | adversarial review; refute-by-default; severity ratings |
| code_implementer | `team-code-implementer` | sonnet | implement a clear spec; no scope creep |
| mechanical_refactorer | `team-mechanical-refactorer` | sonnet (xhigh) / haiku (low) | behavior-preserving rename/simplify; smallest diff |
| human_voice_writer | `team-human-voice-writer` | opus | first-person engineer prose; no AI tells |
| github_operator | `team-github-operator` | haiku | `gh` inspect/fetch/post; posts pre-written prose only |

The exact `cc_model` for each role at the requested effort level comes from `team_plan`; the table above shows the xhigh defaults as a reference.

---

## Steps — follow IN ORDER

### Step 1 — Refactor the prompt upfront (mandatory)

Call `improve_prompt` with the target plus any surrounding user context:

```
improve_prompt(text="<kind>: <target> [plus any extra user context from $ARGUMENTS]")
```

Use the enriched text returned as the working brief for all downstream steps. This is the whole point of the upfront refactor: a sharper brief costs almost nothing and compounds across every agent that reads it.

### Step 2 — Plan

Call `team_plan` with the parsed arguments:

```
team_plan(task_kind="<kind>", effort="<effort or xhigh>", estimate=<true iff --estimate>)
```

This returns the substrate (`team` or `workflow`), the role→agent→model roster, and the verification loop count.

### Step 3 — If `--estimate` was given

Print the plan + estimate returned by `team_plan` and **STOP**. Do not spawn anything.

### Step 4 — Dispatch by substrate

#### Substrate `team` — kinds `review` and `arch`

Agent teams require `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`. Check whether agent teams are available in this session. If they are not, fall back: dispatch the same roles as parallel subagents (Agent tool) that each report back, then synthesize the consolidated output centrally yourself.

**When teams are available:**

Create a Claude Code agent team. For each row in the roster spawn one teammate using:
- the exact agent type named in the roster (e.g. `team-reviewer`)
- the `cc_model` returned by `team_plan`
- the lens or layer assigned to that slot

For `review`: spawn four `team-reviewer` teammates, one per lens — `security`, `correctness`, `performance`, `tests`. Have each reviewer read the target independently and produce findings with severity ratings. Then run `loops` adversarial rounds: each reviewer reads the others' findings and challenges any they dispute, citing file and line. After the final round, `team-human-voice-writer` reads the reconciled findings and drafts the consolidated write-up in first-person engineer voice. If `<target>` identifies a PR or issue, `team-github-operator` posts the write-up there; otherwise print it.

For `arch`: spawn three `team-arch-analyst` teammates on distinct architectural layers (e.g. data model, API boundary, execution engine — adjust to the codebase) plus one `team-arch-analyst` tagged as devil's advocate whose job is to challenge every hypothesis. Have them debate competing explanations for `loops` rounds. `team-human-voice-writer` drafts the consolidated analysis. Post only if a PR/issue target was given.

#### Substrate `workflow` — kinds `issues` and `implement`

The Workflow tool is the recommended engine when available; fall back to sequential subagent dispatch otherwise.

For `issues`: run the deterministic triage pipeline in stage order:
1. `team-github-operator` (haiku) — fetch and triage the issues matching `<target>`; emit a structured list with labels, priority signals, and linked context
2. `team-arch-analyst` (opus at xhigh) — classify each issue by impact and cross-cutting concern; annotate the list
3. `team-human-voice-writer` (opus) — draft concise, first-person issue updates/comments for each item

Print the triage report and all drafted updates. Post to GitHub only if the user explicitly asks after seeing the output.

For `implement`: run the deterministic build pipeline in stage order:
1. `team-arch-analyst` — design: read the target, produce a concise spec with affected files and edge cases
2. `team-code-implementer` — build: implement the spec in an isolated worktree; follow existing patterns; no scope creep
3. `team-reviewer` — verify: review the diff for correctness, security, and tests; gate acceptance; repeat for `loops` passes (each pass either accepts or returns a fix list to `team-code-implementer`)
4. `team-mechanical-refactorer` — clean: behavior-preserving simplifications and renames; smallest possible diff
5. `team-human-voice-writer` — write: draft PR title, body, and commit message in first-person engineer voice; no AI attribution
6. `team-github-operator` — open the PR with the drafted prose

---

## Help

```
/team — specialized multi-agent orchestration

USAGE
  /team <kind> <target> [--effort low|medium|high|xhigh] [--estimate]

KINDS
  review     Adversarial code review — 4 lens reviewers challenge each other
  arch       Architecture analysis — competing hypotheses + devil's-advocate debate
  issues     Deterministic triage — fetch → classify → draft updates
  implement  Deterministic build — design → build → verify → clean → PR

OPTIONS
  --effort   Model floor + verification loops (default: xhigh)
             low    → haiku/sonnet floor, 0 verification loops
             medium → sonnet floor,       1 verification loop
             high   → sonnet/opus floor,  2 verification loops
             xhigh  → opus floor,         3 verification loops (default)
  --estimate Print roster + cost projection and stop — no agents spawned

AGENT ROSTER
  team-arch-analyst          opus       read-only architecture analysis
  team-reviewer              opus       adversarial review, severity ratings
  team-code-implementer      sonnet     implement a clear spec
  team-mechanical-refactorer sonnet     behavior-preserving cleanup
  team-human-voice-writer    opus       first-person engineer prose
  team-github-operator       haiku      gh inspect/post (never authors)

SUBSTRATE
  review, arch   → agent team (adversarial, live inter-agent messaging)
                   requires CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
                   falls back to parallel subagents if teams disabled
  issues, implement → Workflow tool (deterministic pipeline, resumable)
                      falls back to sequential subagents if Workflow unavailable

EXAMPLES
  /team review 142
  /team arch src/skill_hub/router
  /team issues mcp-skill-hub label:bug --estimate
  /team implement 49 --effort high
  /team review 200 --effort medium --estimate
  /team issues geoid label:bug is:open
```
