# Learning

Skill Hub gets measurably smarter over time by combining **five signals** — the mix is what makes the suggestions feel aware of you.

## The five signals

| # | Signal | How it's captured | What it drives |
|---|--------|-------------------|----------------|
| 1 | **Teachings** (explicit) | `teach("when X", "suggest Y")` | Embedded and matched semantically at ~0.6 threshold |
| 2 | **Feedback** (semi-explicit) | `record_feedback(skill, helpful=True)` | EMA score on each skill (0.5–1.5) — multiplies cosine similarity |
| 3 | **Session history** (passive) | Stop hook logs tools actually called | Per-plugin usage patterns |
| 4 | **Context Bridge** (live capture) | JSONL transcript tail → `tool_examples` | `{tool_examples}`, `{session_context}`, `{repo_context}`, `{tool_patterns}` injected into local skills |
| 5 | **Skill Evolution** (shadow learning) | Session-end LLM compares local skill output vs Claude's tool usage | Local skills automatically versioned and improved |

Plugin suggestion score: `total = embed_sim + teaching_boost + session_boost`.

---

## 1. Teaching rules

Add persistent rules that match semantically — not keywords:

```python
teach(rule="when I give a URL to check",    suggest="chrome-devtools-mcp")
teach(rule="working on Terraform infra",    suggest="terraform")
teach(rule="debugging CSS or layout issues", suggest="chrome-devtools-mcp")
teach(rule="writing a Telegram bot",         suggest="telegram")
```

Future queries like "inspect this page" match "when I give a URL" at ~0.8 similarity.

```python
list_teachings()          # see all rules
forget_teaching(2)        # remove rule #2
```

---

## 2. Plugin suggestions

Disabled plugins are still suggested when they match — never loses your knowledge of what's installed:

```
suggest_plugins("take a screenshot of this page and check accessibility")
# → [DISABLED] chrome-devtools-mcp: Browser DevTools…
#   → to enable: toggle_plugin("chrome-devtools-mcp", enabled=True)
```

---

## 3. Feedback EMA

Rate skills after use — rankings improve for similar future queries:

```python
record_feedback(skill_id="superpowers:systematic-debugging", helpful=True)
```

Stored as an **EMA score** on each skill (`feedback_score` column, range 0.5–1.5). Every positive signal nudges the score up; negative nudges it down. The score multiplies cosine similarity at search time — no extra queries, no per-search cosine scans.

### Implicit feedback (automatic)

Runs at session end — no manual calls needed. The hook correlates loaded skills against tools Claude actually called:

```
Loaded skills: [hub-status, feature-dev:code-architect, superpowers:brainstorm]
Tools used:    [mcp__skill-hub__status, Edit, Read, Write]

→ hub-status domain "status" matches "mcp__skill-hub__status" → +positive EMA
→ feature-dev & superpowers unrelated to actual tools → -negative EMA
```

Over time this **self-tunes** the skill index to your actual usage with zero effort.

---

## 4. Context Bridge (see [advanced/context-bridge.md](../advanced/context-bridge.md))

Every Claude response is captured and stored at zero token cost. Local skills receive Claude's tool patterns as context variables.

---

## 5. Skill Evolution — shadow learning

Local skills can **automatically improve themselves** by observing what Claude does.

```
Claude works (edits files, runs commands, commits…)
    ↓ captured by Context Bridge
tool_examples DB — per session, categorized by type
    ↓
Session ends → _evolve_skills() runs
    ↓
For each "shadow": true skill:
  1. Match skill domain to Claude's tool_examples from this session
  2. Ask local LLM: "How should this skill change to match Claude's approach?"
  3. If change proposed:
     a. Snapshot current skill JSON → skill_versions table (rollback safety)
     b. Write improved skill JSON to ~/.claude/local-skills/
    ↓
Next run → skill uses improved prompts
    ↓
After N sessions → local skill quality converges toward Claude's
```

### Enable evolution for a skill

Add `"shadow": true`:

```json
{
  "name": "git-commit",
  "shadow": true,
  "description": "Smart git commit…",
  "steps": [...]
}
```

Or enable for all skills at once (more aggressive):

```
configure(key="skill_evolution_auto", value="true")
```

### Version history

Every evolution is snapshotted before writing:

```sql
SELECT version, change_reason, created_at
  FROM skill_versions
 WHERE skill_name = 'git-commit'
 ORDER BY version;

-- Restore version 2
SELECT skill_json FROM skill_versions
 WHERE skill_name = 'git-commit' AND version = 2;
```

### Log output

```
SKILL [git-commit] EVOLVE  v0→v1  →  updated git-commit.json  (ok)
SKILL [git-push]   EVOLVE  no changes needed  (ok)
EVOLVE  Shadow learning: 1 skill(s) evolved in session a3f9b2c1
```

### Control

| Config | Default | Effect |
|--------|---------|--------|
| `skill_evolution_enabled` | `true` | Master switch |
| `skill_evolution_auto` | `false` | Evolve **all** skills vs only `shadow:true` |
| `skill_evolution_max_per_session` | `3` | Max skills evolved per session |
| `skill_evolution_min_session_msgs` | `5` | Min messages before evolution runs |

You stay in full control: evolution only runs at session end (not live), requires substantial sessions (`≥5 messages`), is capped per session, and every change is versioned and reversible.

---

## Related

- [advanced/context-bridge.md](../advanced/context-bridge.md) — how Claude's behavior is captured
- [advanced/fine-tuning.md](../advanced/fine-tuning.md) — export the accumulated signal as training data
