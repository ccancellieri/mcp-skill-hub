---
name: memory-layer-analysis
description: Use when reviewing auto-memory or project memory files to decide what to keep, archive, or consolidate. Categorization methodology behind master-state-compaction. Apply when the user asks to "audit memory", "find stale memory entries", "what should I archive", or before any large memory rewrite. Forces explicit per-entry classification (ACTIVE / SUPERSEDED / ARCHIVE / CONSOLIDATE) with stated reason rather than vibes-based pruning.
---

# Memory Layer Analysis

Memory pools (auto-memory, `.memory/decisions.md`, `MEMORY.md`) accumulate one entry per task. Without periodic categorization, the volume of historical noise drowns out the few load-bearing principles. This skill is the methodology that drives that categorization.

## The four categories

For each candidate entry, classify as exactly one:

1. **ACTIVE** — describes current behavior, ongoing work, or a load-bearing principle. Keep verbatim.
2. **SUPERSEDED** — a more recent entry contradicts or replaces it. Archive (move to `_archive/`); the new entry is the truth.
3. **ARCHIVE** — historically true but no longer load-bearing. SHIPPED one-shots > 14 days old, completed plans, or fix recipes whose lesson lives in a more general patterns file.
4. **CONSOLIDATE** — one of N entries forming a family (notebooks, routing, events). Pick one canonical and archive the siblings; or merge the unique content into the canonical.

## Weighting rules

- **Recent decisions outweigh older ones.** A 2026-04-29 entry that contradicts a 2026-03-15 entry wins; the older is officially ARCHIVED.
- **Architectural principles outweigh tactical fixes.** "Drivers self-contained" stays; "tasks_trigger_deadlock_fix" archives.
- **Feedback rules NEVER archive.** They're small, hard-won, and the user paid for each one with a correction.
- **In-flight work is ACTIVE even if old.** Anything marked `IN-PROGRESS` or `IN-FLIGHT` stays until proved shipped or abandoned.

## Per-entry classification template

For each candidate file, fill out:

```
filename: project_geoid_X.md
last_modified: YYYY-MM-DD
category: ACTIVE | SUPERSEDED | ARCHIVE | CONSOLIDATE
reason: <one sentence — why this category>
action: <keep | move to _archive/ | merge into project_geoid_Y.md>
```

If you can't fill out `reason` in one sentence, you don't understand the entry well enough to classify it. Read it again.

## Family detection

Group by filename prefix and look for:
- Multiple entries with same root (`notebook_*`, `routing_*`, `events_*`, `dimensions_*`)
- Sequential phases of same work (`asset_platform_phase1`, `_phase2`, `_phase3`)
- Date-stamped versions of the same topic (`baseline_2026_04_19`, `_19_cont`)

Within a family: keep the canonical (most recent comprehensive entry); archive the rest.

## Output format

A categorization table for the user to review BEFORE any file moves:

| File | Category | Reason | Action |
|------|----------|--------|--------|
| project_X.md | SUPERSEDED | Replaced by project_Y.md (2026-04-28) | move to _archive/ |
| project_Y.md | ACTIVE | Current behavior, recent work | keep |

Do NOT execute the actions until the user approves the table. Memory edits are high-leverage; classification mistakes are visible only when the next session starts cold.

## Common mistakes

- **Classifying SHIPPED as ARCHIVE too eagerly.** A SHIPPED entry from 5 days ago may still be the only documented behavior — wait for the architectural pattern to stabilize before archiving.
- **Missing in-flight work.** Always check open PRs / branches before marking something SUPERSEDED — the "supersession" may be in flight, not landed.
- **Promoting tactical fixes to invariants.** If the same bug-fix lesson appears in 3 entries, the lesson belongs in `patterns.md`, not in a permanent project entry.
- **Vibes-based pruning.** "This looks old" is not a reason. State the contradicting newer entry, the supersession date, or the SHIPPED-but-irrelevant rationale.

## Related skills

- `master-state-compaction` — uses this categorization methodology internally to produce the snapshot.
