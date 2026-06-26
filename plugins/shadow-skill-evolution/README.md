# Shadow Skill Evolution Plugin

Analyze tool-call patterns from session history and propose new local skills for common workflows.

## Overview

This plugin implements **A3 (hooks)**, **A5 (scheduled_tasks)**, and **A7 (storage)** extension points to:

1. **Capture** tool-call sequences at session end
2. **Cluster** similar patterns via embedding similarity
3. **Propose** new local skills when patterns repeat
4. **Approve/Reject** proposals via web dashboard

## Components

### Hook: `on_session_end`

- Extracts tool invocations from the `events` table
- Creates tool-chain fingerprints: `[tool_name, args_hash]` tuples
- Stores in `tool_chains` table for later analysis

### Scheduled Task: `propose_skills`

Runs weekly (Sunday 3 AM) to:

1. Embed unembedded tool chains
2. Cluster chains by cosine similarity (threshold: 0.85)
3. When cluster size ≥ 3, use local LLM to generate skill proposal
4. Store proposal in `skill_proposals` table

### Web Dashboard

Mount point: `/shadow-skill-evolution`

- View pending/approved/rejected proposals
- Approve → writes skill to `~/.claude/local-skills/{name}.json`
- Reject → marks as rejected

## Database Schema

### `tool_chains`

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `session_id` | TEXT | Source session |
| `chain_hash` | TEXT | SHA1 fingerprint |
| `tool_sequence` | TEXT | JSON array of `[tool_name, args_hash]` |
| `embedding` | TEXT | JSON array of floats |
| `occurrence_count` | INTEGER | Times seen in this session |
| `metadata` | TEXT | JSON (topic, summary) |

### `skill_proposals`

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary key |
| `name` | TEXT | Skill slug |
| `title` | TEXT | Human-readable title |
| `description` | TEXT | What the skill does |
| `triggers` | TEXT | JSON array of trigger phrases |
| `steps` | TEXT | JSON array of skill steps |
| `source_chains` | TEXT | JSON array of chain hashes |
| `cluster_size` | INTEGER | Number of sessions in cluster |
| `status` | TEXT | pending/approved/rejected/expired |

## Configuration

In `plugin.json`:

```json
{
  "hooks": {
    "on_session_end": "hooks/on_session_end.py"
  },
  "scheduled_tasks": [
    {
      "name": "propose_skills",
      "schedule": "0 3 * * 0",
      "script": "scripts/propose_skills.py"
    }
  ]
}
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard UI |
| POST | `/approve/{id}` | Approve proposal |
| POST | `/reject/{id}` | Reject proposal |
| GET | `/api/proposals` | List proposals JSON |
| GET | `/api/chains` | List tool chains JSON |
| DELETE | `/api/proposals/{id}` | Delete proposal |

## Manual Execution

```bash
# Run skill proposal generation
skill-hub-cli run-plugin-task shadow-skill-evolution propose_skills

# Or directly
python plugins/shadow-skill-evolution/scripts/propose_skills.py
```

## Flow Diagram

```
Session End
    │
    ▼
on_session_end.py
    │ Extract tool_chain from events
    │ Hash and store in tool_chains
    ▼
tool_chains table
    │
    ▼
propose_skills.py (scheduled)
    │ Embed chains
    │ Cluster by similarity
    │ LLM generates proposal
    ▼
skill_proposals table
    │
    ▼
Web Dashboard (/shadow-skill-evolution)
    │ User approves
    ▼
~/.claude/local-skills/{name}.json
```

## Thresholds

| Setting | Default | Description |
|---------|---------|-------------|
| `CLUSTER_THRESHOLD` | 3 | Min chains to form cluster |
| `SIMILARITY_THRESHOLD` | 0.85 | Cosine similarity cutoff |

Adjust in `scripts/propose_skills.py` constants.
