# feedback-amplifier

Enhanced EMA feedback scoring for mcp-skill-hub with cross-session learning, skill context capture, and decay for unused skills.

## Purpose

This plugin enhances the existing skill feedback system by:

1. **Capturing context** when a skill is suggested/injected
2. **Correlating with actual usage** at session end
3. **Applying EMA scoring** with configurable parameters
4. **Decaying unused skills** over time
5. **Tracking domain-specific performance** for better recommendations

## Extension Points

- **A3 (hooks)**: `on_skill_activated` and `on_session_end` 
- **A7 (storage)**: Plugin-scoped tables for feedback context and scores
- **A10 (memory_optimizer)**: Compaction rules for long-term learning

## How It Works

### on_skill_activated Hook

Fires when a skill is loaded for injection into the system prompt. Captures:
- `skill_id` — which skill was suggested
- `session_id` — current session
- `query` — the user query that triggered the suggestion
- `domain_hints` — domain context (e.g., "ogc", "api")
- `injection_id` — reference to the core skill_injections table

Stores this in `plugin_fbamp_feedback_context` with `was_used=0` (pending).

### on_session_end Hook

Fires once when the session ends. For each skill that was injected:

1. Queries `events` for `skill.used` events
2. Updates `was_used` (1=used, -1=ignored) in feedback_context
3. Updates EMA score:
   - **Used**: boost score toward 2.0 (max)
   - **Ignored**: reduce score toward 0.3 (min)
4. Updates domain-specific performance metrics
5. Syncs with main `skills.feedback_score` for search ranking

### compute_scores.py

Periodic script to:

1. Apply exponential decay based on days since last use
2. Recalculate EMA scores from usage ratios
3. Clean up stale entries (>90 days old)

Decay formula:
```
decayed_score = old_score * exp(-days_since_use / half_life_days)
new_score = decayed_score * (1 - alpha) + feedback_signal * alpha
```

## Configuration

Add to your plugin config:

```json
{
  "path": "/path/to/feedback-amplifier",
  "enabled": true,
  "config": {
    "decay_half_life_days": 30,
    "ema_alpha": 0.15,
    "min_injections_for_decay": 3,
    "boost_on_skill_used": 0.1,
    "penalty_on_injection_not_used": 0.05
  }
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `decay_half_life_days` | 30 | Days for score to decay to 50% if unused |
| `ema_alpha` | 0.15 | Smoothing factor for EMA (0.1 = slow, 0.3 = fast) |
| `min_injections_for_decay` | 3 | Minimum injections before decay applies |
| `boost_on_skill_used` | 0.1 | Score boost when a skill is actually used |
| `penalty_on_injection_not_used` | 0.05 | Score penalty when injected but not used |

## Database Tables

### plugin_fbamp_feedback_context

Tracks each skill injection and whether it was used:

```sql
skill_id        -- which skill
session_id      -- which session
query           -- triggering query
domain_hints    -- JSON array of domain context
injection_id    -- FK to skill_injections.id
was_used        -- 0=pending, 1=used, -1=ignored
ts              -- timestamp
```

### plugin_fbamp_skill_scores

Aggregated scores per skill:

```sql
skill_id        -- primary key
ema_score       -- current EMA score (0.3-2.0)
last_used_at    -- when last skill.used event
injection_count -- total injections
used_count      -- total times used
decay_applied_at -- last decay computation
```

### plugin_fbamp_domain_performance

Per-domain success rates:

```sql
skill_id        -- which skill
domain          -- domain hint (e.g., "ogc")
success_count   -- times used in this domain
total_count     -- total injections for this domain
```

## Scheduled Tasks

To run score computation periodically:

```bash
# Via skill-hub CLI
skill-hub-cli run_script --plugin feedback-amplifier --script scripts/compute_scores.py

# Or enable as scheduled task (A5)
# In plugin.json add:
"scheduled_tasks": [
  {
    "name": "daily-score-compute",
    "cron": "0 3 * * *",
    "script": "scripts/compute_scores.py",
    "enabled_default": false
  }
]
```

## Integration with Core

The plugin syncs with the main `skills.feedback_score` column:

- When a skill is used, calls `record_feedback(skill_id, helpful=True)` logic
- When a skill is ignored, applies negative feedback
- This ensures search ranking (`store.search()`) benefits from the enhanced scoring

## Logging

Logs are written to:
- `~/.claude/mcp-skill-hub/logs/fbamp-hook.log` — hook activity
- `~/.claude/mcp-skill-hub/logs/fbamp-scores.log` — score computation

## Installation

1. Add plugin path to config:
```python
configure(key="extra_plugin_dirs", value='[{"path": "/path/to/feedback-amplifier", "enabled": true}]')
```

2. Restart skill-hub or run:
```python
index_plugins()
index_skills()
```

3. Verify tables were created:
```python
store.plugin_db("fbamp").fetch_all("SELECT * FROM plugin_fbamp_skill_scores LIMIT 5")
```
