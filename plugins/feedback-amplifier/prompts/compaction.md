# Feedback Amplifier Memory Compaction Prompt

You are analyzing memory files from the feedback-amplifier plugin to determine retention decisions.

## Context

These files contain:
- Skill injection context (which skills were suggested for which queries)
- Usage patterns (which skills were used vs ignored)
- Domain-specific performance metrics
- EMA scores with decay tracking

## Retention Criteria

**KEEP** if:
- The skill has been injected ≥ 5 times in the last 30 days
- The skill has a usage ratio > 30% (used_count / injection_count)
- The skill has domain-specific performance data for multiple domains

**PRUNE** if:
- The skill has not been injected in 60+ days
- The skill has 0 usage across all injections
- The skill's EMA score has decayed below 0.4

**COMPACT** if:
- Multiple entries exist for the same skill_id in feedback_context
- Aggregate summary: keep skill_id, total_injections, total_used, domains, final_ema_score

## Output Format

Return a JSON object with decisions:
```json
{
  "decisions": [
    {
      "file": "feedback_context_2024-01.md",
      "action": "KEEP|PRUNE|COMPACT",
      "reason": "..."
    }
  ]
}
```
