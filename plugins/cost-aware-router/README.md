# cost-aware-router

Bundled mcp-skill-hub plugin. Tracks API costs across sessions, enforces budgets, and suggests cheaper model alternatives.

## Features

- **Cost Tracking**: Automatically logs model invocations, token usage, and costs
- **Budget Management**: Set limits per session, project, daily, or globally
- **Alternative Suggestions**: Recommends cheaper models for low-complexity tasks
- **Dashboard**: Visual cost breakdown by model, session, and time period

## Routes

| Method | Path | Description |
|--------|------|-------------|
| GET | `/cost-router/` | Cost dashboard with daily/session breakdowns |
| GET | `/cost-router/settings` | Budget configuration page |
| GET | `/cost-router/api/cost/session/{id}` | Cost for a specific session |
| GET | `/cost-router/api/cost/daily?date=YYYY-MM-DD` | Daily cost summary |
| GET | `/cost-router/api/cost/range?start=&end=` | Cost for a date range |
| GET | `/cost-router/api/cost/by-model` | Aggregated costs per model |
| GET | `/cost-router/api/budget/{scope}` | Get budget status |
| POST | `/cost-router/api/budget/{scope}?budget_usd=N` | Set budget limit |
| GET | `/cost-router/api/estimate?model=&input_tokens=&output_tokens=` | Pre-call cost estimate |
| GET | `/cost-router/api/pricing` | Current pricing table |

## Hooks

- `on_tool_call` (async): Logs LLM tool invocations and checks budget alerts

## Storage

Tables (prefixed with `plugin_cost_router_`):
- `cost_log`: Individual cost entries (session, model, tokens, cost, timestamp)
- `budget_limits`: Budget configuration per scope
- `daily_summary`: Pre-aggregated daily totals

## Pricing (USD per 1M tokens)

| Model | Input | Output |
|-------|-------|--------|
| claude-opus-4 | $15.00 | $75.00 |
| claude-sonnet-4 | $3.00 | $15.00 |
| claude-haiku-4 | $0.25 | $1.25 |
| haiku | $0.80 | $4.00 |
| sonnet | $3.00 | $15.00 |
| opus | $15.00 | $75.00 |
| deepseek-r1 | $0.55 | $2.19 |
| ollama/* | $0.00 | $0.00 |

## Scripts

```bash
# Estimate cost before calling
python plugins/cost-aware-router/scripts/estimate_cost.py \
  --model claude-opus-4 --input 1000 --output 500

# Estimate from prompt text
python plugins/cost-aware-router/scripts/estimate_cost.py \
  --model sonnet --prompt "Explain quantum computing"
```

## Configuration

Edit `plugin.json` to customize:
- `pricing`: Model cost rates
- `default_budget_usd`: Default budget for new scopes
- `budget_alert_threshold`: Percentage to trigger alerts (default 0.8)
- `suggest_downgrade_at`: Percentage to suggest cheaper models (default 0.7)

## Auto-registration

Discovered at startup when `bundled_plugins_enabled: true` in `~/.claude/mcp-skill-hub/config.json`.
