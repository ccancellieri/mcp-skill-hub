# contradiction-detector

Bundled mcp-skill-hub plugin. Scans wiki pages for contradictory claims using LLM analysis, provides a review UI, and tracks resolutions.

## Features

- **Scheduled Detection**: Weekly scan of entity/concept wiki pages for contradictions
- **LLM-Powered Analysis**: Uses tier_mid model to compare claims across related pages
- **Review UI**: Side-by-side view of conflicting claims with resolution options
- **Resolution Tracking**: Records when and how contradictions were resolved

## Extension Points

- **A1 (web_mount)**: `/contradiction-detector` — Review and resolve UI
- **A5 (scheduled_tasks)**: Weekly `detect_contradictions` task
- **A7 (storage)**: `plugin_contradiction_findings` and `plugin_contradiction_runs` tables

## Usage

### Enable the Plugin

Add to your `~/.claude/settings.json`:

```json
{
  "enabledPlugins": ["contradiction-detector"]
}
```

### Run Detection Manually

```bash
cd /Users/ccancellieri/work/code/mcp-skill-hub
python -m plugins.contradiction-detector.scripts.detect_contradictions --dry-run
```

### Enable Scheduled Task

```python
from skill_hub import server
server.enable_plugin_task("contradiction-detector", "detect_contradictions")
```

### Via Web UI

Navigate to `/contradiction-detector/` in the skill-hub dashboard to:
1. View pending contradictions
2. Review claim pairs side-by-side
3. Resolve with: "Keep A", "Keep B", "Merge", or "Both Valid"

## Configuration

In `plugin.json`:

```json
{
  "config": {
    "llm_tier": "tier_mid",
    "batch_size": 10,
    "confidence_threshold": 0.7,
    "max_page_pairs": 100
  }
}
```

- `llm_tier`: Model tier for contradiction analysis (default: `tier_mid`)
- `batch_size`: Log progress every N page pairs
- `confidence_threshold`: Minimum confidence to record a finding (0.0–1.0)
- `max_page_pairs`: Maximum pairs to analyze per run

## Detection Logic

1. Load all entity and concept pages from wiki
2. Find related page pairs via `wiki_edges` (pages that link to each other)
3. For each pair, extract claims and compare via LLM
4. Store contradictions with confidence scores
5. User resolves via UI

## Tables

### `plugin_contradiction_findings`

| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER | Primary key |
| page_a | TEXT | First page slug |
| page_b | TEXT | Second page slug |
| claim_a | TEXT | Claim from page A |
| claim_b | TEXT | Conflicting claim from page B |
| confidence | REAL | Detection confidence (0.0–1.0) |
| resolution_status | TEXT | `pending` or `resolved` |
| resolution | TEXT | JSON resolution details |
| resolved_by | TEXT | Who resolved it |
| resolved_at | TEXT | ISO timestamp |
| detection_run | TEXT | Run ID |

### `plugin_contradiction_runs`

| Column | Type | Description |
|--------|------|-------------|
| id | TEXT | Run identifier |
| started_at | TEXT | Start timestamp |
| completed_at | TEXT | Completion timestamp |
| status | TEXT | `running`, `completed`, or `failed` |
| pages_scanned | INTEGER | Pages analyzed |
| pairs_analyzed | INTEGER | Page pairs compared |
| contradictions_found | INTEGER | Findings recorded |

## API Endpoints

- `GET /contradiction-detector/` — Review UI
- `GET /contradiction-detector/finding/{id}` — Detail view
- `GET /contradiction-detector/api/stats` — Stats JSON
- `GET /contradiction-detector/api/list` — Findings list JSON
- `POST /contradiction-detector/api/resolve` — Resolve a finding
- `POST /contradiction-detector/api/run-detection` — Trigger detection run
