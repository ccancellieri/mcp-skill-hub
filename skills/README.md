# Local Skills

Local skills are JSON-defined workflows executed by the local LLM (Ollama) rather than injected into Claude's context. They enable automation of repetitive tasks without consuming Claude tokens.

## Location

Local skills are stored as JSON files in the configured `local_skills_dir` (default: `~/.claude/local-skills/`).

## Skill Format

```json
{
  "name": "skill-name",
  "description": "What the skill does",
  "triggers": ["phrase 1", "phrase 2"],
  "steps": [...],
  "output": "Template with {variable} placeholders"
}
```

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Kebab-case identifier |
| `description` | string | One-line summary |
| `triggers` | string[] | Phrases that match via embedding similarity |
| `steps` | array | Sequence of actions to execute |

### Optional Fields

| Field | Type | Description |
|-------|------|-------------|
| `output` | string | Template for formatted output |
| `shadow` | boolean | Enable shadow learning (evolve from Claude's behavior) |
| `evolved_version` | int | Track skill evolution iterations |

## Step Types

### Shell Command

```json
{"run": "git status", "as": "status"}
```

Run a shell command, store output in variable.

Options:
- `if_empty`: fallback command when output is empty
- `on_fail`: recovery command on non-zero exit
- `retry`: re-run after recovery
- `timeout`: max seconds (default 30)

### LLM Prompt

```json
{"llm": "Summarize: {text}", "as": "summary", "temperature": 0.2, "max_tokens": 100}
```

Use local LLM to generate text. Prompt can reference previous `{variables}`.

Options:
- `model`: override model (default from config)
- `temperature`: sampling temperature (default 0.2)
- `max_tokens`: max output length (default 150)
- `timeout`: max seconds (default 15)
- `first_line`: take only first line of output
- `fallback`: value on error

### Control Flow

**Stop if empty:**
```json
{"stop_if_empty": "var_name", "message": "No data to process."}
```

**Labels and jumps:**
```json
{"label": "retry"}
{"goto": "retry"}
```

**Conditional jumps:**
```json
{"if_contains": "var", "value": "error", "goto": "handle_error"}
{"if_match": "var", "pattern": "regex", "goto": "label"}
{"if_empty": "var", "goto": "label"}
{"if_rc": "var", "eq": 0, "goto": "success"}
```

**Explicit stop:**
```json
{"stop": true, "message": "Done."}
```

## Built-in Variables

Every skill has access to:

| Variable | Description |
|----------|-------------|
| `{session_context}` | Current session memory |
| `{tool_examples}` | Recent Claude tool calls |
| `{tool_patterns}` | Aggregated tool usage patterns |
| `{repo_context}` | Git branch, dirty files, recent commits |

## Available Skills

| Skill | Description |
|-------|-------------|
| `git-commit` | Smart commit with LLM-generated message |
| `git-push` | Push with safety checks |
| `decision-to-teaching` | Extract patterns from decisions.md (proposed) |

## Example

```json
{
  "name": "git-commit",
  "description": "Smart git commit with LLM-generated message",
  "triggers": ["commit", "commit changes", "git commit"],
  "steps": [
    {"run": "git diff --staged", "as": "diff", "if_empty": "git add -A && git diff --staged"},
    {"stop_if_empty": "diff", "message": "Nothing to commit."},
    {"llm": "Generate commit message for:\n{diff}", "as": "msg", "first_line": true},
    {"run": "git commit -m '{msg}'", "as": "result"}
  ],
  "output": "{result}"
}
```

## Proposed Extensions

The following step types are proposed for future implementation:

### Read File
```json
{"read": "path/to/file.md", "as": "content"}
```

### Parse JSON
```json
{"parse_json": "{llm_output}", "as": "parsed"}
```

### Iterate
```json
{"iterate": "{items}", "mcp_tool": "tool_name", "args": {"key": "{item.field}"}}
```

### MCP Tool Call
```json
{"mcp_tool": "skill-hub_teach", "args": {"rule": "...", "suggest": "..."}, "as": "result"}
```

## Related

- Context-injected skills: `~/.claude/skills/` (SKILL.md format)
- Auto-skill generation: `skill-hub-cli analyze-session`
- Shadow learning: evolves skills from Claude's behavior
