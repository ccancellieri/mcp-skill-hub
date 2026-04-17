# Installation

All install flows are **idempotent** — safe to re-run. Settings are merged, not overwritten.

## TL;DR

```bash
git clone https://github.com/ccancellieri/mcp-skill-hub.git
cd mcp-skill-hub
./install.sh          # macOS / Linux
python install.py     # cross-platform
```

The installer:
1. Creates a `.venv/` and installs the package
2. Pulls `nomic-embed-text` (274 MB) via Ollama
3. Registers the MCP server in `~/.mcp.json`
4. Merges hooks into `~/.claude/settings.json`

**After install** — restart Claude Code, then:

```
index_skills()      # index all plugin skills
index_plugins()     # index plugin descriptions for suggestions
```

## Installer modes

`install.py` supports several opt-in flags:

| Flag | What it adds |
|------|--------------|
| *(none)* | Interactive — prompts for each optional component |
| `--minimal` | Core only (venv + Ollama + MCP + hooks) |
| `--full` | Everything — core + SearXNG + VPS prompt |
| `--searxng` | Core + SearXNG Docker deployment |
| `--vps http://host:11434` | Core + remote VPS Ollama for L4 offload |

On Windows, the hooks use the Python versions (`hooks/*.py`) instead of bash scripts.

## Manual install

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
ollama pull nomic-embed-text
```

Register the server in `~/.mcp.json`:

```json
{
  "mcpServers": {
    "skill-hub": {
      "type": "stdio",
      "command": "/absolute/path/to/mcp-skill-hub/.venv/bin/skill-hub"
    }
  }
}
```

---

## Ollama models — what to pull

The installer pulls `nomic-embed-text` (embedding, **required**). Pick a reasoning model by RAM:

### Reasoning model (`reason_model`)

| RAM | Model | Size | Notes |
|-----|-------|------|-------|
| 8 GB | `deepseek-r1:1.5b` | 1.1 GB | Fast, basic |
| **16 GB** | `qwen2.5-coder:7b-instruct-q4_k_m` | 4.7 GB | **Recommended** — instruct tuning → reliable JSON |
| 32 GB | `qwen2.5-coder:14b-instruct-q4_k_m` | 9 GB | Best quality / speed ratio |
| 64 GB+ | `qwen2.5-coder:32b` | 19 GB | Maximum — 4-6× slower than 14b |

```bash
ollama pull qwen2.5-coder:7b-instruct-q4_k_m
```

Then activate:

```
configure(key="reason_model", value="qwen2.5-coder:7b-instruct-q4_k_m")
```

### Why `-instruct` variants?

The hook pipeline uses two **focused** LLM calls: `eval_skill_lifecycle` (structured JSON at temp=0) and `optimize_prompt` (free-form rewrite at temp=0.2). Instruct-tuned models follow these reliably — base models frequently drop one task or wrap JSON in markdown.

On Apple Silicon, `7b-instruct-q4_k_m` benchmarks at ~3.4 s lifecycle + ~1.3 s prompt-opt — same total latency as the old single-call `3b`, with substantially better output quality.

### Better embeddings (optional)

```bash
ollama pull mxbai-embed-large
configure(key="embed_model", value="mxbai-embed-large")
index_skills()   # rebuild vectors with new model
```

### Level 4 local agent

The L4 agent model is used for the full tool-calling loop (plan → execute).

| RAM | L4 Model |
|-----|----------|
| 16 GB | `qwen2.5-coder:7b-instruct-q4_k_m` (same as reason model) |
| 32 GB | `qwen2.5-coder:14b` |
| 64 GB+ | `qwen2.5-coder:32b` |

Configure per-level:

```
configure(key="local_models",
  value='{"level_1":"qwen2.5-coder:3b","level_2":"qwen2.5-coder:7b-instruct-q4_k_m","level_3":"qwen2.5-coder:14b","level_4":"qwen2.5-coder:32b"}')
```

---

## SearXNG web search (optional)

SearXNG provides **free, private web search**. It works in two modes:

1. **Active** — `search_web("query")` or `/hub-search-web query`
2. **Passive** — the hook pipeline falls back to SearXNG when skill/task search returns no results

Results are summarized by the local LLM before being shown (active) or injected into Claude's context (passive).

```bash
python install.py --searxng      # cross-platform auto-setup
# OR manual
docker compose -f docker/docker-compose.searxng.yml up -d
```

Then configure:

```
configure(key="searxng_url", value="http://localhost:8989")
configure(key="searxng_enabled", value="true")
```

**Verify:**

```bash
curl "http://localhost:8989/search?q=test&format=json" \
  | python3 -c "import sys,json; print(len(json.load(sys.stdin)['results']), 'results')"
```

**Resource limits** (set in `docker/docker-compose.searxng.yml`):

| Resource | Limit | Reservation |
|----------|-------|-------------|
| Memory | 128 MB | 64 MB |
| CPU | 0.5 cores | 0.25 cores |

Stateless, no volumes. Engines configured in `docker/searxng-settings.yml`: Brave, Qwant, Startpage, GitHub, StackOverflow, arXiv, Semantic Scholar. CAPTCHA-heavy engines (Google, DuckDuckGo) disabled.

**Remote VPS:** point `searxng_url` at your VPS:

```
configure(key="searxng_url", value="http://your-vps-ip:8989")
```

---

## Remote VPS (Ollama)

Offload the heaviest L4 agent to a remote server. Levels 1–3 stay local for speed; L4 routes to the VPS for quality.

```bash
python install.py --vps http://myserver:11434
```

This tests connectivity, prompts for model name and optional API key, then configures:

- `remote_llm.base_url` — Ollama or OpenAI-compatible endpoint
- `remote_llm.model` — model on the remote server
- `local_models.level_4` — `remote:<url>` routes L4 traffic

**Manual:**

```
configure(key="remote_llm",
  value='{"base_url":"http://myserver:11434","model":"qwen2.5-coder:32b","timeout":120}')

configure(key="local_models",
  value='{"level_1":"qwen2.5-coder:3b","level_2":"qwen2.5-coder:7b-instruct-q4_k_m","level_3":"qwen2.5-coder:14b","level_4":"remote:http://myserver:11434"}')
```

**VPS requirements:**

- Ollama listening on `0.0.0.0:11434`
- At least one model pulled (e.g. `qwen2.5-coder:32b`)
- Inbound TCP on port `11434` (and `8989` if SearXNG runs there too)
- **32 GB+ RAM** for 32b models, 16 GB for 14b

**API key authentication** (for LiteLLM, OpenRouter, etc.):

```
configure(key="remote_llm",
  value='{"base_url":"https://api.example.com","api_key":"sk-...","model":"qwen2.5-coder:32b","timeout":120}')
```

The agent tries OpenAI-compatible `/v1/chat/completions` first, falls back to Ollama `/api/chat`.

---

## What to do next

- Run `status()` to verify everything is wired → [reference/logs.md](reference/logs.md)
- Learn how the hook saves tokens → [features/hooks.md](features/hooks.md)
- Teach the hub your vocabulary → [features/learning.md](features/learning.md)
