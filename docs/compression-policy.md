# Deterministic vs. LLM compression — policy

Skill Hub reduces context two ways. This is the rule for which to use.

## Use deterministic compression (the `compression` extra, `headroom-ai`)

For **structured payloads** where structure carries the signal:

- command / shell output, build & test logs (keep errors, tracebacks, summary)
- search / grep output (keep first+last + top-scored matches per file)
- JSON arrays of records (SmartCrusher: anchors + error rows + outliers + dedup)
- git diffs, CSV/TSV/markdown tables

These run through `skill_hub.compression.compress_payload` / `maybe_compress`. They are
**free, microsecond-fast, deterministic, and offline** — no model, no tokens, no network.
The ML "Kompress" path is deliberately disabled, so the install stays light and prose/code
are never AST-mangled. Large lossy reductions leave a reversible `<<ccr:HASH>>` marker; call
the `retrieve_compressed` tool to rehydrate the original.

## Keep the LLM (Ollama / Anthropic)

For **prose synthesis** where the value is in rewriting, not selecting:

- `compact_master_state` — architectural synthesis for cold-start context
- `optimize_prompt` / query rewriting — paraphrase
- narrative task summaries and conversation digests where coherence matters

Prose and source code route to the disabled Kompress strategy and **pass through unchanged**,
so feeding prose to `compress_payload` is a safe no-op — it simply returns the original and the
LLM still does the synthesis.

## The rule of thumb

> Deterministic when the failure mode is "slightly worse selection". LLM when the failure mode
> is "wrong synthesis / lost meaning".

This mirrors the existing principle in `docs/master-state-compaction.md`: tactical helpers stay
cheap; high-stakes cold-start synthesis pays for the smart model.

## Wiring points (this layer)

- `cli.py` Level-2 shell executor — compress command output before the 5000-char backstop.
- `searxng.py` `_summarize_results` — compress concatenated web results before the local LLM.
- Prose skill-body truncations (`server.py`, `cli.py` per-skill clips) are intentionally **left
  as-is** — they are markdown prose and correctly pass through.

## Config

| Key | Default | Meaning |
|---|---|---|
| `compression_enabled` | `True` | Master switch. Auto-no-ops if `headroom-ai` is absent. |
| `compression_min_tokens` | `200` | Skip payloads below ~this token count. |
| `compression_context_aware` | `True` | Pass the user query as relevance context. |

Install: `pip install "mcp-skill-hub[compression]"`.
