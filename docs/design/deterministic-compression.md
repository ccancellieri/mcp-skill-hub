# Deterministic Compression Layer — Design Spec

**Date:** 2026-06-22
**Status:** Approved — depend on `headroom-ai` (do not port)
**Goal:** Reduce structured payloads (JSON, logs, search results, diffs) *before* they reach an
LLM or get injected into Claude's context — a cheap, fast, offline-safe alternative to the
local-Ollama "compaction tax", and a structure-aware replacement for blunt char-truncation.

## 0. Decision log

- Initial plan was to *port* Headroom's pure-Python compressors. **Superseded** by user direction
  (2026-06-22): "use them as a library… no need to translate to Python."
- We therefore **depend on `headroom-ai`** (Apache-2.0, on PyPI, v0.27.0, requires-python ≥3.10).
- Verified by smoke test: `pip install headroom-ai` pulls a **prebuilt maturin/PyO3 wheel** (Rust core
  inside) plus only `tiktoken, pydantic, litellm, click, rich, opentelemetry-api, ast-grep-cli`.
  **No torch, no tree-sitter, no transformers.** `litellm` is already a Skill Hub dependency.
- Smoke result: `ContentRouter(enable_kompress=False).compress(<json array>)` → SmartCrusher,
  **ratio 0.203 (80% smaller), error rows preserved, no torch**.

## 1. Why this is the right leverage

Skill Hub pays an Ollama cost to compact/summarize and elsewhere uses blunt char-truncation
(`content[:N] + "<!-- truncated -->"`) that discards the *tail* of logs/search output — exactly
where errors and summaries live. Headroom's deterministic compressors (SmartCrusher for JSON,
Log/Search/Diff/Tabular/HTML compressors) fix both: free, microsecond-fast, deterministic, and
structure-aware (keep errors, signatures, first/last, outliers; drop filler). CCR keeps the
original retrievable by hash.

The LLM is **not** removed from prose-synthesis paths (`compact_master_state`, `optimize_prompt`,
narrative digests). Prose/code route to the (disabled) Kompress path and pass through untouched —
so prose stays for the LLM. Swapping the LLM on hot routing paths is a separate follow-up.

## 2. Scope (this PR)

In:
1. **Optional dependency:** `compression = ["headroom-ai>=0.27,<1.0"]` extra in `pyproject.toml`.
2. **Adapter** `src/skill_hub/compression/`:
   - `is_available() -> bool` — import probe.
   - `compress_payload(content, *, context="", min_tokens=...) -> CompressedPayload` — builds a
     deterministic `ContentRouterConfig(enable_kompress=False)`, calls a cached `ContentRouter`,
     returns `{compressed, content_type, ratio, bytes_before, bytes_after, lossy, ccr_keys}`.
     **Never raises**; on unavailable/error/below-threshold/ratio≥1.0 → returns original (passthrough).
   - `retrieve_original(hash) -> str | None` — delegates to headroom's `CompressionStore`.
3. **`retrieve_compressed(hash)` MCP tool** in `server.py` so Claude (talking to skill-hub) can pull
   originals behind `<<ccr:HASH>>` markers on demand.
4. **Wiring** as a pre-LLM / pre-injection stage, behind `compression_enabled`, fallback-safe:
   - context-bridge captured tool outputs (logs/search/JSON),
   - searxng web results (before the existing summarizer),
   - blunt-truncation sites (per-skill content clips) for structured content.
5. **Config flags** + a written deterministic-vs-LLM **policy doc**.

Out (→ follow-up issues): aggressive deterministic-first swap of the LLM on hot routing paths;
the `[ml]` Kompress prose compressor; the `[code]` tree-sitter path (we leave code as passthrough);
metering local Ollama cost.

## 3. Adapter contract

```python
@dataclass
class CompressedPayload:
    compressed: str
    content_type: str          # e.g. "SMART_CRUSHER", "LOG", "PASSTHROUGH"
    ratio: float               # compressed/original bytes (1.0 = unchanged)
    bytes_before: int
    bytes_after: int
    lossy: bool
    ccr_keys: list[str]        # hashes of any stashed originals

def is_available() -> bool: ...
def compress_payload(content, *, context="", min_tokens=None) -> CompressedPayload: ...
def retrieve_original(hash: str) -> str | None: ...
```

Implementation notes:
- A module-level cached `ContentRouter` built from a `ContentRouterConfig` with `enable_kompress=False`
  (so no ML attempt), `protect_error_outputs=True`, `ccr_enabled=True`.
- Below `compression_min_tokens` (chars-based proxy) or when headroom is missing → passthrough.
- Treat `strategy_used in {KOMPRESS, PASSTHROUGH}` or `ratio >= 1.0` as "not worth it" → return original.
- All headroom imports are lazy + `try/except ImportError`. Any exception during compress → passthrough
  (log at debug). Skill Hub must run identically when `headroom-ai` is not installed.

## 4. CCR

Headroom maintains a process-singleton `CompressionStore` (`headroom.cache.compression_store.
get_compression_store()`), TTL'd, with `store()`/`retrieve()`. Lossy compressors inject
`<<ccr:HASH ...>>` markers. The `retrieve_compressed` MCP tool calls `retrieve(hash)`.
We do **not** build our own store; we surface retrieval through skill-hub's MCP surface.

## 5. Config (config.py)

```
compression_enabled: bool = True       # master switch; if headroom missing, auto-no-op
compression_min_tokens: int = 200      # skip small payloads (approx via chars/4)
compression_context_aware: bool = True # pass the user query as `context` for relevance-aware crush
```

## 6. Testing

- Adapter tests guarded by `pytest.importorskip("headroom")`: JSON array → SmartCrusher, ratio < 1.0,
  error row preserved; plain prose → passthrough (ratio 1.0); malformed input → returns original, no raise;
  CCR round-trip via `retrieve_original`.
- Wiring tests monkeypatch `compress_payload` so they run without headroom installed and assert the
  flag gates behavior and failures fall back to current output.
- CI: add an env/extra that installs `.[compression]` for the adapter tests; base test run skips them.

## 7. Risks & mitigations

- **External dependency / API drift:** pin `>=0.27,<1.0`; adapter is the single seam; import-guarded so a
  missing/broken headroom degrades to today's behavior.
- **Lossy without retrieval in hook context:** same risk as today's truncation but smarter; CCR +
  `retrieve_compressed` give a recovery path where a tool call is available.
- **Wheel availability on a platform:** optional extra; base install unaffected if no wheel.

## 8. Follow-up issues (gaps)

1. Aggressive cost-cut: deterministic-first swap of local LLM on hot routing paths (triage, precompact,
   conversation_digest) with LLM fallback on poor ratio + an eval harness.
2. Opt-in `[ml]` Kompress and `[code]` tree-sitter paths for prose/code, measured against quality.
3. Active CacheAligner (stabilize KV-cache prefixes).
4. Meter local Ollama cost so deterministic-vs-LLM savings are measurable.
