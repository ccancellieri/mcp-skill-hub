"""Compression evaluation harness.

Runs the compression cascade over a representative corpus under three flag
configurations, reports per-payload results, and prints a verdict on the
lossy paths so a human can decide whether to enable them.

Usage
-----
    uv run python scripts/compression_eval.py

    # Skip the slow ML section (no model download):
    EVAL_SKIP_ML=1 uv run python scripts/compression_eval.py

Environment variables
---------------------
EVAL_SKIP_ML=1   Skip Kompress and code-aware evaluations (faster, offline).
"""

from __future__ import annotations

import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Make skill_hub importable when running from the repo root.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Thresholds — tune these once you have a larger sample.
# ---------------------------------------------------------------------------
FIDELITY_GOOD_THRESHOLD = 0.85   # fidelity >= this → "acceptable"
RATIO_GOOD_THRESHOLD = 0.85       # ratio < this is meaningful compression


# ---------------------------------------------------------------------------
# Corpus — representative payloads keyed by (label, payload_type).
# Each entry must be >800 chars so the compressors actually engage.
# ---------------------------------------------------------------------------

_JSON_ARRAY = """[
""" + "\n".join(
    f'  {{"id": {i}, "level": "INFO", "msg": "Worker {i} processed batch {i*10}-{i*10+9}", '
    f'"ts": {1700000000 + i * 60}, "host": "worker-{i % 4}.internal", "latency_ms": {20 + i % 50}}}'
    + ("," if i < 79 else "")
    for i in range(80)
) + "\n]"

_LOG_DUMP = "\n".join([
    f"2024-01-{(i % 28) + 1:02d} 12:{i % 60:02d}:00 INFO  [app.core] Processing request id={i:05d} user=u{i % 200}"
    for i in range(30)
] + [
    "2024-01-15 12:30:00 ERROR [app.db] Connection pool exhausted after 30 retries",
    "2024-01-15 12:30:01 ERROR [app.db]   File \"/app/db.py\", line 142, in acquire",
    "2024-01-15 12:30:01 ERROR [app.db]     raise PoolExhausted('all 10 connections in use')",
    "2024-01-15 12:30:01 ERROR [app.db] PoolExhausted: all 10 connections in use",
    "2024-01-15 12:30:02 WARN  [app.core] Falling back to read replica for request 00031",
] + [
    f"2024-01-{(i % 28) + 1:02d} 13:{i % 60:02d}:00 INFO  [app.core] Request id={i + 31:05d} completed in {10 + i % 80}ms"
    for i in range(25)
])

_SEARCH_RESULTS = "\n".join([
    f"""Result {i+1}: {['Python async patterns for scalable APIs', 'FastAPI dependency injection guide', 'SQLite performance tuning techniques', 'Pydantic v2 migration tips', 'Compression algorithms compared'][i % 5]}
URL: https://example.com/articles/{i+1}
Snippet: {'This comprehensive article covers the essential patterns and best practices for building robust, maintainable systems. The author provides concrete examples and benchmarks. ' * 3}
Published: 2024-{(i % 12) + 1:02d}-01
Relevance: {0.99 - i * 0.04:.2f}
"""
    for i in range(8)
])

_PROSE = textwrap.dedent("""\
    The transformation of software architecture over the past decade has been
    profound. Microservices promised autonomy and independent scalability, yet
    teams consistently discovered that the operational complexity they introduced
    often eclipsed the agility they offered. A service that can be deployed
    independently is valuable; a system of two hundred such services that all
    depend on a shared event bus, a shared database migration tool, and a shared
    secrets manager is, in practice, a distributed monolith with extra network
    hops added.

    The pendulum has swung. Modular monoliths, once dismissed as architectural
    cowardice, are being reconsidered by engineering teams who have spent three
    years on-call for cascading failures in their service meshes. The insight
    driving this reconsideration is simple: cohesion is a property of the code,
    not the deployment unit. A well-structured single process with clear internal
    boundaries is no less maintainable than a cluster of services — and it
    eliminates an entire class of distributed-systems failure modes: network
    partitions, serialisation mismatches, partial deployments, and the temporal
    coupling that emerges when services must be upgraded in coordinated waves.

    None of this is to say that microservices are wrong. High-throughput data
    ingestion pipelines, teams with genuine organisational boundaries, and systems
    that must scale components at wildly different rates all benefit from service
    decomposition. The key lesson is that decomposition should follow team
    topology and data-flow topology, not the intuition that smaller is always
    better. Conway's Law remains the most reliable predictor of whether a service
    boundary will stay clean over time.

    Observability has become the great equaliser. Whether a system is a monolith
    or a mesh of five hundred microservices, the ability to answer "why is this
    slow?" and "which deployment caused this error?" determines how quickly teams
    can iterate. Structured logging, distributed traces, and continuous profiling
    are no longer luxuries; they are the load-bearing infrastructure of any
    serious production system. Teams that invest early in observability consistently
    outperform those that treat it as a polish concern. The cost of adding a trace
    span after a production incident is ten times the cost of adding it before.

    The emergence of large language models as coding assistants has introduced
    a new variable into this calculus. AI-generated code is often structurally
    correct but contextually naive — it does not know your team's conventions,
    your database's idiosyncrasies, or the invisible invariants that your senior
    engineers carry in their heads. The teams getting the most value from these
    tools are treating AI output as a first draft that must be reviewed with the
    same rigour as any junior engineer's pull request: read it, understand it,
    and resist the temptation to merge it simply because it passes the linter.
    The linter does not know that your message broker retries on 5xx but not 4xx.
""")

_SOURCE_CODE = textwrap.dedent('''\
    """Session-aware rate limiter with exponential backoff and per-tenant quotas."""

    from __future__ import annotations

    import time
    import threading
    from collections import defaultdict
    from dataclasses import dataclass, field
    from typing import ClassVar

    _SENTINEL = object()


    @dataclass
    class _Bucket:
        """Token bucket for a single (tenant, endpoint) pair."""

        capacity: float
        refill_rate: float          # tokens per second
        _tokens: float = field(init=False)
        _last_refill: float = field(init=False)
        _lock: threading.Lock = field(default_factory=threading.Lock, compare=False)

        def __post_init__(self) -> None:
            self._tokens = self.capacity
            self._last_refill = time.monotonic()

        def consume(self, tokens: float = 1.0) -> bool:
            """Try to consume *tokens* from the bucket. Returns True on success."""
            with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self.capacity, self._tokens + elapsed * self.refill_rate)
                self._last_refill = now
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return True
                return False


    class RateLimiter:
        """Multi-tenant, per-endpoint token-bucket rate limiter.

        Buckets are created lazily on first request and evicted after
        *eviction_ttl* seconds of inactivity to bound memory usage.

        Example
        -------
        >>> rl = RateLimiter(default_capacity=100, default_rate=10)
        >>> ok = rl.check("tenant-a", "/api/search")
        """

        _instances: ClassVar[dict[str, "RateLimiter"]] = {}

        def __init__(
            self,
            default_capacity: float = 60.0,
            default_rate: float = 1.0,
            eviction_ttl: float = 3600.0,
            overrides: dict[str, tuple[float, float]] | None = None,
        ) -> None:
            self._default_capacity = default_capacity
            self._default_rate = default_rate
            self._eviction_ttl = eviction_ttl
            self._overrides: dict[str, tuple[float, float]] = overrides or {}
            self._buckets: dict[tuple[str, str], _Bucket] = {}
            self._last_seen: dict[tuple[str, str], float] = defaultdict(float)
            self._lock = threading.Lock()

        def _get_bucket(self, tenant: str, endpoint: str) -> _Bucket:
            key = (tenant, endpoint)
            with self._lock:
                bucket = self._buckets.get(key)
                if bucket is None:
                    cap, rate = self._overrides.get(
                        endpoint, (self._default_capacity, self._default_rate)
                    )
                    bucket = _Bucket(capacity=cap, refill_rate=rate)
                    self._buckets[key] = bucket
                self._last_seen[key] = time.monotonic()
            return bucket

        def check(self, tenant: str, endpoint: str, tokens: float = 1.0) -> bool:
            """Return True if the request is within quota, False if rate-limited."""
            return self._get_bucket(tenant, endpoint).consume(tokens)

        def evict_stale(self) -> int:
            """Remove buckets inactive for longer than *eviction_ttl*. Returns count."""
            cutoff = time.monotonic() - self._eviction_ttl
            with self._lock:
                stale = [k for k, ts in self._last_seen.items() if ts < cutoff]
                for k in stale:
                    self._buckets.pop(k, None)
                    self._last_seen.pop(k, None)
            return len(stale)
''')

CORPUS: list[dict[str, Any]] = [
    {"label": "JSON array (80 rows)", "text": _JSON_ARRAY, "kind": "json"},
    {"label": "Build/log dump",       "text": _LOG_DUMP,   "kind": "log"},
    {"label": "Search results",       "text": _SEARCH_RESULTS, "kind": "search"},
    {"label": "Verbose prose",        "text": _PROSE,      "kind": "prose"},
    {"label": "Python source code",   "text": _SOURCE_CODE, "kind": "code"},
]


# ---------------------------------------------------------------------------
# Fidelity metric
# ---------------------------------------------------------------------------

def _fidelity_embedding(original: str, compressed: str) -> tuple[float, str]:
    """Cosine similarity between sentence embeddings.  Requires sentence-transformers."""
    from sentence_transformers import SentenceTransformer  # type: ignore[import]
    import numpy as np  # type: ignore[import]

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vecs = model.encode([original, compressed], normalize_embeddings=True)
    score = float(np.dot(vecs[0], vecs[1]))
    return score, "embedding-cosine"


def _fidelity_lexical(original: str, compressed: str) -> tuple[float, str]:
    """Jaccard similarity over content-word unigrams (stopword-filtered)."""
    _STOPWORDS = frozenset({
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "shall", "this", "that", "these",
        "those", "it", "its", "as", "not", "no", "so", "if", "then", "than",
    })

    def _words(text: str) -> frozenset[str]:
        import re
        tokens = re.findall(r"[a-zA-Z0-9_]{3,}", text.lower())
        return frozenset(t for t in tokens if t not in _STOPWORDS)

    orig_words = _words(original)
    comp_words = _words(compressed)
    if not orig_words:
        return 1.0, "lexical-jaccard"
    intersection = len(orig_words & comp_words)
    union = len(orig_words | comp_words)
    return (intersection / union) if union else 0.0, "lexical-jaccard"


def compute_fidelity(original: str, compressed: str) -> tuple[float, str]:
    """Return (score, metric_name).  Prefers embedding cosine, falls back to Jaccard."""
    try:
        return _fidelity_embedding(original, compressed)
    except Exception:
        return _fidelity_lexical(original, compressed)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _set_lossy_flags(ml: bool, code: bool) -> None:
    from skill_hub import config as cfg
    cfg.set("compression_ml_enabled", ml)
    cfg.set("compression_code_aware_enabled", code)


def _restore_flags() -> None:
    _set_lossy_flags(False, False)


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

@dataclass
class _RunResult:
    label: str
    kind: str
    config_name: str
    strategy: str
    bytes_before: int
    bytes_after: int
    ratio: float
    lossy: bool
    fidelity: float | None
    fidelity_metric: str | None


def _run_one(payload: dict[str, Any], config_name: str) -> _RunResult:
    from skill_hub.compression import compress_payload

    text = payload["text"]
    result = compress_payload(text, allow_lossy=True)

    fidelity: float | None = None
    fidelity_metric: str | None = None
    if result.lossy and result.bytes_after < result.bytes_before:
        fidelity, fidelity_metric = compute_fidelity(text, result.compressed)

    return _RunResult(
        label=payload["label"],
        kind=payload["kind"],
        config_name=config_name,
        strategy=result.content_type,
        bytes_before=result.bytes_before,
        bytes_after=result.bytes_after,
        ratio=result.ratio,
        lossy=result.lossy,
        fidelity=fidelity,
        fidelity_metric=fidelity_metric,
    )


def _run_config(
    config_name: str,
    ml: bool,
    code: bool,
    corpus: list[dict[str, Any]],
) -> list[_RunResult]:
    _set_lossy_flags(ml, code)
    try:
        return [_run_one(p, config_name) for p in corpus]
    finally:
        _restore_flags()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_COL_W = 22  # label column width


def _fmt_row(r: _RunResult) -> str:
    fid = f"{r.fidelity:.3f} ({r.fidelity_metric})" if r.fidelity is not None else "-"
    lossy_flag = "LOSSY" if r.lossy else "lossless"
    return (
        f"  {r.label:<{_COL_W}}  {r.strategy:<16}"
        f"  {r.bytes_before:>7} → {r.bytes_after:>7}"
        f"  ratio={r.ratio:.3f}  {lossy_flag:<9}  fidelity={fid}"
    )


def _print_config_block(results: list[_RunResult]) -> None:
    config_name = results[0].config_name if results else "?"
    print(f"\n{'─' * 90}")
    print(f"  Config: {config_name}")
    print(f"{'─' * 90}")
    print(
        f"  {'Payload':<{_COL_W}}  {'Strategy':<16}"
        f"  {'Bytes before→after':>21}  {'Ratio':<9}  {'Lossy':<9}  Fidelity"
    )
    print(f"  {'-' * (_COL_W)}  {'-' * 16}  {'-' * 21}  {'-' * 8}  {'-' * 9}  {'-' * 30}")
    for r in results:
        print(_fmt_row(r))


def _summary(all_results: list[list[_RunResult]]) -> None:
    print(f"\n{'═' * 90}")
    print("  SUMMARY — lossy configurations")
    print(f"{'═' * 90}")

    for results in all_results:
        config_name = results[0].config_name if results else "?"
        lossy_results = [r for r in results if r.lossy and r.fidelity is not None]
        if not lossy_results:
            print(f"  {config_name}: no lossy compressions triggered")
            continue
        avg_ratio = sum(r.ratio for r in lossy_results) / len(lossy_results)
        avg_fidelity = sum(r.fidelity for r in lossy_results) / len(lossy_results)  # type: ignore[arg-type]
        metric = lossy_results[0].fidelity_metric or "?"
        verdict = "acceptable" if avg_fidelity >= FIDELITY_GOOD_THRESHOLD else "REVIEW — fidelity below threshold"
        print(
            f"  {config_name}: avg ratio={avg_ratio:.3f}  avg fidelity={avg_fidelity:.3f} "
            f"({metric}) — {verdict}"
        )
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    skip_ml = os.environ.get("EVAL_SKIP_ML", "").strip() in {"1", "true", "yes"}

    # Sanity-check payload sizes.
    for entry in CORPUS:
        n = len(entry["text"])
        if n < 800:
            print(f"WARNING: corpus entry '{entry['label']}' is only {n} chars — may not trigger compression")

    configs = [
        ("(a) lossy OFF",      False, False),
        ("(b) ML (Kompress) ON only", True, False),
        ("(c) code-aware ON only",    False, True),
    ]

    if skip_ml:
        print("EVAL_SKIP_ML is set — running deterministic-only configuration.\n")
        configs = [("(a) lossy OFF", False, False)]

    all_block_results: list[list[_RunResult]] = []
    for config_name, ml, code in configs:
        if (ml or code) and skip_ml:
            continue
        print(f"\nRunning configuration: {config_name} …")
        results = _run_config(config_name, ml, code, CORPUS)
        all_block_results.append(results)
        _print_config_block(results)

    _summary(all_block_results)
    return 0


if __name__ == "__main__":
    sys.exit(main())
