"""S1.7 — search latency benchmark: sqlite-vec vs legacy cosine.

Run explicitly::

    uv run pytest tests/bench_search.py -v -s

Skipped when the live skill corpus is not populated or ollama is unreachable.
Prints p50/p95 and speedup. Not a hard-assert — for reporting only.
"""
from __future__ import annotations

import statistics
import time

import pytest

from skill_hub.embeddings import embed
from skill_hub.store import SkillStore


QUERIES = [
    "git commit", "docker build", "pdf extraction", "react hooks",
    "pandas dataframe", "sql optimization", "testing in python",
    "webhook handling", "parse csv", "async event loop",
    "typescript interface", "python dataclass", "kubernetes deploy",
    "aws lambda", "graphql schema", "postgres indexing",
    "fastapi routing", "llm prompt engineering", "vector embedding",
    "ci/cd pipeline",
]


def _percentile(data: list[float], pct: float) -> float:
    return statistics.quantiles(data, n=100)[int(pct) - 1] if len(data) >= 2 else data[0]


def test_bench_search(capsys) -> None:
    s = SkillStore()
    if s._vec_engine != "sqlite-vec":
        pytest.skip("sqlite-vec extension unavailable")
    if s._conn.execute(
        "SELECT count(*) FROM skills_vec_bin"
    ).fetchone()[0] < 100:
        pytest.skip("skill corpus too small to benchmark")
    try:
        vectors = [embed(q) for q in QUERIES]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ollama not reachable: {exc}")

    # Warm-up
    s.search(vectors[0], top_k=5)

    legacy_times: list[float] = []
    s._vec_engine = "legacy"
    for v in vectors:
        t = time.perf_counter()
        s.search(v, top_k=5)
        legacy_times.append((time.perf_counter() - t) * 1000)

    vec_times: list[float] = []
    s._vec_engine = "sqlite-vec"
    for v in vectors:
        t = time.perf_counter()
        s.search(v, top_k=5)
        vec_times.append((time.perf_counter() - t) * 1000)

    n_skills = s._conn.execute("SELECT count(*) FROM skills_vec_bin").fetchone()[0]
    report = (
        f"\nSearch benchmark on {n_skills} skills ({len(QUERIES)} queries):\n"
        f"  legacy     p50={statistics.median(legacy_times):.1f}ms  "
        f"p95={_percentile(legacy_times, 95):.1f}ms\n"
        f"  sqlite-vec p50={statistics.median(vec_times):.1f}ms  "
        f"p95={_percentile(vec_times, 95):.1f}ms\n"
        f"  speedup    p50={statistics.median(legacy_times) / max(statistics.median(vec_times), 0.01):.1f}x"
    )
    with capsys.disabled():
        print(report)
