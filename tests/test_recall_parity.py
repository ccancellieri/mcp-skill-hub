"""S1.7 — recall parity: sqlite-vec (binary + float32 rerank) vs legacy float32.

Skipped when the live skill corpus is not populated (fresh install).
"""
from __future__ import annotations

import pytest

from skill_hub.embeddings import embed
from skill_hub.store import SkillStore


QUERIES = [
    "git commit", "docker build", "pdf extraction", "react hooks",
    "pandas dataframe", "sql optimization", "testing in python",
    "webhook handling", "parse csv", "async event loop",
    "typescript interface", "python dataclass", "kubernetes deploy",
    "aws lambda", "graphql schema",
]

MIN_RECALL_AT_5 = 0.95


@pytest.fixture(scope="module")
def store() -> SkillStore:
    s = SkillStore()
    if s._vec_engine != "sqlite-vec":
        pytest.skip("sqlite-vec extension unavailable")
    if s._conn.execute(
        "SELECT count(*) FROM skills_vec_bin"
    ).fetchone()[0] < 100:
        pytest.skip("skill corpus too small to measure recall parity")
    return s


def test_recall_at_5_parity(store: SkillStore) -> None:
    try:
        vectors = [embed(q) for q in QUERIES]
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"ollama not reachable: {exc}")

    matches = total = 0
    for v in vectors:
        store._vec_engine = "legacy"
        legacy = [h["id"] for h in store.search(v, top_k=5)]
        store._vec_engine = "sqlite-vec"
        vec = [h["id"] for h in store.search(v, top_k=5)]
        matches += len(set(legacy) & set(vec))
        total += len(legacy)

    assert total > 0
    recall = matches / total
    assert recall >= MIN_RECALL_AT_5, (
        f"recall@5 {recall:.2%} < target {MIN_RECALL_AT_5:.0%}"
    )
