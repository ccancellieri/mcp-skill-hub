"""Integrity checks for the offline modular-context experiment."""
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "modular_context.py"
_SPEC = importlib.util.spec_from_file_location("modular_context", _SCRIPT)
assert _SPEC and _SPEC.loader
modular_context = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(modular_context)


@pytest.fixture(autouse=True)
def offline_tokenizer():
    try:
        modular_context.load_tokenizer()
    except RuntimeError as exc:
        pytest.skip(str(exc))


def test_modular_probe_preserves_scope_freshness_and_counts_extra_reads():
    result = modular_context.probe(1)
    assert result["prompt_preserved"]
    assert result["stale_source_rejected"]
    assert result["foreign_scope_leaks"] == 0
    schedules = result["schedules"]
    assert [row["read_count"] for row in schedules] == [0, 1, 2, 6]
    assert all(row["expected_read_markers_present"] for row in schedules)
    assert schedules[0]["request_tokens"] == schedules[0]["response_tokens"] == 0
    assert all(row["distinct_payload_tokens"] == row["index_tokens"] +
               row["request_tokens"] + row["response_tokens"] for row in schedules)
    assert schedules[-1]["distinct_payload_tokens"] > result["full_context_tokens"]


def test_modular_token_counts_and_corpus_are_reproducible():
    assert modular_context.probe(0, "it") == modular_context.probe(0, "it")
