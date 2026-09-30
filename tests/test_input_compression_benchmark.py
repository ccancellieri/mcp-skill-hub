"""Checks that the offline benchmark reports real size and fidelity observations."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest


_SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "input_compression.py"
_SPEC = importlib.util.spec_from_file_location("input_compression", _SCRIPT)
assert _SPEC and _SPEC.loader
input_compression = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(input_compression)


@pytest.fixture
def offline_tokenizer():
    try:
        return input_compression.load_tokenizer()
    except RuntimeError as exc:
        pytest.skip(f"verified offline o200k_base tokenizer unavailable: {exc}")


def test_measurements_match_tokenizer_and_utf8_lengths(offline_tokenizer):
    result = input_compression.run(repeats=2)
    rows = result["cases"]
    encoding = offline_tokenizer
    sources = {case["name"]: case["text"] for case in input_compression.corpus()}

    assert len(rows) == 2 * len(sources)
    for row in rows:
        source = sources[row["name"]]
        assert row["bytes_before"] == len(source.encode("utf-8"))
        assert row["tokens_before"] == len(encoding.encode(source))
        assert row["original_sha256"] == hashlib.sha256(source.encode("utf-8")).hexdigest()
        assert row["tokens_saved"] == row["tokens_before"] - row["tokens_after"]
        assert row["latency_ns_median"] >= 0
        assert len(row["latency_ns_samples"]) == 2
        if row["transform"] == "PASSTHROUGH":
            assert row["byte_exact"]
            assert row["tokens_before"] == row["tokens_after"]
            assert row["fidelity_pass"] is True
        elif not row["lossy"]:
            assert row["fidelity_pass"] is True

    for mode in ("production_default_safe", "explicit_lossy_opt_in"):
        scoped = [row for row in rows if row["mode"] == mode]
        totals = [group for group in result["by_content_type"] if group["mode"] == mode]
        assert sum(group["tokens_before"] for group in totals) == sum(row["tokens_before"] for row in scoped)
        assert sum(group["tokens_after"] for group in totals) == sum(row["tokens_after"] for row in scoped)


def test_modes_keep_unsupported_inputs_and_label_lossy_log_collapse(offline_tokenizer):
    rows = input_compression.run(repeats=1)["cases"]
    by_key = {(row["mode"], row["name"]): row for row in rows}

    for name in ("english_prose", "italian_prose", "python_code", "already_compact_json", "below_threshold_json"):
        for mode in ("production_default_safe", "explicit_lossy_opt_in"):
            assert by_key[(mode, name)]["transform"] == "PASSTHROUGH"

    safe = by_key[("production_default_safe", "repeated_logs")]
    opt_in = by_key[("explicit_lossy_opt_in", "repeated_logs")]
    assert safe["transform"] == "PASSTHROUGH" and safe["lossy"] is False
    assert opt_in["transform"] == "DEDUP" and opt_in["lossy"] is True
    assert opt_in["fidelity_pass"] is None
    assert opt_in["tokens_after"] < opt_in["tokens_before"]

    for name in ("long_pretty_json", "nested_duplicate_lexemes"):
        row = by_key[("production_default_safe", name)]
        assert row["transform"] == "JSON_MIN"
        assert row["lossy"] is False and row["fidelity_pass"] is True
