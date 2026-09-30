"""Offline, deterministic measurements of the production input compressor.

Run with a verified local tokenizer cache, for example::

    TIKTOKEN_CACHE_DIR=/private/tmp/skill-hub-tokenizer-cache \
      python benchmarks/input_compression.py --output /private/tmp/skill-hub-input-compression/results.json

Only ``compress_payload`` is timed. Tokenization, corpus construction, and
integrity checks happen outside the timed interval. No model or network calls
are made by the benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import statistics
import sys
from time import perf_counter_ns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmarks.context_compare import load_tokenizer
from skill_hub.compression import _CHARS_PER_TOKEN, _DEFAULT_MIN_TOKENS, compress_payload


_STRING_TOKEN = re.compile(r'"(?:\\.|[^"\\])*"')


def corpus() -> list[dict[str, str]]:
    """Fixed synthetic inputs; no user data or runtime state."""
    rows = [
        {"id": index, "region": "Emilia-Romagna", "measure": index * 0.125,
         "path": f"/data/tiles/{index:03d}/scene.json", "active": index % 3 != 0}
        for index in range(100)
    ]
    nested = (' { "batch": [\n' + ',\n'.join(
        ' { "value": 1.2300, "value": 1e400, "label": "caf\\u00e9", '
        '"nested": { "id": %d, "ok": true } }' % index for index in range(35)
    ) + '\n], "note": "retain duplicate keys and numeric spelling" } ')
    english = "\n".join(
        f"Step {index}: Do not delete /srv/cache/{index:03d}; keep 1.2300 and "
        f"retry at most {index % 4 + 1} times. The fallback is not approved."
        for index in range(45)
    )
    italian = "\n".join(
        f"Passo {index}: non eliminare /dati/archivio/{index:03d}; conserva "
        f"1.2300 e attendi {index % 5 + 1} secondi. Non usare il percorso alternativo."
        for index in range(45)
    )
    code = "def render(items):\n" + "\n".join(
        f"    value_{index} = items[{index}]  # preserve whitespace and index {index}"
        for index in range(80)
    ) + "\n    return items\n"
    logs = "\n".join(["INFO worker started"] * 80 + ["ERROR retry pending"] * 20)
    return [
        {"name": "long_pretty_json", "category": "json", "text": json.dumps(rows, indent=2, ensure_ascii=False)},
        {"name": "nested_duplicate_lexemes", "category": "json", "text": nested},
        {"name": "english_prose", "category": "prose", "text": english},
        {"name": "italian_prose", "category": "prose", "text": italian},
        {"name": "python_code", "category": "code", "text": code},
        {"name": "repeated_logs", "category": "logs", "text": logs},
        {"name": "already_compact_json", "category": "json", "text": json.dumps(rows, separators=(",", ":"), ensure_ascii=False)},
        {"name": "below_threshold_json", "category": "json", "text": '{ "id": 7, "ok": true }'},
    ]


def _json_signature(text: str) -> object:
    return json.loads(
        text,
        object_pairs_hook=lambda pairs: pairs,
        parse_int=lambda value: ("integer", value),
        parse_float=lambda value: ("number", value),
    )


def _fidelity(original: str, output: str, transform: str, lossy: bool) -> tuple[str, bool | None]:
    if lossy:
        return "lossy_output_not_reversible", None
    if transform == "PASSTHROUGH":
        return "byte_exact", original == output
    if transform == "JSON_MIN":
        try:
            same = (_json_signature(original) == _json_signature(output)
                    and _STRING_TOKEN.findall(original) == _STRING_TOKEN.findall(output))
        except (ValueError, RecursionError):
            same = False
        return "json_pairs_number_spellings_and_string_lexemes", same
    return "unsupported_transform", False


def measure_case(case: dict[str, str], encoding, *, allow_lossy: bool, repeats: int = 7) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be positive")
    source = case["text"]
    # Warm the Python path before timing; the production default threshold is
    # left untouched, including its approximate character-based gate.
    compress_payload(source, allow_lossy=allow_lossy)
    samples = []
    for _ in range(repeats):
        start = perf_counter_ns()
        payload = compress_payload(source, allow_lossy=allow_lossy)
        samples.append(perf_counter_ns() - start)
    output = payload.compressed
    before_bytes = len(source.encode("utf-8"))
    after_bytes = len(output.encode("utf-8"))
    if payload.bytes_before != before_bytes or payload.bytes_after != after_bytes:
        raise AssertionError("production byte counts disagree with UTF-8 lengths")
    check, fidelity = _fidelity(source, output, payload.content_type, payload.lossy)
    if fidelity is False:
        raise AssertionError(f"fidelity failed for {case['name']}: {check}")
    before_tokens = len(encoding.encode(source))
    after_tokens = len(encoding.encode(output))
    return {
        "name": case["name"], "category": case["category"],
        "mode": "explicit_lossy_opt_in" if allow_lossy else "production_default_safe",
        "transform": payload.content_type, "lossy": payload.lossy,
        "original_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        "bytes_before": before_bytes, "bytes_after": after_bytes,
        "tokens_before": before_tokens, "tokens_after": after_tokens,
        "tokens_saved": before_tokens - after_tokens,
        "byte_exact": source == output, "fidelity_check": check,
        "fidelity_pass": fidelity,
        "latency_ns_median": int(statistics.median(samples)),
        "latency_ns_samples": samples,
    }


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        groups.setdefault((row["mode"], row["category"]), []).append(row)
    return [
        {"mode": mode, "category": category, "cases": len(group),
         "tokens_before": sum(row["tokens_before"] for row in group),
         "tokens_after": sum(row["tokens_after"] for row in group),
         "tokens_saved": sum(row["tokens_saved"] for row in group),
         "bytes_before": sum(row["bytes_before"] for row in group),
         "bytes_after": sum(row["bytes_after"] for row in group),
         "transforms": {name: sum(row["transform"] == name for row in group)
                        for name in sorted({row["transform"] for row in group})}}
        for (mode, category), group in sorted(groups.items())
    ]


def run(*, repeats: int = 7) -> dict:
    encoding = load_tokenizer()  # Verifies the offline cache and SHA-256.
    cases = corpus()
    rows = [measure_case(case, encoding, allow_lossy=lossy, repeats=repeats)
            for lossy in (False, True) for case in cases]
    return {
        "method": "production compress_payload; tokenizer and verification outside timed interval",
        "tokenizer": {"encoding": "o200k_base", "tiktoken_version": importlib.metadata.version("tiktoken")},
        "configuration": {
            "api": "compress_payload", "min_tokens_argument": None,
            "production_default_min_tokens": _DEFAULT_MIN_TOKENS,
            "production_chars_per_token_gate": _CHARS_PER_TOKEN,
            "config_file_read": False, "network_or_model_calls": False,
            "warmups_per_case": 1, "timed_repeats_per_case": repeats,
        },
        "cases": rows,
        "by_content_type": summarize(rows),
        "limits": [
            "Synthetic fixtures measure output size and fidelity properties, not downstream task quality.",
            "The threshold uses character count; o200k tokens here measure results, not eligibility.",
            "Lossy duplicate-line collapse is opt-in and cannot be verified as reversible.",
            "Latency is local process timing and excludes tokenization, startup, and integration overhead.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    result = run(repeats=args.repeats)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
