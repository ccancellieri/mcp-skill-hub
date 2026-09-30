"""Fidelity checks for deterministic compression."""

from skill_hub.compression import compress_payload, maybe_compress
from skill_hub.context_composer import compact_structured


def test_json_minification_preserves_original_lexemes_and_utf8_byte_counts():
    source = ' { "x": 1.2300, "x": 1e400, "s": "café \\u0061  z" } '
    result = compress_payload(source, min_tokens=0, allow_lossy=False)

    assert result.content_type == "JSON_MIN"
    assert result.compressed == '{"x":1.2300,"x":1e400,"s":"café \\u0061  z"}'
    assert result.lossy is False
    assert result.bytes_before == len(source.encode("utf-8"))
    assert result.bytes_after == len(result.compressed.encode("utf-8"))
    assert result.saved_bytes == result.bytes_before - result.bytes_after
    assert result.ratio == result.bytes_after / result.bytes_before
    assert compact_structured(source) == result.compressed


def test_invalid_json_and_code_pass_through_even_if_lines_repeat():
    for source in (
        '{"x": NaN, "x": NaN, "x": NaN}',
        '{"x": Infinity, "x": Infinity, "x": Infinity}',
        '{"x": 1}\n{"x": 1}\n{"x": 1}',
        '```python\npass\npass\npass\n```',
        'def work():\n    pass\n    pass\n    pass',
        'INFO = 1\nINFO = 1\nINFO = 1',
        'run task()\nrun task()\nrun task()',
        'INFO literal … (x7)\nINFO literal … (x7)\nINFO literal … (x7)',
    ):
        result = compress_payload(source, min_tokens=0, allow_lossy=True)
        assert result.content_type == "PASSTHROUGH"
        assert result.compressed == source
        assert compact_structured(source) == source


def test_repeated_line_collapse_requires_lossy_opt_in_and_reports_loss():
    source = "\n".join(["INFO connecting"] * 50)
    safe = compress_payload(source, min_tokens=0, allow_lossy=False)
    assert safe.content_type == "PASSTHROUGH"
    assert safe.compressed == source
    assert safe.lossy is False

    lossy = compress_payload(source, min_tokens=0, allow_lossy=True)
    assert lossy.content_type == "DEDUP"
    assert lossy.lossy is True
    assert lossy.changed
    assert "(x50)" in lossy.compressed


def test_maybe_compress_preserves_repeated_logs_by_default(monkeypatch):
    from skill_hub import config

    source = "\n".join(["INFO connecting"] * 200)
    monkeypatch.setattr(config, "get", lambda key: key == "compression_enabled")
    assert maybe_compress(source) == source
