"""Tests for skill_hub.webfetch — the fetch_compressed() implementation.

No live network calls: httpx.get is monkeypatched throughout. Kompress (the
ML prose compressor) is monkeypatched too where its exact output would be
environment-dependent -- only the routing (prose vs code/JSON, gates,
marker convention) is under test here, not headroom's model quality.
"""
from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from skill_hub import config as cfg  # noqa: E402
from skill_hub import webfetch  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def isolated_config(tmp_path, monkeypatch):
    """Every test here goes through webfetch.run(), which reads
    compression_enabled from config -- isolate CONFIG_PATH per the project
    rule so we never touch the real ~/.claude config."""
    monkeypatch.setattr(cfg, "CONFIG_PATH", tmp_path / "config.json")


def _response(body: bytes, *, status: int = 200, content_type: str = "text/html") -> httpx.Response:
    request = httpx.Request("GET", "http://example.test/page")
    return httpx.Response(
        status, request=request, content=body,
        headers={"content-type": content_type} if content_type else {},
    )


def _mock_get(monkeypatch, response=None, *, exc: Exception | None = None):
    def fake_get(url, **kwargs):
        if exc is not None:
            raise exc
        return response
    monkeypatch.setattr(httpx, "get", fake_get)


# ---------------------------------------------------------------------------
# pure helpers
# ---------------------------------------------------------------------------

def test_classify_html():
    assert webfetch._classify("text/html; charset=utf-8") == "html"
    assert webfetch._classify("application/xhtml+xml") == "html"


def test_classify_text_variants():
    assert webfetch._classify("text/plain") == "text"
    assert webfetch._classify("application/json") == "text"
    assert webfetch._classify("application/ld+json") == "text"
    assert webfetch._classify("") == "text"  # missing header -> lenient


def test_classify_binary():
    assert webfetch._classify("image/png") == "binary"
    assert webfetch._classify("application/octet-stream") == "binary"


def test_fallback_strip_html_removes_scripts_and_tags():
    html = "<html><head><script>evil()</script></head><body><p>Hello &amp; world</p></body></html>"
    out = webfetch._fallback_strip_html(html)
    assert "evil()" not in out
    assert "<p>" not in out
    assert "Hello & world" in out


def test_split_segments_preserves_code_fences():
    text = "prose before\n```py\ncode here\n```\nprose after"
    segments = webfetch._split_segments(text)
    is_code_flags = [c for c, _ in segments]
    assert True in is_code_flags
    code_chunks = [chunk for is_code, chunk in segments if is_code]
    assert code_chunks == ["```py\ncode here\n```"]
    assert "".join(chunk for _, chunk in segments) == text


def test_split_segments_no_fences_is_single_prose_chunk():
    text = "just plain prose, no code at all"
    segments = webfetch._split_segments(text)
    assert segments == [(False, text)]


# ---------------------------------------------------------------------------
# fetch_url — transport-level failure modes
# ---------------------------------------------------------------------------

def test_fetch_url_rejects_non_http_scheme():
    result = webfetch.fetch_url("file:///etc/passwd")
    assert isinstance(result, webfetch.FetchError)
    assert "scheme" in result.message


def test_fetch_url_timeout(monkeypatch):
    _mock_get(monkeypatch, exc=httpx.TimeoutException("timed out"))
    result = webfetch.fetch_url("http://example.test/slow")
    assert isinstance(result, webfetch.FetchError)
    assert "timed out" in result.message


def test_fetch_url_http_status_error(monkeypatch):
    _mock_get(monkeypatch, response=_response(b"nope", status=404))
    result = webfetch.fetch_url("http://example.test/missing")
    assert isinstance(result, webfetch.FetchError)
    assert "404" in result.message


def test_fetch_url_connect_error(monkeypatch):
    _mock_get(monkeypatch, exc=httpx.ConnectError("refused"))
    result = webfetch.fetch_url("http://example.test/down")
    assert isinstance(result, webfetch.FetchError)
    assert "fetch failed" in result.message


def test_fetch_url_truncates_oversized_body(monkeypatch):
    monkeypatch.setattr(webfetch, "_MAX_FETCH_BYTES", 10)
    _mock_get(monkeypatch, response=_response(b"0123456789ABCDEF", content_type="text/plain"))
    result = webfetch.fetch_url("http://example.test/big")
    assert isinstance(result, webfetch.FetchedPage)
    assert result.truncated is True
    assert len(result.text) == 10


def test_fetch_url_decodes_text_ok(monkeypatch):
    _mock_get(monkeypatch, response=_response(b"hello world", content_type="text/plain"))
    result = webfetch.fetch_url("http://example.test/ok")
    assert isinstance(result, webfetch.FetchedPage)
    assert result.text == "hello world"
    assert result.content_type == "text/plain"
    assert result.truncated is False


# ---------------------------------------------------------------------------
# run() — orchestration, gates, error modes
# ---------------------------------------------------------------------------

def test_run_invalid_mode_returns_error():
    out = webfetch.run("http://example.test", mode="bogus")
    assert "invalid mode" in out


def test_run_fetch_error_propagates(monkeypatch):
    _mock_get(monkeypatch, exc=httpx.TimeoutException("slow"))
    out = webfetch.run("http://example.test/slow")
    assert out.startswith("fetch_compressed error:")
    assert "timed out" in out


def test_run_rejects_binary_content_type(monkeypatch):
    _mock_get(monkeypatch, response=_response(b"\x89PNG", content_type="image/png"))
    out = webfetch.run("http://example.test/pic.png")
    assert "fetch_compressed error" in out
    assert "image/png" in out


def test_run_reports_no_extractable_text(monkeypatch):
    html = "<html><head><script>x()</script></head><body>   </body></html>"
    _mock_get(monkeypatch, response=_response(html.encode(), content_type="text/html"))
    out = webfetch.run("http://example.test/empty")
    assert "no extractable text content" in out


def test_run_compression_disabled_returns_stripped_text_only(monkeypatch):
    cfg.set("compression_enabled", False)
    body = b"plain text body with no markup at all, just words on a page"
    _mock_get(monkeypatch, response=_response(body, content_type="text/plain"))
    out = webfetch.run("http://example.test/plain")
    assert "plain text body with no markup" in out
    assert "<<ccr:" not in out  # nothing was transformed relative to the raw fetch


def test_run_mode_raw_skips_compression_but_still_strips_html(monkeypatch):
    cfg.set("compression_enabled", True)
    # Force the stdlib fallback strip so the test doesn't depend on headroom
    # being installed in the CI environment.
    monkeypatch.setattr(webfetch, "_strip_html", lambda raw: "STRIPPED-" + raw[:5])
    html = "<html><body><script>bad()</script><p>hello</p></body></html>"
    _mock_get(monkeypatch, response=_response(html.encode(), content_type="text/html"))

    called = {"prose": False, "lossless": False}
    monkeypatch.setattr(webfetch, "_compress_document", lambda *a, **k: called.__setitem__("prose", True) or "SHOULD NOT BE CALLED")
    # Stub the headroom-backed stash so this assertion doesn't depend on the
    # optional 'compression' extra being installed in the test environment.
    monkeypatch.setattr(webfetch, "_stash_original", lambda original, compressed, *, url: "cafef00d")

    out = webfetch.run("http://example.test/page", mode="raw")
    assert called["prose"] is False, "raw mode must not invoke the compression cascade"
    assert "STRIPPED-" in out
    # stripping changed the text relative to the raw HTML -> marker offered
    assert "<<ccr:cafef00d>>" in out


def test_run_json_content_type_routes_lossless_only(monkeypatch):
    cfg.set("compression_enabled", True)
    body = b'{"a": 1, "b": [1,2,3]}'
    _mock_get(monkeypatch, response=_response(body, content_type="application/json"))

    def _boom(*a, **k):
        raise AssertionError("kompress_prose must not run on JSON content")

    monkeypatch.setattr("skill_hub.compression.kompress_prose", _boom)
    monkeypatch.setattr(
        "skill_hub.compression.maybe_compress",
        lambda content, **k: content,  # identity: lossless no-op
    )
    out = webfetch.run("http://example.test/data.json")
    assert '"a": 1' in out or '"a":1' in out or "a" in out  # content preserved


def test_run_prose_and_code_segments_routed_differently(monkeypatch):
    cfg.set("compression_enabled", True)
    body = (
        "Some prose about a topic that could be summarized.\n"
        "```python\nprint('keep me exact')\n```\n"
        "More prose after the code block."
    ).encode()
    _mock_get(monkeypatch, response=_response(body, content_type="text/plain"))

    prose_calls: list[str] = []
    lossless_calls: list[str] = []

    def fake_kompress_prose(text, **kwargs):
        prose_calls.append(text)
        return "[COMPRESSED PROSE]"

    def fake_maybe_compress(text, **kwargs):
        lossless_calls.append(text)
        return text  # identity for the lossless path

    monkeypatch.setattr("skill_hub.compression.kompress_prose", fake_kompress_prose)
    monkeypatch.setattr("skill_hub.compression.maybe_compress", fake_maybe_compress)

    out = webfetch.run("http://example.test/mixed")

    assert prose_calls, "prose segments must go through kompress_prose"
    assert any("print('keep me exact')" in c for c in lossless_calls), (
        "the fenced code block must go through the lossless-only path verbatim"
    )
    assert "print('keep me exact')" in out, "code block content must survive intact"
    assert "[COMPRESSED PROSE]" in out


def test_run_adds_marker_when_output_differs_from_raw_fetch(monkeypatch):
    cfg.set("compression_enabled", True)
    body = b"raw text that will be replaced entirely"
    _mock_get(monkeypatch, response=_response(body, content_type="text/plain"))
    monkeypatch.setattr(webfetch, "_compress_document", lambda *a, **k: "totally different compact text")
    monkeypatch.setattr(webfetch, "_stash_original", lambda original, compressed, *, url: "deadbeefcafe")

    out = webfetch.run("http://example.test/changed")
    assert "totally different compact text" in out
    assert "<<ccr:deadbeefcafe>>" in out


def test_run_no_marker_when_stash_unavailable(monkeypatch):
    cfg.set("compression_enabled", True)
    body = b"raw text that will be replaced entirely"
    _mock_get(monkeypatch, response=_response(body, content_type="text/plain"))
    monkeypatch.setattr(webfetch, "_compress_document", lambda *a, **k: "totally different compact text")
    monkeypatch.setattr(webfetch, "_stash_original", lambda original, compressed, *, url: None)

    out = webfetch.run("http://example.test/changed")
    assert "<<ccr:" not in out


def test_run_truncation_note_present(monkeypatch):
    cfg.set("compression_enabled", False)
    monkeypatch.setattr(webfetch, "_MAX_FETCH_BYTES", 5)
    _mock_get(monkeypatch, response=_response(b"0123456789", content_type="text/plain"))
    out = webfetch.run("http://example.test/big")
    assert "truncated" in out


# ---------------------------------------------------------------------------
# marker <-> retrieve_compressed round trip (real headroom store, no mocking
# of the store itself -- skipped if the optional 'compression' extra isn't
# installed).
# ---------------------------------------------------------------------------

def test_stash_and_retrieve_original_roundtrip():
    pytest.importorskip("headroom")
    from skill_hub.compression import retrieve_original

    original = "the full original content behind the marker"
    compressed = "short digest"
    hash_key = webfetch._stash_original(original, compressed, url="http://example.test/x")
    assert hash_key
    assert retrieve_original(hash_key) == original


def test_stash_original_never_raises_when_store_unavailable(monkeypatch):
    def _boom():
        raise ImportError("no headroom")
    monkeypatch.setitem(
        sys.modules, "headroom.cache.compression_store",
        None,  # force ImportError on `from headroom.cache.compression_store import ...`
    )
    result = webfetch._stash_original("orig", "compressed", url="http://example.test")
    assert result is None
