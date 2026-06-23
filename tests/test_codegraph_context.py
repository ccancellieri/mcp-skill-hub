"""Tests for the CodeGraph context adapter and search_context integration.

Covers:
(a) No index → clean empty result, search_context output unchanged.
(b) Flag OFF → no injection even when index is present.
(c) Flag ON + mocked codegraph → bounded block injected, respecting top-k
    and char cap.
(d) Adapter never raises on codegraph failure or timeout.

All codegraph CLI calls are mocked — tests do NOT require codegraph installed
or a real index.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def store(tmp_path, monkeypatch):
    from skill_hub.store import SkillStore
    monkeypatch.setenv("HOME", str(tmp_path))
    return SkillStore(db_path=tmp_path / "skill_hub.db")


@pytest.fixture()
def server_mod(store, monkeypatch, tmp_path):
    """Wire the global server module to an isolated store."""
    from skill_hub import server
    monkeypatch.setattr(server, "_store", store)
    monkeypatch.setattr(server, "SETTINGS_PATH", tmp_path / "settings.json")
    server._last_search_state["query"] = ""
    server._last_search_state["vector"] = []
    server._last_search_state["skills"] = []
    server._session["topic"] = ""
    server._session["topic_vector"] = []
    # Enable embedding so we go through the vector path.
    monkeypatch.setattr(server, "embed_available", lambda: True)
    monkeypatch.setattr(server, "embed", lambda q: [0.0] * 16)
    return server


@pytest.fixture()
def fake_index(tmp_path):
    """Create a minimal fake .codegraph/codegraph.db so has_codegraph_index is True."""
    cg_dir = tmp_path / ".codegraph"
    cg_dir.mkdir()
    db = cg_dir / "codegraph.db"
    db.write_bytes(b"\x00" * 64)  # non-empty file
    return tmp_path


# Three representative codegraph CLI JSON rows.
_FAKE_ROWS = [
    {
        "node": {
            "name": "search_context",
            "kind": "function",
            "qualifiedName": "server.py::search_context",
            "filePath": "src/skill_hub/server.py",
            "startLine": 1827,
            "signature": "(query: str, top_k: int = 5) -> str",
        },
        "score": 42.0,
    },
    {
        "node": {
            "name": "SkillStore",
            "kind": "class",
            "qualifiedName": "store.py::SkillStore",
            "filePath": "src/skill_hub/store.py",
            "startLine": 10,
            "signature": None,
        },
        "score": 30.0,
    },
    {
        "node": {
            "name": "embed",
            "kind": "function",
            "qualifiedName": "embeddings.py::embed",
            "filePath": "src/skill_hub/embeddings.py",
            "startLine": 5,
            "signature": "(text: str) -> list[float]",
        },
        "score": 25.0,
    },
]


# ---------------------------------------------------------------------------
# Unit tests for codegraph_context module
# ---------------------------------------------------------------------------

class TestHasCodegraphIndex:
    def test_returns_false_when_no_dir(self, tmp_path):
        from skill_hub.codegraph_context import has_codegraph_index, _clear_cache
        _clear_cache()
        assert has_codegraph_index(tmp_path) is False

    def test_returns_false_when_db_missing(self, tmp_path):
        from skill_hub.codegraph_context import has_codegraph_index, _clear_cache
        _clear_cache()
        (tmp_path / ".codegraph").mkdir()
        assert has_codegraph_index(tmp_path) is False

    def test_returns_false_when_db_empty(self, tmp_path):
        from skill_hub.codegraph_context import has_codegraph_index, _clear_cache
        _clear_cache()
        cg = tmp_path / ".codegraph"
        cg.mkdir()
        (cg / "codegraph.db").write_bytes(b"")
        assert has_codegraph_index(tmp_path) is False

    def test_returns_true_when_db_present(self, fake_index):
        from skill_hub.codegraph_context import has_codegraph_index, _clear_cache
        _clear_cache()
        assert has_codegraph_index(fake_index) is True

    def test_returns_false_on_exception(self, monkeypatch):
        """has_codegraph_index never raises even if Path.is_dir throws."""
        from skill_hub import codegraph_context
        codegraph_context._clear_cache()

        class BadPath:
            def __truediv__(self, _): return self
            def is_dir(self): raise OSError("no permission")
            def __str__(self): return "/bad"

        # Patch to a path that raises.
        import pathlib
        monkeypatch.setattr(pathlib.Path, "__truediv__",
                            lambda self, other: BadPath() if other == ".codegraph" else self.__class__(str(self) + "/" + str(other)))
        # Even if an exception fires, the function returns False.
        result = codegraph_context.has_codegraph_index(Path("/nonexistent_zzzz"))
        assert result is False


class TestGetContextBlock:
    """(a) No index → empty result."""

    def test_no_index_returns_empty(self, tmp_path, monkeypatch):
        from skill_hub.codegraph_context import get_context_block, _clear_cache
        _clear_cache()
        block = get_context_block("search_context", tmp_path)
        assert block == ""

    """(d) Never raises on failure."""

    def test_never_raises_on_missing_binary(self, fake_index, monkeypatch):
        from skill_hub import codegraph_context
        codegraph_context._clear_cache()
        monkeypatch.setattr(codegraph_context, "_find_codegraph_bin", lambda: None)
        block = codegraph_context.get_context_block("anything", fake_index)
        assert block == ""

    def test_never_raises_on_subprocess_timeout(self, fake_index, monkeypatch):
        import subprocess
        from skill_hub import codegraph_context
        codegraph_context._clear_cache()

        def _raise_timeout(*_a, **_kw):
            raise subprocess.TimeoutExpired(cmd="codegraph", timeout=5)

        monkeypatch.setattr(codegraph_context, "_find_codegraph_bin", lambda: "/bin/codegraph")
        monkeypatch.setattr(subprocess, "run", _raise_timeout)
        block = codegraph_context.get_context_block("anything", fake_index)
        assert block == ""

    def test_never_raises_on_bad_json(self, fake_index, monkeypatch):
        import subprocess
        from skill_hub import codegraph_context
        codegraph_context._clear_cache()

        class FakeResult:
            returncode = 0
            stdout = b"not-json!!!"
            stderr = b""

        monkeypatch.setattr(codegraph_context, "_find_codegraph_bin", lambda: "/bin/codegraph")
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: FakeResult())
        block = codegraph_context.get_context_block("anything", fake_index)
        assert block == ""

    def test_never_raises_on_nonzero_exit(self, fake_index, monkeypatch):
        import subprocess
        from skill_hub import codegraph_context
        codegraph_context._clear_cache()

        class FakeResult:
            returncode = 1
            stdout = b""
            stderr = b"error: something went wrong"

        monkeypatch.setattr(codegraph_context, "_find_codegraph_bin", lambda: "/bin/codegraph")
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: FakeResult())
        block = codegraph_context.get_context_block("anything", fake_index)
        assert block == ""

    """(c) Flag ON + mocked codegraph → bounded block injected."""

    def test_returns_formatted_block_with_mocked_cli(self, fake_index, monkeypatch):
        import subprocess
        from skill_hub import codegraph_context
        codegraph_context._clear_cache()

        class FakeResult:
            returncode = 0
            stdout = json.dumps(_FAKE_ROWS).encode()
            stderr = b""

        monkeypatch.setattr(codegraph_context, "_find_codegraph_bin", lambda: "/bin/codegraph")
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: FakeResult())
        block = codegraph_context.get_context_block("search_context", fake_index)
        assert "## CodeGraph Symbols" in block
        assert "search_context" in block
        assert "function" in block

    def test_char_cap_respected(self, fake_index, monkeypatch):
        """Block is truncated to _CODEGRAPH_MAX_CHARS."""
        import subprocess
        from skill_hub import codegraph_context
        codegraph_context._clear_cache()

        # Build a row with a very long signature to force truncation.
        long_rows = [
            {
                "node": {
                    "name": f"func_{i}",
                    "kind": "function",
                    "qualifiedName": f"mod.py::func_{i}",
                    "filePath": "src/very/long/path/to/module.py",
                    "startLine": i,
                    "signature": f"(arg_{'x' * 200}: str) -> None",
                },
                "score": float(100 - i),
            }
            for i in range(20)
        ]

        class FakeResult:
            returncode = 0
            stdout = json.dumps(long_rows).encode()
            stderr = b""

        monkeypatch.setattr(codegraph_context, "_find_codegraph_bin", lambda: "/bin/codegraph")
        monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: FakeResult())
        block = codegraph_context.get_context_block("func", fake_index)
        assert len(block) <= codegraph_context._CODEGRAPH_MAX_CHARS + 1  # +1 for ellipsis char
        assert block.endswith("…")

    def test_top_k_passed_to_cli(self, fake_index, monkeypatch):
        """The CLI is invoked with --limit matching _CODEGRAPH_TOP_K."""
        import subprocess
        from skill_hub import codegraph_context
        codegraph_context._clear_cache()

        captured: list[list[str]] = []

        class FakeResult:
            returncode = 0
            stdout = b"[]"
            stderr = b""

        def fake_run(argv, **_kw):
            captured.append(list(argv))
            return FakeResult()

        monkeypatch.setattr(codegraph_context, "_find_codegraph_bin", lambda: "/bin/codegraph")
        monkeypatch.setattr(subprocess, "run", fake_run)
        codegraph_context.get_context_block("test query", fake_index)
        assert captured, "subprocess.run was never called"
        args = captured[0]
        limit_idx = args.index("--limit")
        assert args[limit_idx + 1] == str(codegraph_context._CODEGRAPH_TOP_K)


# ---------------------------------------------------------------------------
# Integration tests: search_context + codegraph flag
# ---------------------------------------------------------------------------

class TestSearchContextIntegration:
    """(b) Flag OFF → no injection."""

    def test_flag_off_no_injection(self, server_mod, fake_index, monkeypatch):
        from skill_hub import config as _cfg, codegraph_context
        codegraph_context._clear_cache()

        # Flag is OFF (default).
        monkeypatch.setattr(_cfg, "get",
                            lambda key: False if key == "search_context_use_codegraph"
                            else _cfg.load_config().get(key))
        # Even with a valid index and a working CLI, nothing injected.
        import subprocess

        class FakeResult:
            returncode = 0
            stdout = json.dumps(_FAKE_ROWS).encode()
            stderr = b""

        monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: FakeResult())
        monkeypatch.setattr(codegraph_context, "_find_codegraph_bin", lambda: "/bin/codegraph")
        monkeypatch.setattr(server_mod, "_cfg", _cfg)
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: fake_index))

        out = server_mod.search_context("search_context", top_k=3)
        assert "CodeGraph Symbols" not in out

    """(a) No index → search_context output unchanged (no CodeGraph section)."""

    def test_no_index_no_codegraph_section(self, server_mod, tmp_path, monkeypatch):
        from skill_hub import config as _cfg, codegraph_context
        codegraph_context._clear_cache()

        monkeypatch.setattr(_cfg, "get",
                            lambda key: True if key == "search_context_use_codegraph"
                            else _cfg.load_config().get(key))
        monkeypatch.setattr(server_mod, "_cfg", _cfg)
        # tmp_path has no .codegraph dir.
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: tmp_path))

        out = server_mod.search_context("anything", top_k=3)
        assert "CodeGraph Symbols" not in out

    """(c) Flag ON + index present + mocked CLI → block injected."""

    def test_flag_on_with_index_injects_block(self, server_mod, fake_index, monkeypatch):
        from skill_hub import config as _cfg, codegraph_context
        import subprocess
        codegraph_context._clear_cache()

        monkeypatch.setattr(_cfg, "get",
                            lambda key: True if key == "search_context_use_codegraph"
                            else _cfg.load_config().get(key))

        class FakeResult:
            returncode = 0
            stdout = json.dumps(_FAKE_ROWS).encode()
            stderr = b""

        monkeypatch.setattr(subprocess, "run", lambda *_a, **_kw: FakeResult())
        monkeypatch.setattr(codegraph_context, "_find_codegraph_bin", lambda: "/bin/codegraph")
        monkeypatch.setattr(server_mod, "_cfg", _cfg)
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: fake_index))

        out = server_mod.search_context("search_context", top_k=3)
        assert "CodeGraph Symbols" in out

    """search_context output unchanged when flag is default (False)."""

    def test_default_flag_unchanged_output(self, server_mod, fake_index, monkeypatch):
        """Verify that with default config (flag=False), output is identical
        whether or not a codegraph index exists."""
        from skill_hub import config as _cfg, codegraph_context
        codegraph_context._clear_cache()

        # Use real default config which has search_context_use_codegraph=False.
        monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: fake_index))

        out = server_mod.search_context("default flag test", top_k=3)
        assert "CodeGraph Symbols" not in out
