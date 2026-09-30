"""Measure real minimal-MCP responses over isolated stdio with fixed reads.

No LLM selects sources. The fixture clock and IDs are fixed; the server, schemas,
response serialization and scope checks are real. Counts are not client billing.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from benchmarks.context_compare import load_tokenizer
from benchmarks.installed_smoke import _child_env
from benchmarks.modular_context import SCOPE, modules, seed, serialize
from skill_hub import config
from skill_hub.store import SkillStore

ROOT = Path(__file__).resolve().parents[1]
CHILD = """
import itertools
from benchmarks.modular_context import FixedClock
from skill_hub import context_composer
sequence = itertools.count()
context_composer.datetime = FixedClock
context_composer._opaque_id = lambda prefix: f'{prefix}_{next(sequence):032x}'
from skill_hub.mcp_entry import main
main()
"""


async def measure(size: int, language: str, encoding) -> dict:
    docs = modules(size)
    prompt = ("Review migration decisions. Do not change public API compatibility."
              if language == "en" else
              "Verifica le decisioni di migration. Non cambiare la compatibilità delle API.")

    def tokens(value) -> int:
        return len(encoding.encode(value if isinstance(value, str) else serialize(value)))

    def measurements(result, request) -> dict:
        envelope = {"content": [part.model_dump(mode="json", exclude_none=True) for part in result.content],
                "structuredContent": result.structured_content, "isError": result.is_error}
        return {"request_tokens": tokens(request),
                "structured_tokens": tokens(result.structured_content),
                "text_content_tokens": sum(tokens(part.text) for part in result.content if hasattr(part, "text")),
                "result_envelope_tokens": tokens(envelope),
                "result_envelope_sha256": hashlib.sha256(serialize(envelope).encode()).hexdigest()}

    with tempfile.TemporaryDirectory(prefix="mcp-context-payloads-") as directory:
        root = Path(directory)
        home = root / "home"
        state = home / ".claude" / "mcp-skill-hub"
        state.mkdir(parents=True)
        cfg = state / "config.json"
        cfg.write_text(serialize({"services": {"auto_reconcile": False},
                                  "continuous_sweep_enabled": False, "reindex_sweep_enabled": False,
                                  "context_project_aliases": {}}), encoding="utf-8")
        with patch.object(config, "CONFIG_PATH", cfg):
            store = SkillStore(db_path=state / "skill_hub.db")
            try:
                seed(store, docs)
            finally:
                store.close()
        env = _child_env(home, ROOT / "src")
        env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "src"), str(ROOT)))
        transport = StdioTransport(command=sys.executable,
                                   args=["-c", CHILD, "--profile", "minimal"],
                                   env=env, cwd=str(root))
        async with Client(transport, timeout=45, init_timeout=45) as client:
            listed = await client.list_tools()
            catalog = {"tools": [tool.model_dump(mode="json", exclude_none=True) for tool in listed],
                       "instructions": client.initialize_result.instructions}
            assert len(listed) == 8

            async def call(name, arguments):
                result = await client.call_tool(name, arguments)
                assert not result.is_error, result
                assert "FOREIGN_SCOPE" not in serialize(result.structured_content)
                return result.structured_content, measurements(result, {"name": name, "arguments": arguments})

            request = {"prompt": prompt, "project_roots": [SCOPE], "mode": "manual"}
            preview, preview_measure = await call("prepare_composition", request)
            index, index_measure = await call("prepare_composition", {**request, "detail": "index"})
            assert preview["original_prompt"] == index["original_prompt"] == prompt
            assert preview["warnings"] == index["warnings"]
            assert preview["needs_review"] == index["needs_review"]
            assert len(index["candidates"]) == len(preview["candidates"]) == 6
            assert all("text" not in item and "features" not in item for item in index["candidates"])
            expected = {"memory:" + doc["id"]: doc["text"] for doc in docs}
            old = sorted(preview["candidates"], key=lambda item: item["source"])
            new = sorted(index["candidates"], key=lambda item: item["source"])
            scenarios = []
            for amount in (1, 2, 6):
                old_measures = [preview_measure]
                old_texts = []
                for item in old[:amount]:
                    data, measure_result = await call("expand_context_candidate", {
                        "draft_id": preview["draft_id"], "candidate_id": item["candidate_id"],
                    })
                    assert data["text"] == expected[data["source"]]
                    old_texts.append(data["text"])
                    old_measures.append(measure_result)
                expanded, compact_measure = await call("expand_context_candidate", {
                    "draft_id": index["draft_id"],
                    "candidate_id": [item["candidate_id"] for item in new[:amount]],
                    "detail": "compact",
                })
                assert [item["text"] for item in expanded["items"]] == old_texts
                assert all("features" not in item for item in expanded["items"])
                new_measures = [index_measure, compact_measure]

                def total(rows):
                    return {key: sum(item[key] for item in rows) for key in
                            ("request_tokens", "structured_tokens", "text_content_tokens", "result_envelope_tokens")}

                before, after = total(old_measures), total(new_measures)
                base = before["request_tokens"] + before["structured_tokens"]
                final = after["request_tokens"] + after["structured_tokens"]
                scenarios.append({"read_count": amount, "legacy_tool_calls": amount + 1,
                                  "compact_tool_calls": 2, "legacy": before, "compact": after,
                                  "request_plus_structured_reduction_percent": round((1 - final / base) * 100, 2),
                                  "exact_source_texts_match": True})
            return {"paragraphs_per_module": size, "language": language,
                    "corpus_sha256": hashlib.sha256(serialize(docs).encode()).hexdigest(),
                    "tool_count": len(listed), "catalog_tokens": tokens(catalog),
                    "preview_response": preview_measure, "index_response": index_measure,
                    "prompt_preserved": True, "warnings_preserved": True,
                    "foreign_scope_leaks": 0, "scenarios": scenarios}


async def run() -> dict:
    encoding = load_tokenizer()
    return {"evidence_level": "real_stdio_mcp_fixed_read_schedule", "tokenizer": "o200k_base",
            "model_calls_requested": 0,
            "limitations": ["Fixture clock and opaque IDs are fixed; source checkout is used, not an installed wheel.",
                            "Structured, text-content and reconstructed result-envelope counts are alternatives; do not add them.",
                            "Request/response sums count each once; no repeated model context, billing, cache or reasoning is measured.",
                            "Legacy comparator is preview plus full reads, not an optimally hand-selected lean context.",
                            "Fixed read schedules do not establish model selection recall or completed-task savings.",
                            "Composer still loads source text internally; no I/O or memory savings are claimed."],
            "cases": [await measure(size, language, encoding)
                      for size in (0, 12) for language in ("en", "it")]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(run())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
