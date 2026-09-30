"""Offline payload probe for indexed context with explicit on-demand reads.

This uses real scoped composer preparation/expansion, with an experimental
metadata projection at the caller boundary. Read schedules are fixed, not LLM
decisions. Results measure serialized text, not billed or whole-task tokens.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from benchmarks.context_compare import load_tokenizer
from skill_hub import config, context_composer
from skill_hub.compression import compress_payload
from skill_hub.store import SkillStore

SCOPE = "/synthetic/project-a"
GUARD = "Preserve the original request. Do not change public API compatibility."
TOPICS = ("rollback", "database", "authorization", "cache", "timeouts", "deployment")


class FixedClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 1, 2, tzinfo=UTC).astimezone(tz)


def serialize(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def modules(paragraphs: int) -> list[dict]:
    return [{
        "id": topic,
        "text": "\n".join([
            f"Migration decision for {topic}. RULE_{topic.upper()}: "
            f"verify {topic} before deployment; never skip this check.",
            *[f"Evidence {i}: preserve revision {i + 10}, inspect /service/{topic}/{i}, "
              "and record the outcome without changing the approval boundary."
              for i in range(paragraphs)],
        ]),
    } for topic in TOPICS]


def seed(store: SkillStore, docs: list[dict]) -> None:
    for project, prefix in ((SCOPE, ""), ("/synthetic/project-b", "foreign-")):
        for doc in docs:
            name = prefix + doc["id"]
            content = ("FOREIGN_SCOPE " if prefix else "") + doc["text"]
            store._conn.execute(
                "INSERT INTO vectors "
                "(namespace,doc_id,vector,norm,metadata,level,source,project,indexed_at) "
                "VALUES ('memory:project',?,'[]',0,'{}','L3',?,?,?)",
                (name, "memory:" + name, project, "2026-01-01"),
            )
            store._conn.execute(
                "INSERT INTO context_digests "
                "(key,content_hash,digest,content,updated_at) VALUES (?,?,?,?,?)",
                ("memory:" + name, hashlib.sha256(content.encode()).hexdigest(),
                 content, content, "2026-01-01"),
            )
    store._conn.commit()


def probe(paragraphs: int, language: str = "en") -> dict:
    docs = modules(paragraphs)
    prompt = ("Review migration decisions before deployment. Preserve API compatibility."
              if language == "en" else
              "Verifica le decisioni di migration prima del deployment. Non cambiare le API.")
    encoder = load_tokenizer()

    def count(value: object) -> int:
        return len(encoder.encode(value if isinstance(value, str) else serialize(value)))

    with tempfile.TemporaryDirectory(prefix="modular-context-") as directory:
        root = Path(directory)
        cfg = root / "config.json"
        cfg.write_text('{"context_project_aliases":{}}', encoding="utf-8")
        store = SkillStore(db_path=root / "probe.db")
        try:
            seed(store, docs)
            sequence = itertools.count()
            # Fixed opaque IDs stabilize serialization counts across reruns.
            with patch.object(config, "CONFIG_PATH", cfg), patch.object(
                context_composer, "datetime", FixedClock
            ), patch.object(
                context_composer, "_opaque_id", lambda prefix: f"{prefix}_{next(sequence):032x}"
            ):
                draft = context_composer.prepare_composition(
                    prompt, project_roots=[SCOPE], mode="manual", store=store,
                )
                assert len(draft["candidates"]) == len(docs)
                assert draft["original_prompt"] == prompt
                assert "FOREIGN_SCOPE" not in serialize(draft)
                candidates = sorted(draft["candidates"], key=lambda item: item["source"])
                expanded = [context_composer.get_composition_candidate(
                    draft["draft_id"], item["candidate_id"], store=store
                ) for item in candidates]
                assert {item["text"] for item in expanded} == {doc["text"] for doc in docs}
                assert "FOREIGN_SCOPE" not in serialize(expanded)
                index = {
                    "draft_id": draft["draft_id"], "mandatory": GUARD,
                    "needs_review": draft["needs_review"], "warnings": draft["warnings"],
                    "instruction": "Read relevant modules with expand_context_candidate before relying on their rules.",
                    "modules": [{key: item[key] for key in
                                 ("candidate_id", "title", "kind", "source", "project_root")}
                                for item in candidates],
                }
                full = serialize({"mandatory": GUARD, "modules": [
                    {"source": item["source"], "text": item["text"]} for item in expanded
                ]})
                safe = compress_payload(full, allow_lossy=False)
                rows = []
                for reads in (0, 1, 2, len(candidates)):
                    requests = [{"name": "expand_context_candidate", "arguments": {
                        "draft_id": draft["draft_id"], "candidate_id": item["candidate_id"],
                    }} for item in candidates[:reads]]
                    responses = expanded[:reads]
                    request_tokens = sum(count(item) for item in requests)
                    response_tokens = sum(count(item) for item in responses)
                    distinct = count(index) + request_tokens + response_tokens
                    # Uncached illustrative transcript replay: initial model turn,
                    # then one turn after each sequential tool response. Excludes
                    # all model reasoning/final answer and protocol framing.
                    replay = count(prompt) + count(index)
                    history = count(prompt) + count(index)
                    for request, response in zip(requests, responses):
                        history += count(request) + count(response)
                        replay += history
                    rows.append({
                        "read_count": reads, "index_tokens": count(index),
                        "request_tokens": request_tokens, "response_tokens": response_tokens,
                        "distinct_payload_tokens": distinct,
                        "delta_vs_lean_full_payload_percent": round((distinct / count(full) - 1) * 100, 2),
                        "illustrative_uncached_replayed_input_tokens": replay,
                        "read_sources": [item["source"] for item in responses],
                        "expected_read_markers_present": all(
                            "RULE_" + item["source"].split(":")[-1].upper() in item["text"]
                            for item in responses),
                    })
                first = candidates[0]
                store._conn.execute("UPDATE context_digests SET content = content || ' changed' WHERE key = ?",
                                    (first["source"],))
                store._conn.commit()
                stale_rejected = False
                try:
                    context_composer.get_composition_candidate(draft["draft_id"], first["candidate_id"], store=store)
                except ValueError as exc:
                    stale_rejected = "stale" in str(exc)
                return {
                    "language": language, "paragraphs_per_module": paragraphs,
                    "corpus_sha256": hashlib.sha256(serialize(docs).encode()).hexdigest(),
                    "prompt_tokens": count(prompt), "full_context_tokens": count(full),
                    "safe_compressed_context_tokens": count(safe.compressed),
                    "safe_transform": safe.content_type,
                    "current_prepare_response_tokens": count(draft),
                    "projected_index_tokens": count(index),
                    "stale_source_rejected": stale_rejected, "foreign_scope_leaks": 0,
                    "prompt_preserved": True, "schedules": rows,
                }
        finally:
            store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "evidence_level": "offline_fixed_read_schedule_payload_probe",
        "tokenizer": "o200k_base", "model_calls": 0,
        "limitations": [
            "Index projection is experimental, not the current MCP prepare response.",
            "Full baseline is a lean source/text serialization, not current prepare_composition output.",
            "Fixture clock and opaque IDs are fixed for reproducible serialization counts.",
            "Composer still reads full sources internally; no I/O or RAM savings measured.",
            "Fixed reads do not establish model selection recall, correctness or task savings.",
            "Counts exclude tool schemas, protocol framing, reasoning and final answer.",
            "Sequential replay is illustrative uncached input accounting, not client usage.",
        ],
        "cases": [probe(size, language) for size in (0, 12) for language in ("en", "it")],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(serialize(report))


if __name__ == "__main__":
    main()
