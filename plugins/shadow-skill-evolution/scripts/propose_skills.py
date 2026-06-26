#!/usr/bin/env python3
"""Scheduled task: cluster tool-chain patterns and generate skill proposals.

Reads tool_chains table, embeds sequences, clusters by similarity, and
when a cluster reaches the threshold (default 3), generates a skill proposal
via local LLM for human approval.

Usage:
    skill-hub-cli run-plugin-task shadow-skill-evolution propose_skills

Or via scheduled task runner (cowork/cron).
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "skill_hub.db"
PLUGIN_DB_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "plugins" / "shadow_skill_evolution.db"
LOCAL_SKILLS_DIR = Path.home() / ".claude" / "local-skills"

DEFAULT_CLUSTER_THRESHOLD = 3
DEFAULT_SIMILARITY_THRESHOLD = 0.85


@dataclass
class Chain:
    id: int
    chain_hash: str
    tool_sequence: list[tuple[str, str]]
    embedding: list[float] | None
    occurrence_count: int
    metadata: dict


def _get_store_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _get_plugin_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(PLUGIN_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0


def _embed_text(text: str) -> list[float]:
    from skill_hub import store as _store

    s = _store.get_store()
    return s.embed(text)


def _get_existing_proposal_hashes(plugin_conn: sqlite3.Connection) -> set[str]:
    rows = plugin_conn.execute(
        "SELECT source_chains FROM skill_proposals WHERE status IN ('pending', 'approved')"
    ).fetchall()
    hashes = set()
    for row in rows:
        try:
            chains = json.loads(row["source_chains"])
            hashes.update(chains)
        except (json.JSONDecodeError, TypeError):
            pass
    return hashes


def _load_chains(plugin_conn: sqlite3.Connection) -> list[Chain]:
    rows = plugin_conn.execute(
        "SELECT id, chain_hash, tool_sequence, embedding, occurrence_count, metadata "
        "FROM tool_chains ORDER BY last_seen_at DESC LIMIT 1000"
    ).fetchall()

    chains = []
    for row in rows:
        try:
            tool_seq = json.loads(row["tool_sequence"])
            embedding = json.loads(row["embedding"]) if row["embedding"] else None
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
        except json.JSONDecodeError:
            continue

        chains.append(Chain(
            id=row["id"],
            chain_hash=row["chain_hash"],
            tool_sequence=tool_seq,
            embedding=embedding,
            occurrence_count=row["occurrence_count"],
            metadata=metadata,
        ))

    return chains


def _cluster_chains(chains: list[Chain], threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> list[list[Chain]]:
    clusters: list[list[Chain]] = []
    assigned = set()

    for chain in chains:
        if chain.id in assigned:
            continue
        if not chain.embedding:
            continue

        cluster = [chain]
        assigned.add(chain.id)

        for other in chains:
            if other.id in assigned:
                continue
            if not other.embedding:
                continue

            sim = _cosine(chain.embedding, other.embedding)
            if sim >= threshold:
                cluster.append(other)
                assigned.add(other.id)

        if len(cluster) >= DEFAULT_CLUSTER_THRESHOLD:
            clusters.append(cluster)

    return clusters


def _generate_skill_proposal(cluster: list[Chain], llm_tier: str = "tier_mid") -> dict[str, Any] | None:
    try:
        from skill_hub.llm import litellm_adapter as _llm
    except ImportError:
        return None

    representative = cluster[0]
    tool_names = [t for t, _ in representative.tool_sequence]

    prompt = f"""Analyze this recurring tool-call pattern and generate a local skill proposal.

Tool sequence: {' → '.join(tool_names)}

Occurrence count: {sum(c.occurrence_count for c in cluster)} across {len(cluster)} sessions

Generate a JSON object with:
{{
  "name": "skill-slug-name",
  "title": "Human-readable title",
  "description": "What this skill does",
  "triggers": ["trigger phrase 1", "trigger phrase 2"],
  "steps": [
    {{"read": "path/to/file", "as": "varname"}},
    {{"llm": "prompt text", "as": "result"}},
    {{"mcp": "tool_name", "args": {{"arg": "value"}}}}
  ]
}}

Only output the JSON, no explanation."""

    try:
        result = _llm.complete(prompt, tier=llm_tier, max_tokens=800)
        if not result or not result.get("content"):
            return None
        content = result["content"].strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].startswith("```") else lines[1:])
        return json.loads(content)
    except (json.JSONDecodeError, Exception):
        return None


def _store_proposal(
    plugin_conn: sqlite3.Connection,
    proposal: dict[str, Any],
    cluster: list[Chain],
    similarity_score: float,
) -> int:
    source_chains = json.dumps([c.chain_hash for c in cluster])
    cur = plugin_conn.execute(
        "INSERT INTO skill_proposals "
        "(name, title, description, triggers, steps, source_chains, cluster_size, similarity_score) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            proposal.get("name", "unnamed-skill"),
            proposal.get("title", "Untitled Skill"),
            proposal.get("description", ""),
            json.dumps(proposal.get("triggers", [])),
            json.dumps(proposal.get("steps", [])),
            source_chains,
            len(cluster),
            similarity_score,
        ),
    )
    plugin_conn.commit()
    return int(cur.lastrowid)


def _embed_unembedded_chains(plugin_conn: sqlite3.Connection, chains: list[Chain]) -> None:
    for chain in chains:
        if chain.embedding:
            continue
        text = " → ".join(t for t, _ in chain.tool_sequence)
        try:
            embedding = _embed_text(text)
            plugin_conn.execute(
                "UPDATE tool_chains SET embedding = ? WHERE id = ?",
                (json.dumps(embedding), chain.id),
            )
        except Exception:
            pass
    plugin_conn.commit()


def main() -> int:
    try:
        plugin_conn = _get_plugin_conn()
    except sqlite3.Error:
        print("Failed to connect to plugin database")
        return 1

    existing_hashes = _get_existing_proposal_hashes(plugin_conn)
    chains = _load_chains(plugin_conn)

    chains = [c for c in chains if c.chain_hash not in existing_hashes]

    if not chains:
        print("No new tool chains to process")
        return 0

    print(f"Loaded {len(chains)} tool chains")

    _embed_unembedded_chains(plugin_conn, chains)
    chains = _load_chains(plugin_conn)

    clusters = _cluster_chains(chains)
    print(f"Found {len(clusters)} clusters above threshold")

    proposals_created = 0
    for cluster in clusters:
        if len(cluster) < DEFAULT_CLUSTER_THRESHOLD:
            continue

        avg_sim = 0.0
        if cluster[0].embedding:
            sims = []
            for c in cluster[1:]:
                if c.embedding:
                    sims.append(_cosine(cluster[0].embedding, c.embedding))
            avg_sim = sum(sims) / len(sims) if sims else 0.0

        proposal = _generate_skill_proposal(cluster)
        if proposal:
            proposal_id = _store_proposal(plugin_conn, proposal, cluster, avg_sim)
            print(f"Created proposal {proposal_id}: {proposal.get('name')}")
            proposals_created += 1

    print(f"Created {proposals_created} skill proposals")
    plugin_conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
