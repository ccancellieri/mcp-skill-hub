"""Vector source abstraction — unified catalog + merge protocol.

Each first-class vector-bearing store (skills, tasks, teachings, verdicts)
and the generic `vectors` table (namespaces) becomes a `MergeableSource`.
`SourceRegistry` drives both the Indexes tab and the `/vector/merge/*`
endpoints from one list.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


class MergeMode(str, enum.Enum):
    LLM = "llm"
    MECHANICAL = "mechanical"
    REJECTED = "rejected"


@dataclass
class IndexStats:
    name: str
    source_type: str                   # "first_class_table" | "namespace"
    doc_count: int
    embedded_count: int
    avg_age_days: float | None
    last_indexed: str | None
    embedding_model: str | None
    level_breakdown: dict[str, int]
    scatter_url: str
    supports_merge: bool
    merge_mode: MergeMode


@dataclass
class MergeItem:
    id: str
    label: str
    body: str
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class MergeDraft:
    proposed_label: str
    proposed_body: str
    proposed_raw: dict[str, Any]
    tier_used: str
    tokens_used: int | None = None


@dataclass
class CommitResult:
    new_id: str | None
    closed_ids: list[str]
    audit_id: int | None


@runtime_checkable
class MergeableSource(Protocol):
    name: str
    merge_mode: MergeMode

    def index_stats(self) -> IndexStats: ...
    def fetch_for_merge(self, ids: list[str]) -> list[MergeItem]: ...
    def draft_merge(self, items: list[MergeItem], tier: str, instruction: str) -> MergeDraft: ...
    def commit_merge(self, items: list[MergeItem], draft: MergeDraft) -> CommitResult: ...


class SourceRegistry:
    """Ordered registry of MergeableSource implementations."""

    def __init__(self) -> None:
        self._sources: dict[str, MergeableSource] = {}

    def register(self, source: MergeableSource) -> None:
        self._sources[source.name] = source

    def get(self, name: str) -> MergeableSource:
        if name not in self._sources:
            raise KeyError(f"no vector source registered: {name!r}")
        return self._sources[name]

    def all(self) -> list[MergeableSource]:
        return list(self._sources.values())

    def names(self) -> list[str]:
        return list(self._sources.keys())


import json as _json


class TaskSource:
    """Tasks merge — mechanical concatenation (default), LLM optional in PR-2."""

    name = "tasks"
    merge_mode = MergeMode.MECHANICAL

    def __init__(self, store: Any) -> None:
        self._store = store

    def index_stats(self) -> IndexStats:
        c = self._store._conn
        total = c.execute("SELECT COUNT(*) AS n FROM tasks").fetchone()["n"]
        embedded = c.execute(
            "SELECT COUNT(*) AS n FROM tasks WHERE vector IS NOT NULL AND vector != ''"
        ).fetchone()["n"]
        last = c.execute(
            "SELECT MAX(updated_at) AS t FROM tasks WHERE vector IS NOT NULL"
        ).fetchone()["t"]
        return IndexStats(
            name="tasks", source_type="first_class_table",
            doc_count=total, embedded_count=embedded,
            avg_age_days=None, last_indexed=(last or "")[:16] or None,
            embedding_model=None, level_breakdown={},
            scatter_url="/vector?source=tasks&tab=scatter",
            supports_merge=True, merge_mode=MergeMode.MECHANICAL,
        )

    def fetch_for_merge(self, ids: list[str]) -> list[MergeItem]:
        int_ids = [int(i) for i in ids if i.isdigit()]
        if not int_ids:
            return []
        ph = ",".join("?" * len(int_ids))
        rows = self._store._conn.execute(
            f"SELECT id, title, summary, context, tags FROM tasks "
            f"WHERE id IN ({ph})", int_ids,
        ).fetchall()
        return [
            MergeItem(
                id=str(r["id"]),
                label=r["title"] or f"task:{r['id']}",
                body=f"[{r['title']}] {r['summary'] or ''}\n{(r['context'] or '')[:500]}",
                raw={
                    "title": r["title"], "summary": r["summary"],
                    "context": r["context"], "tags": r["tags"],
                },
            ) for r in rows
        ]

    def draft_merge(self, items: list[MergeItem], tier: str, instruction: str) -> MergeDraft:
        title = "Merged: " + " + ".join((it.raw.get("title") or "")[:40] for it in items)
        parts = [f"## Task #{it.id}: {it.raw.get('title') or ''}\n"
                 f"{it.raw.get('summary') or ''}" for it in items]
        ctx_parts = [f"### From #{it.id}\n{it.raw['context']}"
                     for it in items if it.raw.get("context")]
        tags = ",".join(sorted({
            t.strip() for it in items for t in (it.raw.get("tags") or "").split(",")
            if t.strip()
        }))
        return MergeDraft(
            proposed_label=title[:200],
            proposed_body="\n\n".join(parts),
            proposed_raw={"context": "\n\n".join(ctx_parts), "tags": tags},
            tier_used="mechanical",
            tokens_used=None,
        )

    def commit_merge(self, items: list[MergeItem], draft: MergeDraft) -> CommitResult:
        c = self._store._conn
        cur = c.execute(
            "INSERT INTO tasks (title, summary, context, tags, vector, status) "
            "VALUES (?, ?, ?, ?, NULL, 'open')",
            (draft.proposed_label, draft.proposed_body,
             draft.proposed_raw.get("context", ""),
             draft.proposed_raw.get("tags", "")),
        )
        new_id = cur.lastrowid or 0
        closed: list[str] = []
        for it in items:
            c.execute(
                "UPDATE tasks SET status='closed', closed_at=datetime('now'), "
                "compact=?, updated_at=datetime('now') WHERE id = ?",
                (_json.dumps({"merged_into": new_id}), int(it.id)),
            )
            closed.append(it.id)
        audit_id = self._store.record_memory_audit(
            action="merge", namespace="tasks", doc_id=str(new_id),
            reason_json={"closed_ids": closed, "tier": draft.tier_used},
        )
        c.commit()
        return CommitResult(new_id=str(new_id), closed_ids=closed, audit_id=audit_id)
