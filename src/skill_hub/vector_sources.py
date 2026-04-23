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


def _get_provider():
    """Thin indirection so tests can monkeypatch."""
    from .llm import get_provider
    return get_provider()


_TIER_MAP = {"local": "tier_cheap", "mid": "tier_mid", "claude": "tier_strong"}


class NamespaceSource:
    """Memory namespace merge — LLM consolidates bodies, sums access_count."""

    source_type = "namespace"
    merge_mode = MergeMode.LLM

    def __init__(self, store: Any, namespace: str) -> None:
        self._store = store
        self.name = namespace
        self.namespace = namespace

    def index_stats(self) -> IndexStats:
        c = self._store._conn
        total = c.execute(
            "SELECT COUNT(*) AS n FROM vectors WHERE namespace = ?", (self.namespace,)
        ).fetchone()["n"]
        embedded = c.execute(
            "SELECT COUNT(*) AS n FROM vectors WHERE namespace = ? AND vector IS NOT NULL",
            (self.namespace,),
        ).fetchone()["n"]
        last = c.execute(
            "SELECT MAX(indexed_at) AS t FROM vectors WHERE namespace = ?",
            (self.namespace,),
        ).fetchone()["t"]
        level_rows = c.execute(
            "SELECT level, COUNT(*) AS n FROM vectors WHERE namespace = ? GROUP BY level",
            (self.namespace,),
        ).fetchall()
        breakdown = {r["level"] or "?": r["n"] for r in level_rows}
        try:
            cfg = c.execute(
                "SELECT embedding_model FROM vector_index_config WHERE name = ?",
                (self.namespace,),
            ).fetchone()
        except Exception:
            cfg = None
        return IndexStats(
            name=self.namespace, source_type="namespace",
            doc_count=total, embedded_count=embedded,
            avg_age_days=None, last_indexed=(last or "")[:16] or None,
            embedding_model=(cfg["embedding_model"] if cfg else None),
            level_breakdown=breakdown,
            scatter_url=f"/vector?source=namespaces&namespace={self.namespace}&tab=scatter",
            supports_merge=True, merge_mode=MergeMode.LLM,
        )

    def fetch_for_merge(self, ids: list[str]) -> list[MergeItem]:
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        rows = self._store._conn.execute(
            f"SELECT doc_id, metadata, level, access_count, last_accessed, vector "
            f"FROM vectors WHERE namespace = ? AND doc_id IN ({ph})",
            [self.namespace, *ids],
        ).fetchall()
        out: list[MergeItem] = []
        for r in rows:
            try:
                meta = _json.loads(r["metadata"] or "{}")
            except (TypeError, _json.JSONDecodeError):
                meta = {}
            body = meta.get("content") or meta.get("text") or meta.get("body") or ""
            out.append(MergeItem(
                id=r["doc_id"],
                label=str(meta.get("title") or meta.get("name") or r["doc_id"])[:60],
                body=f"[{r['level']}] {body[:800]}",
                raw={
                    "metadata": meta, "level": r["level"],
                    "access_count": r["access_count"] or 0,
                    "last_accessed": r["last_accessed"],
                },
            ))
        return out

    def draft_merge(self, items: list[MergeItem], tier: str, instruction: str) -> MergeDraft:
        items_text = "\n\n".join(
            f"### [{i+1}] {it.label} (id={it.id})\n{it.body}"
            for i, it in enumerate(items[:30])
        )
        prompt = (
            f"You are consolidating memory items from namespace '{self.namespace}'.\n\n"
            f"INSTRUCTION: {instruction or 'Merge these items into ONE cohesive note. Preserve all unique facts. Eliminate duplication.'}\n\n"
            f"ITEMS ({len(items)}):\n\n{items_text}\n\n"
            f"Output ONLY the merged body text. No preamble, no markdown headers, no commentary."
        )
        provider = _get_provider()
        resolved_tier = _TIER_MAP.get(tier, "tier_cheap")
        body = provider.complete(
            prompt, tier=resolved_tier, max_tokens=1500,
            temperature=0.3, timeout=120.0,
        )
        summed_access = sum(it.raw.get("access_count", 0) for it in items)
        merged_meta: dict = {}
        for it in items:
            for k, v in (it.raw.get("metadata") or {}).items():
                merged_meta.setdefault(k, v)
        latest = max((it.raw.get("last_accessed") or "" for it in items), default="")
        merged_meta["content"] = body
        label = f"merged({len(items)}) — {items[0].label[:40]}"
        merged_meta["title"] = label
        return MergeDraft(
            proposed_label=label,
            proposed_body=body,
            proposed_raw={
                "metadata": merged_meta,
                "level": items[0].raw.get("level", "L2"),
                "access_count": summed_access,
                "last_accessed": latest,
            },
            tier_used=tier,
            tokens_used=None,
        )

    def commit_merge(self, items: list[MergeItem], draft: MergeDraft) -> CommitResult:
        import uuid
        c = self._store._conn
        new_id = f"merged-{uuid.uuid4().hex[:12]}"
        raw = draft.proposed_raw
        # vector is NOT NULL in schema; insert empty-array placeholder, re-embed later.
        c.execute(
            "INSERT INTO vectors (doc_id, namespace, level, metadata, vector, "
            "access_count, indexed_at, last_accessed) "
            "VALUES (?, ?, ?, ?, '[]', ?, datetime('now'), ?)",
            (new_id, self.namespace, raw.get("level", "L2"),
             _json.dumps(raw.get("metadata", {})),
             raw.get("access_count", 0),
             raw.get("last_accessed") or None),
        )
        id_list = [it.id for it in items]
        ph = ",".join("?" * len(id_list))
        snapshot = [
            dict(row) for row in c.execute(
                f"SELECT doc_id, namespace, level, metadata, vector, access_count, "
                f"indexed_at, last_accessed FROM vectors "
                f"WHERE namespace = ? AND doc_id IN ({ph})",
                [self.namespace, *id_list],
            ).fetchall()
        ]
        c.execute(
            f"DELETE FROM vectors WHERE namespace = ? AND doc_id IN ({ph})",
            [self.namespace, *id_list],
        )
        audit_id = self._store.record_memory_audit(
            action="merge", namespace=self.namespace, doc_id=new_id,
            reason_json={
                "closed_ids": id_list,
                "tier": draft.tier_used,
                "rollback": {"restore_docs": snapshot},
            },
        )
        c.commit()
        return CommitResult(new_id=new_id, closed_ids=id_list, audit_id=audit_id)


import sqlite3 as _sqlite
from pathlib import Path as _Path


def _reject(name: str):
    raise NotImplementedError(f"merge for source '{name}' lands in PR-2")


class SkillSource:
    """Skills display-only in PR-1 (merge lands in PR-2 with LLM-rewrite)."""

    name = "skills"
    merge_mode = MergeMode.REJECTED

    def __init__(self, store: Any) -> None:
        self._store = store

    def index_stats(self) -> IndexStats:
        c = self._store._conn
        total = c.execute("SELECT COUNT(*) AS n FROM skills").fetchone()["n"]
        embedded = c.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"]
        return IndexStats(
            name="skills", source_type="first_class_table",
            doc_count=total, embedded_count=embedded,
            avg_age_days=None, last_indexed=None,
            embedding_model=None, level_breakdown={},
            scatter_url="/vector?source=skills&tab=scatter",
            supports_merge=False, merge_mode=MergeMode.REJECTED,
        )

    def fetch_for_merge(self, ids): return []
    def draft_merge(self, items, tier, instruction): _reject("skills")
    def commit_merge(self, items, draft): _reject("skills")


class TeachingSource:
    name = "teachings"
    merge_mode = MergeMode.REJECTED

    def __init__(self, store: Any) -> None:
        self._store = store

    def index_stats(self) -> IndexStats:
        c = self._store._conn
        total = c.execute("SELECT COUNT(*) AS n FROM teachings").fetchone()["n"]
        embedded = c.execute(
            "SELECT COUNT(*) AS n FROM teachings WHERE rule_vector IS NOT NULL"
        ).fetchone()["n"]
        return IndexStats(
            name="teachings", source_type="first_class_table",
            doc_count=total, embedded_count=embedded,
            avg_age_days=None, last_indexed=None,
            embedding_model=None, level_breakdown={},
            scatter_url="/vector?source=teachings&tab=scatter",
            supports_merge=False, merge_mode=MergeMode.REJECTED,
        )

    def fetch_for_merge(self, ids): return []
    def draft_merge(self, items, tier, instruction): _reject("teachings")
    def commit_merge(self, items, draft): _reject("teachings")


class VerdictSource:
    """Verdicts live in a separate sqlite file; permanently rejected for merge."""

    name = "verdicts"
    merge_mode = MergeMode.REJECTED

    def __init__(self, db_path: _Path | None) -> None:
        self._db_path = db_path

    def index_stats(self) -> IndexStats:
        doc_count = 0
        embedded = 0
        if self._db_path and self._db_path.exists():
            try:
                conn = _sqlite.connect(str(self._db_path))
                conn.row_factory = _sqlite.Row
                doc_count = conn.execute(
                    "SELECT COUNT(*) AS n FROM command_verdicts"
                ).fetchone()["n"]
                embedded = conn.execute(
                    "SELECT COUNT(*) AS n FROM command_verdicts "
                    "WHERE vector IS NOT NULL AND vector != ''"
                ).fetchone()["n"]
                conn.close()
            except _sqlite.Error:
                pass
        return IndexStats(
            name="verdicts", source_type="first_class_table",
            doc_count=doc_count, embedded_count=embedded,
            avg_age_days=None, last_indexed=None,
            embedding_model=None, level_breakdown={},
            scatter_url="/vector?source=verdicts&tab=scatter",
            supports_merge=False, merge_mode=MergeMode.REJECTED,
        )

    def fetch_for_merge(self, ids): return []
    def draft_merge(self, items, tier, instruction): _reject("verdicts")
    def commit_merge(self, items, draft): _reject("verdicts")


import sqlite3 as _sqlite
from pathlib import Path as _Path


def _reject(name: str):
    raise NotImplementedError(f"merge for source '{name}' lands in PR-2")


class SkillSource:
    """Skills display-only in PR-1 (merge lands in PR-2 with LLM-rewrite)."""

    name = "skills"
    merge_mode = MergeMode.REJECTED  # PR-1 only; flips to LLM in PR-2

    def __init__(self, store: Any) -> None:
        self._store = store

    def index_stats(self) -> IndexStats:
        c = self._store._conn
        total = c.execute("SELECT COUNT(*) AS n FROM skills").fetchone()["n"]
        embedded = c.execute(
            "SELECT COUNT(*) AS n FROM embeddings"
        ).fetchone()["n"]
        return IndexStats(
            name="skills", source_type="first_class_table",
            doc_count=total, embedded_count=embedded,
            avg_age_days=None, last_indexed=None,
            embedding_model=None, level_breakdown={},
            scatter_url="/vector?source=skills&tab=scatter",
            supports_merge=False, merge_mode=MergeMode.REJECTED,
        )

    def fetch_for_merge(self, ids): return []
    def draft_merge(self, items, tier, instruction): _reject("skills")
    def commit_merge(self, items, draft): _reject("skills")


class TeachingSource:
    name = "teachings"
    merge_mode = MergeMode.REJECTED  # flips to LLM in PR-2

    def __init__(self, store: Any) -> None:
        self._store = store

    def index_stats(self) -> IndexStats:
        c = self._store._conn
        total = c.execute("SELECT COUNT(*) AS n FROM teachings").fetchone()["n"]
        embedded = c.execute(
            "SELECT COUNT(*) AS n FROM teachings WHERE rule_vector IS NOT NULL"
        ).fetchone()["n"]
        return IndexStats(
            name="teachings", source_type="first_class_table",
            doc_count=total, embedded_count=embedded,
            avg_age_days=None, last_indexed=None,
            embedding_model=None, level_breakdown={},
            scatter_url="/vector?source=teachings&tab=scatter",
            supports_merge=False, merge_mode=MergeMode.REJECTED,
        )

    def fetch_for_merge(self, ids): return []
    def draft_merge(self, items, tier, instruction): _reject("teachings")
    def commit_merge(self, items, draft): _reject("teachings")


class VerdictSource:
    """Verdicts live in a separate sqlite file; permanently rejected for merge."""

    name = "verdicts"
    merge_mode = MergeMode.REJECTED

    def __init__(self, db_path: _Path | None) -> None:
        self._db_path = db_path

    def index_stats(self) -> IndexStats:
        doc_count = 0
        embedded = 0
        if self._db_path and self._db_path.exists():
            try:
                conn = _sqlite.connect(str(self._db_path))
                conn.row_factory = _sqlite.Row
                doc_count = conn.execute(
                    "SELECT COUNT(*) AS n FROM command_verdicts"
                ).fetchone()["n"]
                embedded = conn.execute(
                    "SELECT COUNT(*) AS n FROM command_verdicts "
                    "WHERE vector IS NOT NULL AND vector != ''"
                ).fetchone()["n"]
                conn.close()
            except _sqlite.Error:
                pass
        return IndexStats(
            name="verdicts", source_type="first_class_table",
            doc_count=doc_count, embedded_count=embedded,
            avg_age_days=None, last_indexed=None,
            embedding_model=None, level_breakdown={},
            scatter_url="/vector?source=verdicts&tab=scatter",
            supports_merge=False, merge_mode=MergeMode.REJECTED,
        )

    def fetch_for_merge(self, ids): return []
    def draft_merge(self, items, tier, instruction): _reject("verdicts")
    def commit_merge(self, items, draft): _reject("verdicts")
