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
    # In-session directive for LLM-mode sources: a structured prompt the active
    # Claude Code agent can dispatch to a Haiku/Sonnet subagent via the Agent
    # tool (no ANTHROPIC_API_KEY required). When populated, the UI surfaces a
    # "Refine with Haiku" button; the mechanical body stays as fallback.
    directive: str | None = None


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
    """Skills merge — mechanical concatenation + Haiku directive (LLM mode).

    Draft path is offline-safe: it produces a readable merged body from the
    source rows (deduping content, uniting `plugin`, averaging feedback_score)
    plus a `directive` string. The active Claude Code agent can optionally
    dispatch the directive to a Haiku subagent to get a rewrite — no API key
    is called server-side.
    """

    name = "skills"
    merge_mode = MergeMode.LLM

    def __init__(self, store: Any) -> None:
        self._store = store

    def index_stats(self) -> IndexStats:
        c = self._store._conn
        total = c.execute("SELECT COUNT(*) AS n FROM skills").fetchone()["n"]
        embedded = c.execute("SELECT COUNT(*) AS n FROM embeddings").fetchone()["n"]
        last = c.execute("SELECT MAX(indexed_at) AS t FROM skills").fetchone()["t"]
        model_row = c.execute(
            "SELECT model, COUNT(*) AS n FROM embeddings GROUP BY model "
            "ORDER BY n DESC LIMIT 1"
        ).fetchone()
        return IndexStats(
            name="skills", source_type="first_class_table",
            doc_count=total, embedded_count=embedded,
            avg_age_days=None, last_indexed=(last or "")[:16] or None,
            embedding_model=(model_row["model"] if model_row else None),
            level_breakdown={},
            scatter_url="/vector?source=skills&tab=scatter",
            supports_merge=True, merge_mode=MergeMode.LLM,
        )

    def fetch_for_merge(self, ids: list[str]) -> list[MergeItem]:
        if not ids:
            return []
        ph = ",".join("?" * len(ids))
        rows = self._store._conn.execute(
            f"SELECT id, name, description, content, file_path, plugin, target, "
            f"feedback_score, content_hash, indexed_at FROM skills "
            f"WHERE id IN ({ph})",
            list(ids),
        ).fetchall()
        return [
            MergeItem(
                id=r["id"],
                label=(r["name"] or r["id"])[:80],
                body=f"# {r['name']}\n{r['description'] or ''}\n\n{(r['content'] or '')[:800]}",
                raw={
                    "name": r["name"], "description": r["description"],
                    "content": r["content"], "file_path": r["file_path"],
                    "plugin": r["plugin"], "target": r["target"],
                    "feedback_score": r["feedback_score"],
                    "content_hash": r["content_hash"],
                    "indexed_at": r["indexed_at"],
                },
            ) for r in rows
        ]

    def draft_merge(self, items: list[MergeItem], tier: str, instruction: str) -> MergeDraft:
        # Mechanical fallback body: deduped concat of descriptions + contents.
        names = [it.raw.get("name") or it.id for it in items]
        label = f"merged: {' + '.join(n[:30] for n in names[:3])}"[:120]
        descs = [(it.raw.get("description") or "").strip() for it in items]
        descs_clean = list(dict.fromkeys(d for d in descs if d))
        contents = [(it.raw.get("content") or "").strip() for it in items]
        sections = [
            f"## From: {it.raw.get('name') or it.id}\n{c}"
            for it, c in zip(items, contents, strict=False) if c
        ]
        body = (
            ("\n\n".join(descs_clean) + "\n\n---\n\n" if descs_clean else "")
            + "\n\n".join(sections)
        )
        plugins = sorted({(it.raw.get("plugin") or "") for it in items if it.raw.get("plugin")})
        targets = sorted({(it.raw.get("target") or "claude") for it in items})
        scores = [float(it.raw.get("feedback_score") or 1.0) for it in items]
        avg_score = sum(scores) / len(scores) if scores else 1.0

        directive = (
            f"Dispatch an Agent call (subagent_type='general-purpose', "
            f"model='claude-haiku-4-5') with this prompt:\n\n"
            f"---\n"
            f"You are consolidating {len(items)} skill entries from the skill-hub "
            f"into a single cohesive skill.\n"
            f"USER INSTRUCTION: {instruction or '(none — merge faithfully, preserve all unique facts)'}\n\n"
            f"SOURCE SKILLS:\n\n"
            + "\n\n".join(
                f"### [{i+1}] {it.raw.get('name')}\nDescription: {it.raw.get('description') or ''}"
                f"\n\n{(it.raw.get('content') or '')[:2000]}"
                for i, it in enumerate(items)
            )
            + "\n\n---\n"
            f"Output ONLY the merged skill body in markdown. No preamble, no commentary."
            f" After receiving the result, POST it to /vector/merge/commit with source='skills'."
        )

        proposed_raw = {
            "name": label,
            "description": (descs_clean[0] if descs_clean else "")[:500],
            "content": body,
            "plugin": ",".join(plugins) if plugins else None,
            "target": targets[0] if len(targets) == 1 else "claude",
            "feedback_score": avg_score,
        }
        return MergeDraft(
            proposed_label=label,
            proposed_body=body,
            proposed_raw=proposed_raw,
            tier_used=tier or "mechanical",
            tokens_used=None,
            directive=directive,
        )

    def commit_merge(self, items: list[MergeItem], draft: MergeDraft) -> CommitResult:
        import hashlib
        import uuid
        c = self._store._conn
        new_id = f"merged-skill-{uuid.uuid4().hex[:12]}"
        raw = draft.proposed_raw
        content = draft.proposed_body or raw.get("content") or ""
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]

        # Snapshot source rows BEFORE deletion, for rollback payload.
        id_list = [it.id for it in items]
        ph = ",".join("?" * len(id_list))
        snapshot = [
            dict(row) for row in c.execute(
                f"SELECT id, name, description, content, file_path, plugin, target, "
                f"feedback_score, content_hash, indexed_at FROM skills "
                f"WHERE id IN ({ph})", id_list,
            ).fetchall()
        ]
        emb_snapshot = [
            dict(row) for row in c.execute(
                f"SELECT skill_id, model, vector, norm FROM embeddings "
                f"WHERE skill_id IN ({ph})", id_list,
            ).fetchall()
        ]

        c.execute(
            "INSERT INTO skills (id, name, description, content, file_path, plugin, "
            "target, feedback_score, content_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (new_id, draft.proposed_label or raw.get("name") or new_id,
             raw.get("description") or "", content,
             raw.get("file_path"), raw.get("plugin"),
             raw.get("target") or "claude",
             float(raw.get("feedback_score") or 1.0),
             content_hash),
        )
        # SQLite FK enforcement is off by default — delete embeddings explicitly
        # (schema has ON DELETE CASCADE but PRAGMA foreign_keys isn't set on
        # this connection). New row has no embedding yet; re-index populates it.
        c.execute(f"DELETE FROM embeddings WHERE skill_id IN ({ph})", id_list)
        c.execute(f"DELETE FROM skills WHERE id IN ({ph})", id_list)

        audit_id = self._store.record_memory_audit(
            action="merge", namespace="skills", doc_id=new_id,
            reason_json={
                "closed_ids": id_list,
                "tier": draft.tier_used,
                "rollback": {
                    "restore_docs": snapshot,
                    "restore_embeddings": emb_snapshot,
                },
            },
        )
        c.commit()
        return CommitResult(new_id=new_id, closed_ids=id_list, audit_id=audit_id)


class TeachingSource:
    """Teachings merge — mechanical concatenation + Haiku directive (LLM mode).

    Rules are short directives ("when X, do Y"); merging unions rule text,
    keeps the most common `action`/`target_type`/`target_id`, and averages
    weight. A Haiku directive is surfaced for optional refinement.
    """

    name = "teachings"
    merge_mode = MergeMode.LLM

    def __init__(self, store: Any) -> None:
        self._store = store

    def index_stats(self) -> IndexStats:
        c = self._store._conn
        total = c.execute("SELECT COUNT(*) AS n FROM teachings").fetchone()["n"]
        embedded = c.execute(
            "SELECT COUNT(*) AS n FROM teachings WHERE rule_vector IS NOT NULL "
            "AND rule_vector != ''"
        ).fetchone()["n"]
        last = c.execute("SELECT MAX(created_at) AS t FROM teachings").fetchone()["t"]
        return IndexStats(
            name="teachings", source_type="first_class_table",
            doc_count=total, embedded_count=embedded,
            avg_age_days=None, last_indexed=(last or "")[:16] or None,
            embedding_model=None, level_breakdown={},
            scatter_url="/vector?source=teachings&tab=scatter",
            supports_merge=True, merge_mode=MergeMode.LLM,
        )

    def fetch_for_merge(self, ids: list[str]) -> list[MergeItem]:
        int_ids = [int(i) for i in ids if str(i).isdigit()]
        if not int_ids:
            return []
        ph = ",".join("?" * len(int_ids))
        rows = self._store._conn.execute(
            f"SELECT id, rule, action, target_type, target_id, weight, created_at "
            f"FROM teachings WHERE id IN ({ph})", int_ids,
        ).fetchall()
        return [
            MergeItem(
                id=str(r["id"]),
                label=(r["rule"] or f"teaching:{r['id']}")[:80],
                body=f"IF: {r['rule']}\nTHEN: {r['action']} "
                     f"({r['target_type']}={r['target_id']}) "
                     f"[weight={r['weight']}]",
                raw={
                    "rule": r["rule"], "action": r["action"],
                    "target_type": r["target_type"], "target_id": r["target_id"],
                    "weight": r["weight"], "created_at": r["created_at"],
                },
            ) for r in rows
        ]

    def draft_merge(self, items: list[MergeItem], tier: str, instruction: str) -> MergeDraft:
        from collections import Counter
        rules = [(it.raw.get("rule") or "").strip() for it in items]
        rules_unique = list(dict.fromkeys(r for r in rules if r))
        merged_rule = " OR ".join(f"({r})" for r in rules_unique) if rules_unique else ""

        action_counts = Counter((it.raw.get("action") or "") for it in items)
        top_action = action_counts.most_common(1)[0][0] if action_counts else ""
        ttype_counts = Counter((it.raw.get("target_type") or "plugin") for it in items)
        top_ttype = ttype_counts.most_common(1)[0][0] if ttype_counts else "plugin"
        tid_counts = Counter((it.raw.get("target_id") or "") for it in items)
        top_tid = tid_counts.most_common(1)[0][0] if tid_counts else ""

        weights = [float(it.raw.get("weight") or 1.0) for it in items]
        avg_weight = sum(weights) / len(weights) if weights else 1.0

        label = f"merged rule: {(rules_unique[0] if rules_unique else 'teaching')[:60]}"
        body = (
            f"RULE: {merged_rule}\n"
            f"ACTION: {top_action}\n"
            f"TARGET: {top_ttype}={top_tid}\n"
            f"WEIGHT: {avg_weight:.2f}"
        )

        directive = (
            f"Dispatch an Agent call (subagent_type='general-purpose', "
            f"model='claude-haiku-4-5') with this prompt:\n\n"
            f"---\n"
            f"You are consolidating {len(items)} hub teaching rules into ONE.\n"
            f"USER INSTRUCTION: {instruction or '(none)'}\n\n"
            f"RULES:\n"
            + "\n".join(f"- IF: {it.raw.get('rule')} THEN: {it.raw.get('action')}"
                       f" ({it.raw.get('target_type')}={it.raw.get('target_id')})"
                       for it in items)
            + "\n\n---\n"
            f"Output a JSON object: {{\"rule\": \"...\", \"action\": \"...\", "
            f"\"target_type\": \"plugin|skill\", \"target_id\": \"...\"}}. "
            f"POST to /vector/merge/commit with source='teachings'."
        )

        return MergeDraft(
            proposed_label=label,
            proposed_body=body,
            proposed_raw={
                "rule": merged_rule,
                "action": top_action,
                "target_type": top_ttype,
                "target_id": top_tid,
                "weight": avg_weight,
            },
            tier_used=tier or "mechanical",
            tokens_used=None,
            directive=directive,
        )

    def commit_merge(self, items: list[MergeItem], draft: MergeDraft) -> CommitResult:
        c = self._store._conn
        raw = draft.proposed_raw
        # Snapshot for rollback (include vector JSON so restore is lossless).
        id_list = [it.id for it in items]
        int_ids = [int(i) for i in id_list]
        ph = ",".join("?" * len(int_ids))
        snapshot = [
            dict(row) for row in c.execute(
                f"SELECT id, rule, rule_vector, action, target_type, target_id, "
                f"weight, created_at FROM teachings WHERE id IN ({ph})",
                int_ids,
            ).fetchall()
        ]

        cur = c.execute(
            "INSERT INTO teachings (rule, rule_vector, action, target_type, "
            "target_id, weight) VALUES (?, '[]', ?, ?, ?, ?)",
            (raw.get("rule") or draft.proposed_body,
             raw.get("action") or "",
             raw.get("target_type") or "plugin",
             raw.get("target_id") or "",
             float(raw.get("weight") or 1.0)),
        )
        new_id = str(cur.lastrowid or 0)
        c.execute(f"DELETE FROM teachings WHERE id IN ({ph})", int_ids)

        audit_id = self._store.record_memory_audit(
            action="merge", namespace="teachings", doc_id=new_id,
            reason_json={
                "closed_ids": id_list,
                "tier": draft.tier_used,
                "rollback": {"restore_docs": snapshot},
            },
        )
        c.commit()
        return CommitResult(new_id=new_id, closed_ids=id_list, audit_id=audit_id)


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

