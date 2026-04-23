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
