"""Pre-conversation 4-tier enrichment pipeline.

Tiers:
    L1 — Classify  (intent tags, domain, complexity)
    L2 — Retrieve + dedup task  (embed → search → create/update task)
    L3 — Synthesize + curate  (write synthesis, optionally switch profile)
    L4 — Rewrite  (optional, only when complexity >= 'medium')

Each tier has a configurable timeout; on timeout/error it degrades gracefully
to the next fallback.
"""
from __future__ import annotations

import concurrent.futures
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

_log = logging.getLogger(__name__)


@dataclass
class TierResult:
    ran: bool = False
    duration_ms: int | None = None
    fallback_used: bool = False
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    session_id: str = ""
    task_id: int | None = None
    tier1: TierResult = field(default_factory=TierResult)
    tier2: TierResult = field(default_factory=TierResult)
    tier3: TierResult = field(default_factory=TierResult)
    tier4: TierResult = field(default_factory=TierResult)
    synthesis: str = ""
    enriched_prompt: str | None = None

    @property
    def fallbacks(self) -> list[str]:
        out = []
        for name, t in [
            ("tier1", self.tier1),
            ("tier2", self.tier2),
            ("tier3", self.tier3),
            ("tier4", self.tier4),
        ]:
            if t.fallback_used:
                out.append(name)
        return out


def _run_with_timeout(fn, timeout_ms: int, *args, **kwargs):
    """Run fn(*args, **kwargs) with a timeout (ms).

    Returns (result, elapsed_ms, timed_out).
    On timeout or unexpected error, result is None and fallback_used should be set.
    """
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(fn, *args, **kwargs)
        try:
            result = fut.result(timeout=timeout_ms / 1000.0)
            elapsed = int((time.time() - start) * 1000)
            return result, elapsed, False
        except concurrent.futures.TimeoutError:
            elapsed = int((time.time() - start) * 1000)
            return None, elapsed, True
        except Exception as exc:
            elapsed = int((time.time() - start) * 1000)
            _log.debug("pipeline tier error: %s", exc)
            return None, elapsed, False  # treat as fallback, not timeout


def _tier1_classify(message: str) -> dict:
    """L1: Classify intent using haiku JSON or keyword extraction."""
    from . import config as _cfg

    backend = str(_cfg.get("classify_backend") or "haiku_json")

    if backend == "yake_keywords":
        return _yake_classify(message)

    # Default: haiku JSON-mode
    from .llm import get_provider

    prompt = f"""Classify this user message for a coding assistant session.

Message: {message[:800]}

Output ONLY a JSON object with these fields:
{{
  "intent_tags": ["<tag1>", "<tag2>"],   // e.g. ["coding:python", "debug", "refactor"]
  "domain_keywords": ["<kw1>", "<kw2>"], // key technical terms
  "complexity": "low|medium|high",
  "scope": "narrow|medium|broad"
}}"""
    try:
        import re as _re

        raw = get_provider().complete(
            prompt,
            tier="tier_mid",
            max_tokens=150,
            temperature=0.0,
            timeout=2.0,
        )
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        if m:
            result = json.loads(m.group())
            if "intent_tags" in result:
                return result
    except Exception:
        pass
    return _yake_classify(message)


def _yake_classify(message: str) -> dict:
    """Deterministic keyword extraction fallback (no LLM)."""
    from collections import Counter

    words = [w.lower().strip(".,!?;:") for w in message.split()]
    words = [w for w in words if len(w) > 3]
    freq = Counter(words)
    keywords = [w for w, _ in freq.most_common(8)]
    # Guess complexity from message length
    if len(message) < 100:
        complexity = "low"
    elif len(message) < 400:
        complexity = "medium"
    else:
        complexity = "high"
    return {
        "intent_tags": keywords[:4],
        "domain_keywords": keywords,
        "complexity": complexity,
        "scope": "medium",
    }


def _tier2_retrieve(message: str, session_id: str, store) -> dict:
    """L2: Embed message, search tasks, create or update the session task."""
    from . import config as _cfg
    from .embeddings import embed, embed_available

    threshold = float(_cfg.get("task_similarity_threshold") or 0.75)
    min_chars = int(_cfg.get("task_auto_create_min_chars") or 0)

    task_id: int | None = None
    top_similarity: float | None = None
    retrieval: list[dict] = []

    if len(message) < min_chars:
        return {"task_id": None, "top_similarity": None, "retrieval": []}

    # Try to embed — fall back to text search if unavailable
    vec: list[float] = []
    if embed_available():
        try:
            vec = embed(message)
        except Exception as exc:
            _log.debug("L2 embed failed: %s", exc)

    # Search for similar tasks (dedup)
    existing_task = store.get_open_task_for_session(session_id)
    if existing_task:
        task_id = dict(existing_task)["id"]
        # Update the existing task with the new message context
        store.update_task(task_id, summary=message[:1000])
    else:
        # Search for semantically similar recent tasks
        if vec:
            try:
                similar = store.search_tasks(
                    query_vector=vec, status="open", top_k=3
                )
                if similar:
                    best = dict(similar[0])
                    top_similarity = best.get("score") or best.get("similarity")
                    if top_similarity and top_similarity >= threshold:
                        # Update existing similar task
                        task_id = best["id"]
                        store.update_task(task_id, summary=message[:500])
            except Exception as exc:
                _log.debug("L2 search_tasks failed: %s", exc)

        if task_id is None:
            # Create new task
            title = message[:60].rstrip() + ("…" if len(message) > 60 else "")
            task_id = store.save_task(
                title=title,
                summary=message[:500],
                vector=vec,
                session_id=session_id,
            )

    # Retrieve context (teachings, closed tasks, skills)
    if vec:
        try:
            # Try search_context if it exists
            if hasattr(store, "search_context"):
                ctx = store.search_context(vec, top_k=8)
                retrieval = [dict(r) for r in (ctx or [])]
            else:
                closed = store.search_tasks(
                    query_vector=vec, status="closed", top_k=5
                )
                retrieval = [dict(r) for r in (closed or [])]
        except Exception as exc:
            _log.debug("L2 retrieval failed: %s", exc)
    else:
        # FTS5 fallback
        try:
            results = store.search_text(message, tables=["tasks"], top_k=5)
            retrieval = [dict(r) for r in (results or [])]
        except Exception as exc:
            _log.debug("L2 FTS fallback failed: %s", exc)

    return {
        "task_id": task_id,
        "top_similarity": top_similarity,
        "retrieval": retrieval,
    }


def _tier3_synthesize(
    message: str, intent: dict, retrieval: list[dict]
) -> str:
    """L3: Write a concise synthesis of retrieved context."""
    from . import config as _cfg

    max_sentences = int(_cfg.get("pipeline_synthesis_max_sentences") or 5)

    if not retrieval:
        return ""

    # Build context string from retrieval
    ctx_parts = []
    for item in retrieval[:5]:
        title = item.get("title", "")
        summary = item.get("summary", item.get("compact", ""))[:200]
        if title or summary:
            ctx_parts.append(
                f"- {title}: {summary}" if title else f"- {summary}"
            )

    if not ctx_parts:
        return ""

    context_str = "\n".join(ctx_parts)
    tags = ", ".join(intent.get("intent_tags", [])[:4])

    try:
        from .llm import get_provider

        prompt = f"""You are a context synthesizer for a coding assistant.

User message intent: {tags or '(unknown)'}
User message: {message[:400]}

Relevant prior work and context:
{context_str}

Write {max_sentences} sentences synthesizing the most relevant context. Be concise and specific.
Output ONLY the synthesis text, no labels or headers."""
        return get_provider().complete(
            prompt,
            tier="tier_mid",
            max_tokens=300,
            temperature=0.1,
            timeout=5.0,
        ).strip()
    except Exception:
        # Fallback: simple concat of top-3 items
        return " | ".join(ctx_parts[:3])[:500]


def _tier4_rewrite(
    message: str, synthesis: str, intent: dict
) -> str | None:
    """L4: Optionally rewrite the prompt with synthesis context.

    Only for complex messages (controlled by pipeline_tier4_min_complexity).
    """
    from . import config as _cfg

    min_complexity = str(_cfg.get("pipeline_tier4_min_complexity") or "medium")
    complexity = intent.get("complexity", "low")

    order = ["low", "medium", "high"]
    if order.index(complexity) < order.index(min_complexity):
        return None  # skip rewrite for simple messages

    if not synthesis:
        return None

    try:
        from .llm import get_provider

        prompt = f"""You are a prompt optimizer for a coding assistant.

Prior context synthesis:
{synthesis}

Original user message:
{message}

Rewrite the user message incorporating the context synthesis to make it more specific and actionable.
Do not change the intent. Output ONLY the rewritten prompt, no labels or explanation."""
        return get_provider().complete(
            prompt,
            tier="tier_smart",
            max_tokens=400,
            temperature=0.1,
            timeout=8.0,
        ).strip()
    except Exception:
        return None


class Pipeline:
    """4-tier pre-conversation enrichment pipeline."""

    def run(
        self,
        message: str,
        session_id: str,
        store=None,
    ) -> PipelineResult:
        from . import config as _cfg
        from .store import SkillStore

        if store is None:
            store = SkillStore()

        result = PipelineResult(session_id=session_id)

        t1_ms = int(_cfg.get("pipeline_tier1_timeout_ms") or 500)
        t2_ms = int(_cfg.get("pipeline_tier2_timeout_ms") or 400)
        t3_ms = int(_cfg.get("pipeline_tier3_timeout_ms") or 1200)
        t4_ms = int(_cfg.get("pipeline_tier4_timeout_ms") or 1500)

        # --- Tier 1: Classify ---
        intent_data, ms1, _timed_out = _run_with_timeout(
            _tier1_classify, t1_ms, message
        )
        if intent_data:
            result.tier1 = TierResult(ran=True, duration_ms=ms1, data=intent_data)
        else:
            fallback_intent = _yake_classify(message)
            result.tier1 = TierResult(
                ran=True,
                duration_ms=ms1,
                fallback_used=True,
                data=fallback_intent,
            )
            intent_data = result.tier1.data

        # --- Tier 2: Retrieve + dedup ---
        t2_data, ms2, _timed_out = _run_with_timeout(
            _tier2_retrieve, t2_ms, message, session_id, store
        )
        if t2_data:
            result.tier2 = TierResult(ran=True, duration_ms=ms2, data=t2_data)
            result.task_id = t2_data.get("task_id")
            retrieval = t2_data.get("retrieval", [])
        else:
            result.tier2 = TierResult(
                ran=True, duration_ms=ms2, fallback_used=True, data={}
            )
            retrieval = []

        # --- Tier 3: Synthesize ---
        synthesis, ms3, _ = _run_with_timeout(
            _tier3_synthesize, t3_ms, message, intent_data, retrieval
        )
        if synthesis:
            result.tier3 = TierResult(
                ran=True, duration_ms=ms3, data={"synthesis": synthesis}
            )
            result.synthesis = synthesis
        else:
            result.tier3 = TierResult(
                ran=True, duration_ms=ms3, fallback_used=True, data={}
            )
            # Fallback: concat top-3 raw retrieval titles/summaries
            if retrieval:
                result.synthesis = " | ".join(
                    (r.get("title") or r.get("summary", ""))[:100]
                    for r in retrieval[:3]
                )

        # --- Tier 4: Rewrite (optional) ---
        enriched, ms4, _ = _run_with_timeout(
            _tier4_rewrite, t4_ms, message, result.synthesis, intent_data
        )
        if enriched:
            result.tier4 = TierResult(
                ran=True, duration_ms=ms4, data={"enriched": enriched}
            )
            result.enriched_prompt = enriched
        else:
            result.tier4 = TierResult(
                ran=False, duration_ms=ms4, fallback_used=True, data={}
            )

        # --- Record telemetry ---
        try:
            store.record_pipeline_run(
                session_id=session_id,
                task_id=result.task_id,
                tier_ms={
                    "tier1": result.tier1.duration_ms,
                    "tier2": result.tier2.duration_ms,
                    "tier3": result.tier3.duration_ms,
                    "tier4": result.tier4.duration_ms,
                },
                fallbacks=result.fallbacks,
                top_similarity=(
                    result.tier2.data.get("top_similarity")
                    if result.tier2.data
                    else None
                ),
                token_cost_usd=0.0,  # TODO: track actual cost
            )
        except Exception as exc:
            _log.debug("pipeline telemetry failed: %s", exc)

        return result
