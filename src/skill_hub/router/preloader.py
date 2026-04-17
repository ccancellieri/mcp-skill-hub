"""Preloader — fetches relevant skill snippets and enriches thin prompts.

Two responsibilities:
  1. Skill preloading: given domain_hints, search the store for the top-N
     skills and return their names + description for inclusion in systemMessage.
     Results are deduplicated by name to avoid repeats from multi-source indexing.
  2. Thin-prompt enrichment: when the user sends a very short / context-free
     message (e.g. "fix it", "done"), reconstruct intent from session history
     and prepend it as userMessage context.

Enrichment priority chain (first non-empty source wins):
  A. Session context summary — LLM-compacted digest of the current session
     (written by the Stop hook via store.save_session_context). Most accurate
     because it was generated from actual conversation content.
  B. Semantic task search — embed the thin prompt, search open tasks by
     cosine similarity. Finds the most *relevant* task even if it's not
     the most recently updated one.
  C. Most recent open task — cheapest fallback, no embedding required.

Token economics:
  Cost:   +~50-150 chars (~15-40 tokens) per enriched prompt
  Saving: avoids one clarification round-trip (~500-1000 tokens)
  Net gain: ~460-960 tokens saved per thin-prompt encounter.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Skill preloading
# ---------------------------------------------------------------------------

def load_skills(
    domain_hints: list[str],
    cfg: dict[str, Any],
    top_k: int = 3,
) -> tuple[list[str], list[str]]:
    """Return (skill_names, plugin_names) relevant to *domain_hints*.

    Skill names are deduplicated by name (preserving best-ranked order).
    Gracefully returns empty lists if Ollama or the store is unavailable.
    """
    if not domain_hints:
        return [], []

    try:
        from ..embeddings import embed, embed_available, EMBED_MODEL
        from ..store import SkillStore
    except ImportError:
        return [], []

    if not embed_available():
        return [], []

    query = " ".join(domain_hints)
    try:
        vec = embed(query)
    except Exception:
        return [], []

    store = SkillStore()
    try:
        # Fetch more than needed so deduplication still yields top_k unique names
        results = store.search(vec, top_k=top_k * 3)
        seen: set[str] = set()
        skill_names: list[str] = []
        for r in results:
            name = r.get("name", "")
            if name and name not in seen:
                seen.add(name)
                skill_names.append(name)
            if len(skill_names) >= top_k:
                break

        plugin_suggestions = store.suggest_plugins(vec)
        plugin_names = [
            p["short_name"] for p in (plugin_suggestions or [])[:2]
            if not p.get("is_enabled", True)  # only suggest disabled ones
        ]
    except Exception:
        skill_names, plugin_names = [], []
    finally:
        store.close()

    return skill_names, plugin_names


# ---------------------------------------------------------------------------
# Thin-prompt enrichment
# ---------------------------------------------------------------------------

import re as _re

_THIN_PROMPT_MAX_LEN: int = 60  # prompts shorter than this are candidates

# Patterns that indicate a self-contained, explicit prompt — enrichment adds noise
_EXPLICIT_TARGET = _re.compile(
    r"\w+\.\w{1,6}"       # file.ext reference
    r"|https?://"          # URL
    r"|#\d+"               # issue/PR number
    r"|line\s+\d+"         # line number reference
    r"|in\s+\w+\.\w{1,6}" # "in utils.py"
)


def _is_thin(prompt: str) -> bool:
    stripped = prompt.strip()
    if len(stripped) >= _THIN_PROMPT_MAX_LEN or "\n" in stripped:
        return False
    # Skip enrichment when the prompt already names a concrete target
    if _EXPLICIT_TARGET.search(stripped):
        return False
    return True


def enrich_thin_prompt(
    prompt: str,
    session_id: str,
    cfg: dict[str, Any],
) -> str | None:
    """Return an enriched userMessage for thin prompts, or None if not needed.

    Tries three context sources in priority order (A → B → C).
    Returns None if no useful context is found — never invents context.
    """
    if not cfg.get("router_enrich_thin_prompts", True):
        return None
    if not _is_thin(prompt):
        return None

    context_snippet = _gather_context(prompt.strip(), session_id, cfg)
    if not context_snippet:
        return None

    return (
        f"[Session context: {context_snippet}]\n\n"
        f"User message: {prompt.strip()}"
    )


def _gather_context(
    prompt: str,
    session_id: str,
    cfg: dict[str, Any],
    max_chars: int = 300,
) -> str:
    """Return a compact context string from the best available source."""
    try:
        from ..store import SkillStore
    except ImportError:
        return ""

    store = SkillStore()
    try:
        # ── Source A: session context summary ────────────────────────────────
        # Written by the Stop hook after each session turn. Contains an
        # LLM-generated compact digest of what happened in this session.
        if session_id:
            try:
                ctx = store.get_session_context(session_id)
                summary = (ctx.get("context_summary") or "").strip()
                if summary:
                    # Prefer last sentence / clause (most recent state)
                    # and cap to max_chars
                    return summary[-max_chars:].lstrip(", \n")
            except Exception:
                pass

        # ── Source B: semantic task search ────────────────────────────────────
        # Embed the thin prompt and find the most relevant open task by
        # cosine similarity. Works even when session context hasn't been
        # written yet (e.g. first turn after a compaction).
        try:
            from ..embeddings import embed, embed_available, EMBED_MODEL
            if embed_available():
                vec = embed(prompt)
                # search() searches skills table; task search needs direct query
                tasks = _search_tasks_by_vector(store, vec, top_k=1)
                if tasks:
                    t = tasks[0]
                    parts = [p for p in [t.get("title"), t.get("summary", "")[:200]] if p]
                    return " — ".join(parts)[:max_chars]
        except Exception:
            pass

        # ── Source C: most recent open task (no embedding) ───────────────────
        try:
            all_open = store.list_tasks("open")
            if all_open:
                t = all_open[0]
                parts = [p for p in [t.get("title"), t.get("summary", "")[:150]] if p]
                return " — ".join(parts)[:max_chars]
        except Exception:
            pass

    finally:
        store.close()

    return ""


def _search_tasks_by_vector(store: Any, vec: list[float], top_k: int = 1) -> list[dict]:
    """Search open tasks by embedding similarity using the store's DB directly."""
    import json
    import math

    try:
        rows = store._conn.execute(
            "SELECT id, title, summary, vector FROM tasks WHERE status = 'open' AND vector IS NOT NULL"
        ).fetchall()
    except Exception:
        return []

    if not rows:
        return []

    qnorm = math.sqrt(sum(x * x for x in vec))
    if qnorm == 0.0:
        return []

    scored: list[tuple[float, dict]] = []
    for row in rows:
        try:
            tvec = json.loads(row["vector"])
            tnorm = math.sqrt(sum(x * x for x in tvec))
            if tnorm == 0.0:
                continue
            dot = sum(a * b for a, b in zip(vec, tvec))
            sim = dot / (qnorm * tnorm)
            scored.append((sim, {"id": row["id"], "title": row["title"], "summary": row["summary"]}))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:top_k]]


# ---------------------------------------------------------------------------
# Compact advisor
# ---------------------------------------------------------------------------

def should_compact(
    session_id: str,
    cfg: dict[str, Any],
) -> tuple[bool, str]:
    """Estimate context pressure and advise /compact if warranted.

    Uses two signals:
      1. Per-session message counter (temp file, incremented by the router hook)
      2. store.get_session_context(session_id).message_count if available

    The store value is authoritative when present (it's written by the Stop hook
    after every session turn and reflects the real message count).
    """
    import os
    import tempfile

    threshold: float = float(cfg.get("router_compact_threshold", 0.70))

    count = 0

    # Prefer the store's authoritative count
    if session_id:
        try:
            from ..store import SkillStore
            store = SkillStore()
            try:
                ctx = store.get_session_context(session_id)
                count = int(ctx.get("message_count") or 0)
            finally:
                store.close()
        except Exception:
            pass

    # Fall back to temp-file counter if store had nothing
    if count == 0 and session_id:
        counter_path = os.path.join(tempfile.gettempdir(), f"claude-router-msg-{session_id}")
        try:
            count = int(open(counter_path).read().strip())
        except (OSError, ValueError):
            pass

    if count == 0:
        return False, ""

    # Heuristic: each user+assistant exchange ≈ 2000 tokens average.
    # Claude Opus context = 200K tokens → ~100 exchanges fills context.
    estimated_fraction = count / 100.0
    if estimated_fraction >= threshold:
        return True, (
            f"~{count} messages in session (~{int(estimated_fraction * 100)}% "
            "estimated context). Mention to the user: they should type /compact "
            "to summarize the context and prevent mid-task truncation."
        )
    return False, ""


def increment_message_counter(session_id: str) -> int:
    """Increment and return the per-session message counter (compact advisor fallback)."""
    import os
    import tempfile

    if not session_id:
        return 0

    path = os.path.join(tempfile.gettempdir(), f"claude-router-msg-{session_id}")
    try:
        count = int(open(path).read().strip()) if os.path.exists(path) else 0
        count += 1
        with open(path, "w") as f:
            f.write(str(count))
        return count
    except (OSError, ValueError):
        return 0
