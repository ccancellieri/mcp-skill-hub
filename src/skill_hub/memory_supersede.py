"""Outdate contradicted memory when a new decision is written (#136).

``promote_memory`` moves entries between levels by age/access and
``optimize_memory`` dry-runs a KEEP/PRUNE/COMPACT/MERGE report, but nothing
notices that a freshly written decision *contradicts* an older one. This does:
after a memory file is written, find the nearest existing entries in the same
namespace, ask the escalation ladder which are now superseded, and mark them
non-destructively — a ``superseded_by`` frontmatter stamp plus a ``memory_audit``
row. Nothing is ever deleted; the old file stays on disk and searchable.

Runs fire-and-forget from the memory-write path (daemon thread, own store) and
degrades to a no-op when no embedding backend is up (neighbor search returns
nothing → no ladder call).
"""
from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path

from . import config as _cfg

log = logging.getLogger(__name__)

_JUDGE_PROMPT = """You are auditing a memory store for outdated entries.

A NEW memory entry was just written. Below it are EXISTING entries that are
semantically similar. Decide which existing entries the NEW one makes OUTDATED
— i.e. it reverses, replaces, or contradicts them. Do NOT flag an entry that is
merely related, complementary, or about a different aspect; only a genuine
supersede.

Return a JSON array (possibly empty) of objects:
[{"id": "<existing id>", "reason": "<one short clause on what changed>"}]
Return ONLY the JSON array, no prose.

=== NEW ENTRY ===
{new_text}

=== EXISTING ENTRIES ===
{candidates}
"""

_CANDIDATE_MAX_CHARS = 800
_NEW_MAX_CHARS = 1500


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _read_body(doc_id: str) -> str:
    try:
        return Path(doc_id).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _extract_json_array(text: str) -> list[dict]:
    """Best-effort parse of a JSON array from an LLM response."""
    if not text:
        return []
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        parsed = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return []
    return [x for x in parsed if isinstance(x, dict)] if isinstance(parsed, list) else []


def find_candidates(store, new_text: str, namespace: str, *,
                    top_k: int, min_sim: float, exclude_doc_id: str) -> list[dict]:
    """Nearest existing entries in ``namespace`` above the raw-similarity floor.

    Level weighting and recency decay are disabled so ``similarity_threshold``
    gates on raw cosine — we want semantic closeness, not promotion score.
    """
    try:
        hits = store.search_vectors(
            new_text,
            namespaces=[namespace],
            top_k=top_k + 1,
            similarity_threshold=min_sim,
            apply_level_weight=False,
            apply_recency_decay=False,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug("memory_supersede: neighbor search failed: %s", exc)
        return []
    return [h for h in hits if h.get("doc_id") and h["doc_id"] != exclude_doc_id][:top_k]


def _judge(new_text: str, candidates: list[dict]) -> list[dict]:
    """Ask the ladder which candidate ids the new entry supersedes."""
    from .llm.request import request

    listing = "\n\n".join(
        f"[id: {c['doc_id']}]\n{c['body'][:_CANDIDATE_MAX_CHARS]}"
        for c in candidates
    )
    prompt = _JUDGE_PROMPT.format(
        new_text=new_text[:_NEW_MAX_CHARS], candidates=listing,
    )
    try:
        out = request("mid", prompt, op="memory_supersede",
                      timeout=60, max_tokens=400)
    except Exception as exc:  # noqa: BLE001
        log.debug("memory_supersede: ladder judge failed: %s", exc)
        return []
    valid_ids = {c["doc_id"] for c in candidates}
    return [v for v in _extract_json_array(out) if v.get("id") in valid_ids]


def _stamp_frontmatter(doc_id: str, new_name: str) -> bool:
    """Add/refresh ``superseded_by`` + ``superseded_at`` in the file frontmatter.

    Returns True when the file was rewritten. Only touches files that already
    carry a ``---`` frontmatter block; leaves the body untouched.
    """
    path = Path(doc_id)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    m = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    if not m:
        return False
    fm = m.group(1)
    # Drop any prior supersede stamp so re-runs stay idempotent.
    fm_lines = [
        ln for ln in fm.split("\n")
        if not ln.startswith(("superseded_by:", "superseded_at:"))
    ]
    fm_lines.append(f"superseded_by: {new_name}")
    fm_lines.append(f"superseded_at: {_now_iso()}")
    new_text = f"---\n" + "\n".join(fm_lines) + "\n---\n" + text[m.end():]
    try:
        path.write_text(new_text, encoding="utf-8")
    except OSError as exc:
        log.debug("memory_supersede: could not stamp %s: %s", doc_id, exc)
        return False
    return True


def run_supersede(store, *, new_doc_id: str, new_name: str,
                  new_text: str, namespace: str) -> dict:
    """One supersede pass. Never raises. Returns a small result dict."""
    if not _cfg.get("memory_supersede_enabled"):
        return {"skipped": "disabled"}
    if not new_text.strip():
        return {"skipped": "empty"}

    min_sim = float(_cfg.get("memory_supersede_min_sim") or 0.72)
    top_k = int(_cfg.get("memory_supersede_top_k") or 6)

    candidates = find_candidates(
        store, new_text, namespace, top_k=top_k, min_sim=min_sim,
        exclude_doc_id=new_doc_id,
    )
    if not candidates:
        return {"candidates": 0}
    for c in candidates:
        c["body"] = _read_body(c["doc_id"])
    candidates = [c for c in candidates if c["body"]]
    if not candidates:
        return {"candidates": 0}

    verdicts = _judge(new_text, candidates)
    marked = []
    for v in verdicts:
        doc_id = v["id"]
        reason = str(v.get("reason", ""))[:300]
        if _stamp_frontmatter(doc_id, new_name):
            try:
                store.record_memory_audit(
                    action="supersede",
                    namespace=namespace,
                    doc_id=doc_id,
                    reason_json={"superseded_by": new_name, "reason": reason},
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("memory_supersede: audit write failed: %s", exc)
            marked.append(doc_id)
    return {"candidates": len(candidates), "superseded": marked}


def supersede_async(*, new_doc_id: str, new_name: str,
                    new_text: str, namespace: str) -> None:
    """Fire-and-forget supersede (daemon thread with its own store)."""
    if not _cfg.get("memory_supersede_enabled"):
        return

    def _run() -> None:
        store = None
        try:
            from .store import SkillStore
            store = SkillStore()
            result = run_supersede(
                store, new_doc_id=new_doc_id, new_name=new_name,
                new_text=new_text, namespace=namespace,
            )
            if result.get("superseded"):
                log.info("memory_supersede: %s", result)
        except Exception as exc:  # noqa: BLE001
            log.warning("memory_supersede: async run failed: %s", exc)
        finally:
            if store is not None:
                try:
                    store.close()
                except Exception:  # noqa: BLE001
                    pass

    t = threading.Thread(target=_run, name="memory-supersede", daemon=True)
    t.start()
