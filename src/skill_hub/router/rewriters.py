"""S5 F-PROMPT — pluggable prompt rewriters.

A rewriter takes a user prompt and returns a prefix string (extra context)
plus a rewritten body. ``improve_prompt`` composes rewriters in order and
emits a single enriched prompt.

Built-in rewriters (registered on import):
- ``add_skill_context`` — injects the top-N semantically relevant skill names
- ``add_recent_tasks`` — injects open tasks titles/summaries
- ``normalize_language`` — optional LLM paraphrase via the F-LLM provider

Design notes
- Rewriters are pure functions ``(prompt, store, cfg) -> RewriterResult``.
- They NEVER raise: all I/O (DB / LLM / embedding) is best-effort wrapped.
- ``improve_prompt`` is the only entry point; callers (CLI, MCP tool,
  hook) pass a list of rewriter names or the ``"all"`` sentinel.
- No rewriter calls ``get_provider()`` unless explicitly selected; keeps
  the default path free of LLM latency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .. import config as _cfg


@dataclass
class RewriterResult:
    prefix: str = ""            # prepended ahead of the user prompt
    body: str | None = None     # optional rewritten body (None = unchanged)
    note: str = ""              # what this rewriter contributed (audit/debug)
    applied: bool = False


Rewriter = Callable[[str, Any, dict[str, Any]], RewriterResult]

_REGISTRY: dict[str, Rewriter] = {}


def _row_get(row: Any, key: str) -> Any:
    """Fetch ``key`` from a dict or sqlite3.Row uniformly."""
    if hasattr(row, "get"):
        return row.get(key)
    try:
        return row[key]
    except (KeyError, IndexError):
        return None


def register(name: str, fn: Rewriter) -> None:
    _REGISTRY[name] = fn


def available() -> list[str]:
    return sorted(_REGISTRY)


def _safe_call(fn: Rewriter, prompt: str, store: Any, cfg: dict[str, Any]) -> RewriterResult:
    try:
        return fn(prompt, store, cfg)
    except Exception as exc:
        return RewriterResult(note=f"error: {exc}")


# ---------------------------------------------------------------------------
# Built-in rewriters
# ---------------------------------------------------------------------------

def _rw_add_skill_context(prompt: str, store: Any, cfg: dict[str, Any]) -> RewriterResult:
    """Inject top-N relevant skill names into a prefix block."""
    top_k = int(cfg.get("improve_prompt_skill_top_k", 3))
    try:
        from ..embeddings import embed, embed_available, EMBED_MODEL
    except ImportError:
        return RewriterResult(note="skill_context: embeddings unavailable")
    if not embed_available():
        return RewriterResult(note="skill_context: embed unavailable")
    try:
        vec = embed(prompt)
    except Exception as exc:
        return RewriterResult(note=f"skill_context: embed failed ({exc})")
    try:
        rows = store.search(vec, top_k=top_k * 3) or []
    except Exception as exc:
        return RewriterResult(note=f"skill_context: search failed ({exc})")
    seen: set[str] = set()
    names: list[str] = []
    for r in rows:
        n = r.get("name", "")
        if n and n not in seen:
            seen.add(n)
            names.append(n)
        if len(names) >= top_k:
            break
    if not names:
        return RewriterResult(note="skill_context: no matches")
    prefix = "[Relevant skills: " + ", ".join(names) + "]"
    return RewriterResult(prefix=prefix, note=f"skill_context: {len(names)} skills", applied=True)


_PRIVILEGED_TAG_RE = __import__("re").compile(
    r"</?\s*(?:system-reminder|system|assistant|user|tool_use|tool_result|"
    r"function_calls|antml:[a-z_]+)\s*/?>",
    __import__("re").IGNORECASE,
)


def _sanitize_for_hook_output(s: str) -> str:
    """Strip privileged tags and control chars that Claude Code's hook-output
    validator treats as prompt-injection and rejects (""Invalid input"").

    Also drop stray braces that some auto-generated task titles carry.
    """
    if not s:
        return ""
    s = _PRIVILEGED_TAG_RE.sub("", s)
    # Drop literal control characters (NUL/backspace/etc.) — keep tab/newline.
    s = "".join(ch for ch in s if ch in ("\t", "\n") or ord(ch) >= 0x20)
    # Collapse runs of whitespace to a single space.
    s = " ".join(s.split())
    return s.strip()


def _rw_add_recent_tasks(prompt: str, store: Any, cfg: dict[str, Any]) -> RewriterResult:
    """Inject up to N recent open tasks (title + first line of summary)."""
    limit = int(cfg.get("improve_prompt_tasks_limit", 2))
    try:
        tasks = store.list_tasks("open") or []
    except Exception as exc:
        return RewriterResult(note=f"recent_tasks: list failed ({exc})")
    if not tasks:
        return RewriterResult(note="recent_tasks: none open")
    parts: list[str] = []
    for t in tasks[:limit]:
        title = _sanitize_for_hook_output(_row_get(t, "title") or "")
        summary_raw = _row_get(t, "summary") or ""
        summary = _sanitize_for_hook_output(
            summary_raw.splitlines()[0] if summary_raw else ""
        )
        if title and summary:
            parts.append(f"{title} — {summary[:120]}")
        elif title:
            parts.append(title)
    # Drop any entries that sanitized down to empty.
    parts = [p for p in parts if p]
    if not parts:
        return RewriterResult(note="recent_tasks: empty titles")
    prefix = "[Open tasks: " + " | ".join(parts) + "]"
    return RewriterResult(prefix=prefix, note=f"recent_tasks: {len(parts)}", applied=True)


_NORMALIZE_SYSTEM = (
    "Rewrite the user's message to be concise and unambiguous without "
    "changing its intent. Keep it short (≤2 sentences). Output ONLY the "
    "rewritten message, no preface or commentary."
)


def _rw_normalize_language(prompt: str, store: Any, cfg: dict[str, Any]) -> RewriterResult:
    """LLM paraphrase via F-LLM provider. Opt-in: only runs on explicit select."""
    if len(prompt.strip()) < 12:
        return RewriterResult(note="normalize: too short to paraphrase")
    try:
        from ..llm import get_provider
    except ImportError:
        return RewriterResult(note="normalize: provider unavailable")
    providers = cfg.get("llm_providers") or {}
    model = providers.get("tier_cheap") if isinstance(providers, dict) else None
    if not model:
        return RewriterResult(note="normalize: no tier_cheap model configured")
    try:
        text = get_provider().chat(
            [
                {"role": "system", "content": _NORMALIZE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            model=str(model),
            max_tokens=180,
            temperature=0.2,
            timeout=8.0,
        )
    except Exception as exc:
        return RewriterResult(note=f"normalize: llm failed ({exc})")
    clean = (text or "").strip().strip("\"'")
    if not clean or clean == prompt.strip():
        return RewriterResult(note="normalize: no change")
    return RewriterResult(body=clean, note="normalize: paraphrased", applied=True)


register("add_skill_context", _rw_add_skill_context)
register("add_recent_tasks", _rw_add_recent_tasks)
register("normalize_language", _rw_normalize_language)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@dataclass
class ImproveResult:
    prompt: str
    original: str
    applied: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def improve_prompt(
    prompt: str,
    store: Any,
    rewriters: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> ImproveResult:
    """Apply the named rewriters to *prompt* and return the combined result.

    - ``rewriters=None`` uses the default chain (``add_skill_context`` +
      ``add_recent_tasks``). ``normalize_language`` is opt-in because it
      calls an LLM.
    - ``rewriters=["all"]`` runs every registered rewriter.
    - Unknown names are silently ignored (noted in the result).
    """
    cfg = cfg or _cfg.load_config()
    default_chain = cfg.get("improve_prompt_default_chain") or [
        "add_skill_context",
        "add_recent_tasks",
    ]
    names = rewriters if rewriters is not None else list(default_chain)
    if names == ["all"]:
        names = available()

    prefixes: list[str] = []
    body = prompt
    applied: list[str] = []
    notes: list[str] = []

    for name in names:
        fn = _REGISTRY.get(name)
        if fn is None:
            notes.append(f"{name}: unknown")
            continue
        r = _safe_call(fn, body, store, cfg)
        notes.append(r.note or f"{name}: ok")
        if r.prefix:
            prefixes.append(r.prefix)
        if r.body is not None:
            body = r.body
        if r.applied:
            applied.append(name)

    enriched = body if not prefixes else "\n".join(prefixes + [body])
    return ImproveResult(prompt=enriched, original=prompt, applied=applied, notes=notes)
