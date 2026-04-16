"""Verdict dataclass, systemMessage formatter, and audit-log writer.

Audit log layout
----------------
  ~/.claude/mcp-skill-hub/router.jsonl  — machine-readable, one JSON per line
  ~/.claude/mcp-skill-hub/router.log    — human-readable, one line per verdict

Rotation policy (applied on every write)
-----------------------------------------
  • Drop JSONL entries older than RETENTION_DAYS (default 5)
  • If file still > SIZE_CAP_MB after date-pruning, keep the most recent MAX_ENTRIES rows
  • Companion .log is rotated to .log.1 once it exceeds SIZE_CAP_MB / 2
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Rotation constants
# ---------------------------------------------------------------------------
RETENTION_DAYS: int = 5
SIZE_CAP_MB: float = 5.0
MAX_ENTRIES: int = 2000   # hard cap after date-pruning

# ---------------------------------------------------------------------------
# Token-saving heuristics and pricing
# ---------------------------------------------------------------------------
_MODEL_PRIORITY = {"haiku": 0, "sonnet": 1, "opus": 2}

# Claude 4.x published pricing: (input $/M tokens, output $/M tokens)
_PRICING: dict[str, tuple[float, float]] = {
    "opus":   (15.0, 75.0),
    "sonnet": ( 3.0, 15.0),
    "haiku":  ( 0.80,  4.0),
}
# Assumed baseline when prev_model is unknown (Claude Code default)
_DEFAULT_BASELINE = "sonnet"

# Flat minimums for non-downgrade savings (skill preload, enrichment, plan mode)
_SKILL_PRELOAD_SAVING = 300   # per skill: avoids one search_skills call + overhead
_ENRICH_SAVING       = 600   # thin-prompt enrichment avoids one clarification round-trip
_PLAN_MODE_SAVING    = 2000  # plan mode on ambiguous prompt avoids rework (speculative)

# Output token estimate from complexity score: complexity × _OUTPUT_SCALE + _OUTPUT_BASE
_OUTPUT_SCALE = 2400  # tokens at complexity=1.0
_OUTPUT_BASE  =  100  # minimum output tokens assumed


def _estimate_tokens_saved(v: "Verdict") -> tuple[int, float, list[str]]:
    """Return (tokens_saved, usd_saved, breakdown_lines).

    Dollar savings use actual API pricing and prompt_len for input tokens.
    Output tokens are estimated from the complexity score.
    """
    saved_tokens = 0
    usd_saved = 0.0
    lines: list[str] = []

    # Model downgrade savings — proportional to actual prompt size
    baseline = v.prev_model if v.prev_model else _DEFAULT_BASELINE
    curr = v.model
    if _MODEL_PRIORITY.get(curr, 1) < _MODEL_PRIORITY.get(baseline, 1):
        input_tokens  = max(v.prompt_len // 4, 50)
        output_tokens = int(v.complexity * _OUTPUT_SCALE) + _OUTPUT_BASE
        b_in, b_out = _PRICING.get(baseline, _PRICING[_DEFAULT_BASELINE])
        c_in, c_out = _PRICING.get(curr,     _PRICING[_DEFAULT_BASELINE])
        tok = input_tokens + output_tokens
        saved_tokens += tok
        usd = (input_tokens * (b_in - c_in) + output_tokens * (b_out - c_out)) / 1_000_000
        usd_saved += usd
        lines.append(f"model {baseline}→{curr}: ~{tok} tok / ${usd:.4f}")

    # Skill preloading
    n_skills = len(v.preload_skills)
    if n_skills:
        n = n_skills * _SKILL_PRELOAD_SAVING
        saved_tokens += n
        lines.append(f"{n_skills} skill(s) preloaded: ~{n} tok")

    # Thin-prompt enrichment
    if v.enrichment_applied:
        saved_tokens += _ENRICH_SAVING
        lines.append(f"thin-prompt enriched ({v.enrichment_source}): ~{_ENRICH_SAVING} tok")

    # Plan mode on ambiguous/complex prompt (avoids rework)
    if v.plan_mode and v.tier_used >= 2:
        saved_tokens += _PLAN_MODE_SAVING
        lines.append(f"plan mode on ambiguous prompt: ~{_PLAN_MODE_SAVING} tok (rework avoided)")

    return saved_tokens, usd_saved, lines


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    model: str = "sonnet"            # "haiku" | "sonnet" | "opus"
    plan_mode: bool = False
    preload_skills: list[str] = field(default_factory=list)
    preload_plugins: list[str] = field(default_factory=list)
    confidence: float = 0.5
    reasoning: str = ""
    tier_used: int = 1               # 1 | 2 | 3

    # Classifier scores
    complexity: float = 0.5
    ambiguity: float = 0.3
    scope: str = "single"
    domain_hints: list[str] = field(default_factory=list)

    # Haiku batch extras (Tier 3 only)
    settings_opt: dict[str, Any] = field(default_factory=dict)
    compact_hint: dict[str, Any] = field(default_factory=dict)
    subtasks: list[str] = field(default_factory=list)

    # Enforcement outcome
    enforcement: str = "suggest"     # "hard_switch" | "suggest" | "none"
    prev_model: str = ""             # model before enforcement (for savings calc)

    # Thin-prompt enrichment
    enrichment_applied: bool = False
    enrichment_source: str = ""      # "session_context" | "semantic_task" | "recent_task"
    enrichment_chars: int = 0

    # Latency breakdown
    latency_ms: int = 0
    tier1_ms: int = 0
    tier2_ms: int = 0
    tier3_ms: int = 0

    # Identity
    session_id: str = ""
    task_id: int | None = None       # active task at routing time (from active_task.json)
    prompt_preview: str = ""         # first 80 chars
    prompt_len: int = 0              # full prompt length in chars (for savings estimation)


# ---------------------------------------------------------------------------
# systemMessage formatter
# ---------------------------------------------------------------------------

_TIER_LABELS = {1: "heuristic", 2: "Ollama", 3: "Haiku"}


def format_system_message(v: Verdict) -> str:
    """Return the text injected as systemMessage into Claude's context."""
    lines: list[str] = []

    # ── Header ─────────────────────────────────────────────────────────────
    mode_tag = "+plan" if v.plan_mode else ""
    enforcement_tag = " [applied]" if v.enforcement == "hard_switch" else " [suggested]"
    tier_label = _TIER_LABELS.get(v.tier_used, str(v.tier_used))
    lines.append(
        f"[Router] {v.model}{mode_tag}{enforcement_tag} "
        f"(confidence={v.confidence:.0%}, tier={tier_label}) — {v.reasoning}"
    )

    # ── Preloaded skills ────────────────────────────────────────────────────
    if v.preload_skills:
        lines.append(f"[Router] Preloaded skills: {', '.join(v.preload_skills)}")

    if v.preload_plugins:
        lines.append(f"[Router] Suggested plugins to enable: {', '.join(v.preload_plugins)}")

    # ── Compact hint ────────────────────────────────────────────────────────
    ch = v.compact_hint
    if ch.get("suggest_compact"):
        reason = ch.get("reason", "")
        lines.append(
            f"[Router] COMPACT ADVISORY: {reason} "
            f"Start your response with a short notice to the user: "
            f"'Context window is getting full — please type /compact before we continue.' "
            f"(/compact is handled by the CLI harness, not a tool you can call.) "
            f"Then proceed with the task normally."
        )

    # ── Subtask decomposition ───────────────────────────────────────────────
    if v.subtasks:
        lines.append("[Router] Multiple independent subtasks detected — tackle serially:")
        for i, st in enumerate(v.subtasks, 1):
            lines.append(f"  {i}. {st}")

    # ── Settings optimisation (Haiku Tier 3) ───────────────────────────────
    so = v.settings_opt
    if so.get("key"):
        lines.append(
            f"[Router] Settings hint: consider `{so['key']}` = {so.get('value')!r} "
            f"— {so.get('reason', '')}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

def append_audit_log(v: Verdict, log_path: str | Path) -> None:
    """Append one entry to both the JSONL machine log and the human-readable log.

    Also runs log rotation (date pruning + size cap) on every write.
    """
    path = Path(log_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return

    tokens_saved, usd_saved, savings_breakdown = _estimate_tokens_saved(v)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

    # ── JSONL entry ─────────────────────────────────────────────────────────
    entry: dict[str, Any] = {
        "ts": ts,
        "session_id": v.session_id,
        "task_id": v.task_id,
        "prompt": v.prompt_preview,
        "verdict": {
            "model": v.model,
            "plan_mode": v.plan_mode,
            "confidence": round(v.confidence, 3),
            "complexity": round(v.complexity, 3),
            "ambiguity": round(v.ambiguity, 3),
            "scope": v.scope,
            "domain": v.domain_hints,
            "tier": v.tier_used,
            "tier_label": _TIER_LABELS.get(v.tier_used, "?"),
            "reasoning": v.reasoning,
            "enforcement": v.enforcement,
            "prev_model": v.prev_model,
        },
        "skills": {
            "preloaded": v.preload_skills,
            "plugins_suggested": v.preload_plugins,
        },
        "enrichment": {
            "applied": v.enrichment_applied,
            "source": v.enrichment_source,
            "chars_added": v.enrichment_chars,
        },
        "compact": {
            "suggested": bool(v.compact_hint.get("suggest_compact")),
            "reason": v.compact_hint.get("reason", ""),
        },
        "subtasks": v.subtasks,
        "savings": {
            "tokens_estimated": tokens_saved,
            "usd_saved": round(usd_saved, 6),
            "breakdown": savings_breakdown,
        },
        "latency": {
            "total_ms": v.latency_ms,
            "tier1_ms": v.tier1_ms,
            "tier2_ms": v.tier2_ms,
            "tier3_ms": v.tier3_ms,
        },
    }

    _safe_append(path, json.dumps(entry))
    _rotate_jsonl(path)

    # ── Human-readable companion .log ───────────────────────────────────────
    human_path = path.with_suffix(".log")
    _safe_append(human_path, _human_line(v, tokens_saved, usd_saved, ts))
    _rotate_humanlog(human_path)


def _human_line(v: Verdict, tokens_saved: int, usd_saved: float, ts: str) -> str:
    """One compact human-readable line per verdict."""
    ts_short = ts[5:19].replace("T", " ")   # "04-13 12:00:05"
    model_tag = f"{v.model}{'+plan' if v.plan_mode else ''}"
    tier_label = _TIER_LABELS.get(v.tier_used, "?")
    conf_pct = f"{v.confidence:.0%}"
    enf = {"hard_switch": "→applied", "suggest": "→suggest", "none": ""}.get(v.enforcement, "")

    skills_tag = ""
    if v.preload_skills:
        skills_tag = "  skills:+" + ",".join(v.preload_skills[:3])
    plugins_tag = ""
    if v.preload_plugins:
        plugins_tag = "  plugins:+" + ",".join(v.preload_plugins)

    enrich_tag = f"  enrich:{v.enrichment_source}" if v.enrichment_applied else ""
    compact_tag = "  COMPACT!" if v.compact_hint.get("suggest_compact") else ""
    subtask_tag = f"  tasks:{len(v.subtasks)}" if v.subtasks else ""
    saved_tag = (
        f"  saved:~{tokens_saved}tok/${usd_saved:.4f}" if usd_saved > 0
        else f"  saved:~{tokens_saved}tok" if tokens_saved > 0
        else ""
    )
    lat_tag = f"  {v.latency_ms}ms"

    prompt_tag = f'  "{v.prompt_preview[:60]}"' if v.prompt_preview else ""

    return (
        f"{ts_short}  {model_tag} [{tier_label} {conf_pct}{enf}]"
        f"  c={v.complexity:.2f} a={v.ambiguity:.2f} s={v.scope}"
        f"{skills_tag}{plugins_tag}{enrich_tag}{compact_tag}{subtask_tag}"
        f"{saved_tag}{lat_tag}{prompt_tag}"
    )


# ---------------------------------------------------------------------------
# Rotation helpers
# ---------------------------------------------------------------------------

def _safe_append(path: Path, line: str) -> None:
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def _rotate_jsonl(path: Path) -> None:
    """Drop entries older than RETENTION_DAYS; cap at MAX_ENTRIES / SIZE_CAP_MB."""
    try:
        if not path.exists():
            return
        size_mb = path.stat().st_size / 1_048_576
        cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)

        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return

        # Date-prune
        kept: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                ts_str = json.loads(line).get("ts", "")
                entry_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                if entry_dt >= cutoff:
                    kept.append(line)
            except Exception:
                kept.append(line)  # keep malformed lines rather than lose data

        # Size cap — keep most recent MAX_ENTRIES rows
        if len(kept) > MAX_ENTRIES:
            kept = kept[-MAX_ENTRIES:]

        # Only rewrite if something was pruned
        if len(kept) < len(lines) or size_mb > SIZE_CAP_MB:
            path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    except OSError:
        pass


def _rotate_humanlog(path: Path) -> None:
    """Rotate .log → .log.1 when it exceeds SIZE_CAP_MB / 2."""
    try:
        if not path.exists():
            return
        if path.stat().st_size / 1_048_576 > SIZE_CAP_MB / 2:
            backup = path.with_suffix(".log.1")
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
            path.write_text("", encoding="utf-8")
    except OSError:
        pass
