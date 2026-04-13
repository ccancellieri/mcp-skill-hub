"""Tier 1 — heuristic prompt classifier.

Fast (<5ms), zero-cost, zero-network. Scores complexity, ambiguity, scope, and
domain from deterministic signal extraction. Returns confidence reflecting how
many clear signals were found — low confidence triggers Tier 2 escalation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

_COMPLEX_VERBS = re.compile(
    r"\b(refactor|redesign|architect|design|migrate|debug|investigate|diagnose|"
    r"analyse|analyze|understand|explain|why|how does|implement|build|create|"
    r"integrate|optimize|rewrite|restructure|plan|review)\b",
    re.IGNORECASE,
)

_SIMPLE_VERBS = re.compile(
    r"\b(rename|format|typo|fix import|add comment|commit|save|push|update "
    r"version|bump|tag|release|revert|undo|delete|remove line|add line)\b",
    re.IGNORECASE,
)

_MULTI_FILE = re.compile(
    r"\b(across|all files|everywhere|project.wide|throughout|globally|"
    r"each file|every file|all tests|all modules)\b",
    re.IGNORECASE,
)

_AMBIGUITY_MARKERS = re.compile(
    r"\b(or|either|should i|which|not sure|unclear|maybe|consider|"
    r"best approach|pros and cons|versus|vs\.?|alternatives?)\b",
    re.IGNORECASE,
)

_DOMAIN_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("debugging", re.compile(
        r"\b(debug|error|crash|traceback|exception|fail|broken|bug|issue|"
        r"not working|stack trace|segfault|timeout|hang)\b", re.IGNORECASE)),
    ("architecture", re.compile(
        r"\b(architect|design|pattern|refactor|restructure|layer|abstraction|"
        r"coupling|cohesion|SOLID|DRY|separation of concerns)\b", re.IGNORECASE)),
    ("testing", re.compile(
        r"\b(test|spec|TDD|coverage|mock|fixture|assert|pytest|unittest|"
        r"flaky|integration test|e2e|end.to.end)\b", re.IGNORECASE)),
    ("frontend", re.compile(
        r"\b(frontend|UI|CSS|React|Vue|Angular|Svelte|component|HTML|DOM|"
        r"layout|responsive|tailwind|shadcn|Vite|webpack)\b", re.IGNORECASE)),
    ("database", re.compile(
        r"\b(database|SQL|migration|schema|index|query|postgres|mysql|"
        r"sqlite|mongodb|redis|transaction|ORM|alembic)\b", re.IGNORECASE)),
    ("security", re.compile(
        r"\b(auth|security|token|permission|vulnerability|XSS|CSRF|"
        r"injection|oauth|JWT|OIDC|RBAC|encrypt|sanitize)\b", re.IGNORECASE)),
    ("devops", re.compile(
        r"\b(deploy|CI|CD|Docker|kubernetes|k8s|pipeline|terraform|"
        r"cloud run|helm|container|infra|Dockerfile|workflow)\b", re.IGNORECASE)),
    ("api", re.compile(
        r"\b(API|endpoint|REST|GraphQL|gRPC|route|request|response|"
        r"openapi|swagger|fastapi|flask|django|express)\b", re.IGNORECASE)),
]


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class HeuristicSignals:
    complexity: float = 0.5   # 0=trivial .. 1=highly complex
    ambiguity: float = 0.3    # 0=clear .. 1=very ambiguous
    scope: str = "single"     # "single" | "multi" | "cross-repo"
    domain_hints: list[str] = field(default_factory=list)
    confidence: float = 0.5   # how certain the heuristic is in its verdict

    # Derived verdict (set by _derive_verdict)
    model: str = "sonnet"
    plan_mode: bool = False


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def classify(prompt: str) -> HeuristicSignals:
    """Return heuristic signals for *prompt*."""
    sig = HeuristicSignals(ambiguity=0.2)  # neutral ambiguity starts lower than complexity
    signal_count = 0  # how many decisive signals found

    length = len(prompt)

    # ── Complexity signals ──────────────────────────────────────────────────
    complex_hits = len(_COMPLEX_VERBS.findall(prompt))
    simple_hits = len(_SIMPLE_VERBS.findall(prompt))

    if complex_hits:
        sig.complexity += complex_hits * 0.12
        signal_count += complex_hits
    if simple_hits:
        sig.complexity -= simple_hits * 0.15
        signal_count += simple_hits

    # Length: long prompts are rarely trivial
    if length > 1000:
        sig.complexity += 0.25
        signal_count += 2
    elif length > 400:
        sig.complexity += 0.12
        signal_count += 1
    elif length < 40:
        sig.complexity -= 0.10  # very short: probably a quick command

    # Code fence → complex context
    if "```" in prompt:
        sig.complexity += 0.10
        signal_count += 1

    # ── Ambiguity signals ───────────────────────────────────────────────────
    q_count = prompt.count("?")
    if q_count > 1:
        sig.ambiguity += 0.15 * min(q_count, 3)
        signal_count += 1
    elif q_count == 1:
        sig.ambiguity += 0.10
        signal_count += 1

    amb_hits = len(_AMBIGUITY_MARKERS.findall(prompt))
    if amb_hits:
        sig.ambiguity += amb_hits * 0.12
        signal_count += amb_hits

    # ── Scope ───────────────────────────────────────────────────────────────
    if _MULTI_FILE.search(prompt):
        sig.scope = "cross-repo" if re.search(r"\brepo\b|\brepository\b", prompt, re.I) else "multi"
        sig.complexity += 0.15
        signal_count += 2
    elif re.search(r"\b\w+\.\w{1,6}\b", prompt):  # any file.ext reference
        sig.scope = "single"

    # ── Domain hints ────────────────────────────────────────────────────────
    for domain, pattern in _DOMAIN_PATTERNS:
        if pattern.search(prompt):
            sig.domain_hints.append(domain)
            signal_count += 1

    # ── Clamp ───────────────────────────────────────────────────────────────
    sig.complexity = _clamp(sig.complexity)
    sig.ambiguity = _clamp(sig.ambiguity)

    # ── Confidence — how decisive were our signals? ─────────────────────────
    # Many clear signals = high confidence; sparse or contradictory = lower
    if signal_count >= 5:
        sig.confidence = 0.90
    elif signal_count >= 3:
        sig.confidence = 0.78
    elif signal_count >= 1:
        sig.confidence = 0.62
    else:
        sig.confidence = 0.45  # neutral prompt — Tier 2 will decide

    # Additional confidence boost when both complexity and ambiguity are extreme
    if sig.complexity >= 0.8 or sig.complexity <= 0.15:
        sig.confidence = min(1.0, sig.confidence + 0.08)

    # ── Derive verdict ───────────────────────────────────────────────────────
    _derive_verdict(sig)
    return sig


def _derive_verdict(sig: HeuristicSignals) -> None:
    """Set sig.model and sig.plan_mode from complexity/ambiguity."""
    c, a = sig.complexity, sig.ambiguity

    if c < 0.30 and a < 0.30:
        sig.model = "haiku"
        sig.plan_mode = False
    elif c >= 0.70 or (c >= 0.50 and a >= 0.60):
        sig.model = "opus"
        sig.plan_mode = True
    elif c >= 0.50 or a >= 0.45:
        sig.model = "sonnet"
        sig.plan_mode = False
    else:
        sig.model = "sonnet"
        sig.plan_mode = False
