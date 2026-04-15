"""Build the file-scoped context bundle for an executor model.

The speedup lever for the plan-aware executor: instead of letting the model
see the whole repo, we feed it only the files the step declared (plus
protocols_ref / pattern_ref). Sonnet on 5 files runs ~Haiku-fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# ~30k chars ≈ 7.5k tokens — safe for any tier, leaves headroom for output.
DEFAULT_CHAR_BUDGET = 30_000


@dataclass
class BundledFile:
    role: str          # "target" | "protocol" | "pattern"
    rel_path: str
    content: str
    truncated: bool


@dataclass
class ContextBundle:
    files: list[BundledFile]
    total_chars: int
    over_budget: bool

    def render(self) -> str:
        """Flatten to a single markdown blob for the LLM prompt."""
        parts: list[str] = []
        for f in self.files:
            trunc = " (TRUNCATED)" if f.truncated else ""
            parts.append(
                f"### {f.role.upper()}: {f.rel_path}{trunc}\n"
                f"```\n{f.content}\n```"
            )
        return "\n\n".join(parts)


def _read_capped(path: Path, budget_left: int) -> tuple[str, bool]:
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return f"<unreadable: {path}>", False
    if len(text) > budget_left:
        return text[:budget_left] + "\n... <truncated>", True
    return text, False


def build_bundle(
    repo_path: Path,
    files: list[str],
    protocols_ref: list[str] | None = None,
    pattern_ref: list[str] | None = None,
    *,
    char_budget: int = DEFAULT_CHAR_BUDGET,
) -> ContextBundle:
    """Read the files for a step into a single bundle.

    ``files`` are treated as target files (may not exist yet — that's fine,
    we include them as empty stubs so the model knows the intended path).
    ``protocols_ref`` and ``pattern_ref`` must exist; if missing, they're
    logged as `<missing>` and the step can still run (validator should have
    caught this, but we're defensive).
    """
    bundled: list[BundledFile] = []
    remaining = char_budget

    def _add(role: str, rels: list[str], *, target: bool = False) -> None:
        nonlocal remaining
        for rel in rels:
            full = repo_path / rel
            if not full.exists():
                content = "<file does not exist yet>" if target else "<missing>"
                bundled.append(BundledFile(role, rel, content, False))
                remaining -= len(content) + len(rel) + 20
                continue
            text, truncated = _read_capped(full, max(remaining, 0))
            bundled.append(BundledFile(role, rel, text, truncated))
            remaining -= len(text) + len(rel) + 20

    _add("target", files, target=True)
    _add("protocol", protocols_ref or [])
    _add("pattern", pattern_ref or [])

    total = sum(len(b.content) for b in bundled)
    return ContextBundle(files=bundled, total_chars=total, over_budget=remaining < 0)
