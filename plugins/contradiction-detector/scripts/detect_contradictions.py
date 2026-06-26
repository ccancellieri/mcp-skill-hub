"""LLM-powered contradiction detection across wiki pages.

Usage:
    python -m contradiction_detector.detect [--batch-size N] [--dry-run]

Scheduled via plugin's scheduled_tasks entry; can also be run manually.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

_log = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 10
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_MAX_PAGE_PAIRS = 100


def get_wiki_root() -> Path:
    from skill_hub import config as _cfg
    return Path(
        _cfg.get("wiki_root")
        or Path.home() / ".claude" / "mcp-skill-hub" / "wiki"
    )


def get_plugin_config() -> dict:
    from skill_hub.plugin_registry import iter_enabled_plugins
    for p in iter_enabled_plugins():
        if p["name"] == "contradiction-detector":
            return p["manifest"].get("config", {})
    return {}


def _get_entity_pages(store: Any, wiki_root: Path) -> list[dict]:
    conn = store._conn
    rows = conn.execute(
        """
        SELECT slug, title, type, scope, rel_path
        FROM wiki_pages
        WHERE type IN ('entity', 'concept')
        ORDER BY slug
        """
    ).fetchall()
    pages = []
    for r in rows:
        rel_path = r["rel_path"] if not isinstance(r, tuple) else r[4]
        page_path = wiki_root / rel_path
        if page_path.exists():
            try:
                body = page_path.read_text(encoding="utf-8")
                fm, content = _parse_frontmatter(body)
                pages.append({
                    "slug": r["slug"] if not isinstance(r, tuple) else r[0],
                    "title": r["title"] if not isinstance(r, tuple) else r[1],
                    "type": r["type"] if not isinstance(r, tuple) else r[2],
                    "scope": r["scope"] if not isinstance(r, tuple) else r[3],
                    "body": content,
                })
            except OSError:
                continue
    return pages


def _parse_frontmatter(text: str) -> tuple[dict, str]:
    if not text.startswith("---"):
        return {}, text
    rest = text[3:]
    nl = rest.find("\n")
    if nl == -1:
        return {}, text
    rest = rest[nl + 1:]
    close = rest.find("\n---")
    if close == -1:
        return {}, text
    fm_text = rest[:close]
    after = rest[close + 4:]
    if after.startswith("\n"):
        after = after[1:]
    try:
        import yaml
        fm = yaml.safe_load(fm_text) or {}
    except Exception:
        fm = {}
    return fm, after


def _find_related_pairs(store: Any, pages: list[dict], max_pairs: int) -> list[tuple[dict, dict]]:
    conn = store._conn
    slug_to_page = {p["slug"]: p for p in pages}
    pairs: list[tuple[dict, dict]] = []
    seen: set[tuple[str, str]] = set()

    for p in pages:
        slug = p["slug"]
        edges = conn.execute(
            """
            SELECT dst_slug FROM wiki_edges
            WHERE src_slug = ? AND resolved = 1
            """,
            (slug,),
        ).fetchall()
        for e in edges:
            dst = e[0]
            if dst in slug_to_page and dst != slug:
                key = tuple(sorted([slug, dst]))
                if key not in seen:
                    seen.add(key)
                    pairs.append((p, slug_to_page[dst]))
                    if len(pairs) >= max_pairs:
                        return pairs
    return pairs


def _build_prompt(page_a: dict, page_b: dict, template: str) -> str:
    return template.replace("{{ page_a_title }}", page_a["title"]) \
                   .replace("{{ page_a_slug }}", page_a["slug"]) \
                   .replace("{{ page_a_body }}", page_a["body"][:3000]) \
                   .replace("{{ page_b_title }}", page_b["title"]) \
                   .replace("{{ page_b_slug }}", page_b["slug"]) \
                   .replace("{{ page_b_body }}", page_b["body"][:3000])


def _detect_contradictions_llm(
    page_a: dict,
    page_b: dict,
    template: str,
    tier: str,
) -> list[dict]:
    from skill_hub.llm import get_provider, LLMError
    prompt = _build_prompt(page_a, page_b, template)
    try:
        raw = get_provider().complete(
            prompt,
            tier=tier,
            temperature=0.3,
            max_tokens=1000,
        )
    except LLMError as exc:
        _log.warning("LLM call failed for %s vs %s: %s", page_a["slug"], page_b["slug"], exc)
        return []

    try:
        findings = json.loads(raw.strip())
        if not isinstance(findings, list):
            return []
        return [
            {
                "page_a": page_a["slug"],
                "page_b": page_b["slug"],
                "claim_a": f.get("claim_a", ""),
                "claim_b": f.get("claim_b", ""),
                "confidence": float(f.get("confidence", 0.5)),
                "reasoning": f.get("reasoning", ""),
            }
            for f in findings
            if isinstance(f, dict) and f.get("claim_a") and f.get("claim_b")
        ]
    except json.JSONDecodeError:
        _log.warning("Failed to parse LLM response for %s vs %s", page_a["slug"], page_b["slug"])
        return []


def detect_contradictions(
    store: Any,
    dry_run: bool = False,
    batch_size: int | None = None,
    max_pairs: int | None = None,
    confidence_threshold: float | None = None,
) -> dict:
    config = get_plugin_config()
    batch_size = batch_size or config.get("batch_size", DEFAULT_BATCH_SIZE)
    max_pairs = max_pairs or config.get("max_page_pairs", DEFAULT_MAX_PAGE_PAIRS)
    confidence_threshold = confidence_threshold or config.get("confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD)
    tier = config.get("llm_tier", "tier_mid")

    wiki_root = get_wiki_root()
    run_id = str(uuid.uuid4())[:8]
    started_at = datetime.utcnow().isoformat()

    plugin_dir = Path(__file__).parent.parent
    template_path = plugin_dir / "templates" / "contradiction_prompt.md"
    template = template_path.read_text(encoding="utf-8") if template_path.exists() else ""

    pages = _get_entity_pages(store, wiki_root)
    pairs = _find_related_pairs(store, pages, max_pairs)

    if dry_run:
        return {
            "dry_run": True,
            "pages_scanned": len(pages),
            "pairs_to_analyze": len(pairs),
            "run_id": run_id,
        }

    conn = store._conn
    conn.execute(
        """
        INSERT INTO plugin_contradiction_runs (id, started_at, status, pages_scanned)
        VALUES (?, ?, 'running', ?)
        """,
        (run_id, started_at, len(pages)),
    )
    conn.commit()

    all_findings: list[dict] = []
    errors: list[str] = []

    for i, (page_a, page_b) in enumerate(pairs):
        findings = _detect_contradictions_llm(page_a, page_b, template, tier)
        for f in findings:
            if f["confidence"] >= confidence_threshold:
                all_findings.append(f)

        if (i + 1) % batch_size == 0:
            _log.info("Processed %d/%d page pairs", i + 1, len(pairs))

    for f in all_findings:
        try:
            conn.execute(
                """
                INSERT OR IGNORE INTO plugin_contradiction_findings
                    (page_a, page_b, claim_a, claim_b, confidence, detection_run)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (f["page_a"], f["page_b"], f["claim_a"], f["claim_b"], f["confidence"], run_id),
            )
        except Exception as exc:
            errors.append(f"{f['page_a']} vs {f['page_b']}: {exc}")

    completed_at = datetime.utcnow().isoformat()
    conn.execute(
        """
        UPDATE plugin_contradiction_runs
        SET completed_at = ?, status = 'completed',
            pairs_analyzed = ?, contradictions_found = ?
        WHERE id = ?
        """,
        (completed_at, len(pairs), len(all_findings), run_id),
    )
    conn.commit()

    return {
        "run_id": run_id,
        "pages_scanned": len(pages),
        "pairs_analyzed": len(pairs),
        "contradictions_found": len(all_findings),
        "findings": all_findings[:20],
        "errors": errors[:5],
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Detect contradictions in wiki pages")
    parser.add_argument("--batch-size", type=int, help="Number of pairs to process before logging progress")
    parser.add_argument("--dry-run", action="store_true", help="Preview without making changes")
    parser.add_argument("--max-pairs", type=int, help="Maximum page pairs to analyze")
    args = parser.parse_args()

    from skill_hub.store import Store
    store = Store()

    result = detect_contradictions(
        store,
        dry_run=args.dry_run,
        batch_size=args.batch_size,
        max_pairs=args.max_pairs,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
