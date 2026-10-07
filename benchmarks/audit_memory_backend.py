#!/usr/bin/env python3
"""Read-only, aggregate source-coverage check before retiring raw indexes.

References and hashes establish migration prerequisites, not semantic quality.
No source paths or content are emitted. This command never migrates a store.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3


def audit(db_path: Path, wiki_root: Path) -> dict:
    from skill_hub.wiki import _hash_text, parse_frontmatter

    def local_path(value: object) -> str | None:
        if not isinstance(value, str) or not value or value.startswith(("http://", "https://")):
            return None
        path = Path(value).expanduser()
        return str(path.resolve()) if path.is_absolute() else None

    raw_paths: set[str] = set()
    unresolved_raw_rows = 0
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as conn:
        rows = conn.execute("SELECT metadata FROM vectors WHERE namespace LIKE 'memory:%'")
        for (metadata,) in rows:
            try:
                value = json.loads(metadata or "{}")
            except (ValueError, TypeError):
                value = None
            path = value.get("path") if isinstance(value, dict) else None
            resolved = local_path(path)
            if resolved is None:
                unresolved_raw_rows += 1
            else:
                raw_paths.add(resolved)

    references: set[str] = set()
    references_by_scope = {"public": set(), "private": set(), "unknown": set()}
    current_first_sources: set[str] = set()
    pages = missing_hash = stale = unreadable_source = unreadable_page = malformed_page = 0
    unresolved_refs = pages_without_refs = nonlocal_first_source = 0
    scopes = {"public": 0, "private": 0, "unknown": 0}
    pages_root = wiki_root / "pages"
    for file in pages_root.rglob("*.md") if pages_root.is_dir() else ():
        try:
            frontmatter, _ = parse_frontmatter(file.read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            unreadable_page += 1
            continue
        if not frontmatter:
            malformed_page += 1
            continue
        pages += 1
        scope = frontmatter.get("scope")
        scope = scope if scope in ("public", "private") else "unknown"
        scopes[scope] += 1
        refs = frontmatter.get("source_refs") or []
        if isinstance(refs, str):
            refs = [refs]
        if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
            malformed_page += 1
            continue
        if not refs:
            pages_without_refs += 1
            continue
        for ref in refs:
            resolved = local_path(ref)
            if resolved is None:
                unresolved_refs += 1
            else:
                references.add(resolved)
                references_by_scope[scope].add(resolved)
        source_hash = frontmatter.get("source_hash")
        if not isinstance(source_hash, str) or not source_hash:
            missing_hash += 1
            continue
        first = local_path(refs[0])
        if first is None:
            nonlocal_first_source += 1
            continue
        try:
            text = Path(first).read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            unreadable_source += 1
            continue
        if _hash_text(text) == source_hash:
            current_first_sources.add(first)
        else:
            stale += 1
    available = pages_root.is_dir()
    coverage = (available and bool(raw_paths) and raw_paths <= references
                and not (unresolved_raw_rows or unresolved_refs or unreadable_page
                         or malformed_page or scopes["unknown"]))
    return {
        "raw_source_paths": len(raw_paths),
        "raw_rows_with_unresolved_path": unresolved_raw_rows,
        "referenced_raw_source_paths": len(raw_paths & references),
        "unreferenced_raw_source_paths": len(raw_paths - references),
        "raw_paths_referenced_by_public_pages": len(raw_paths & references_by_scope["public"]),
        "raw_paths_referenced_by_private_pages": len(raw_paths & references_by_scope["private"]),
        "raw_paths_referenced_by_unknown_scope_pages": len(raw_paths & references_by_scope["unknown"]),
        "raw_paths_with_current_first_source_hash": len(raw_paths & current_first_sources),
        "wiki_pages_root_available": available,
        "wiki_pages": pages,
        "wiki_public_pages": scopes["public"],
        "wiki_private_pages": scopes["private"],
        "wiki_pages_unknown_scope": scopes["unknown"],
        "wiki_pages_without_source_refs": pages_without_refs,
        "wiki_pages_with_unresolved_refs": unresolved_refs,
        "wiki_pages_with_nonlocal_first_source": nonlocal_first_source,
        "wiki_pages_malformed": malformed_page,
        "wiki_pages_unreadable": unreadable_page,
        "wiki_pages_missing_source_hash": missing_hash,
        "wiki_pages_with_stale_first_source": stale,
        "wiki_pages_with_unreadable_first_source": unreadable_source,
        "reference_coverage_complete": coverage,
        "safe_to_retire_raw_index": False,
        "semantic_quality_qualified": False,
        "warning": "Path coverage and first-source hashes cannot verify all references or distilled fact fidelity.",
    }


def main() -> None:
    from skill_hub import config
    from skill_hub.store import DB_PATH

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--wiki-root", type=Path, default=Path(config.get("wiki_root")))
    args = parser.parse_args()
    print(json.dumps(audit(args.db, args.wiki_root), indent=2))


if __name__ == "__main__":
    main()
