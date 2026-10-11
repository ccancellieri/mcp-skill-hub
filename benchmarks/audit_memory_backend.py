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
    references_by_scope = {"public": set(), "private": set()}
    current_first_sources = 0
    all_pages = unreadable_page = malformed_page = 0
    missing_hash = stale = unreadable_source = nonlocal_first_source = 0
    source_pages = source_missing_hash = source_stale = source_unreadable = 0
    source_current = 0
    source_ingest_receipt_current = source_missing_ingest_receipt = 0
    source_nonlocal = pages_without_refs = unresolved_refs = 0
    scope_mismatches = 0
    pages_by_scope = {"public": 0, "private": 0}
    source_pages_by_scope = {"public": 0, "private": 0}
    roots = (("public", wiki_root / "pages"), ("private", wiki_root / "_private"))
    public_root_available = (wiki_root / "pages").is_dir()
    private_root_available = (wiki_root / "_private").is_dir()
    available = public_root_available or private_root_available
    for disk_scope, pages_root in roots:
        for file in pages_root.rglob("*.md") if pages_root.is_dir() else ():
            # Index pages are generated catalogs, not source or knowledge pages.
            if file.name.casefold() == "index.md" or ".backups" in file.parts:
                continue
            try:
                file.resolve().relative_to(pages_root.resolve())
                if disk_scope == "private":
                    if file.relative_to(pages_root).parts[0] != file.resolve().relative_to(pages_root.resolve()).parts[0]:
                        continue
            except (OSError, ValueError):
                continue
            try:
                frontmatter, _ = parse_frontmatter(file.read_text(encoding="utf-8"))
            except (OSError, UnicodeError):
                unreadable_page += 1
                continue
            if not frontmatter:
                malformed_page += 1
                continue
            all_pages += 1
            declared_scope = frontmatter.get("scope")
            scope_valid = declared_scope == disk_scope
            if not scope_valid:
                scope_mismatches += 1
            pages_by_scope[disk_scope] += 1
            page_type = str(frontmatter.get("type") or "entity")
            is_source = page_type == "source"
            if is_source:
                source_pages += 1
                source_pages_by_scope[disk_scope] += 1
            refs = frontmatter.get("source_refs") or []
            if isinstance(refs, str):
                refs = [refs]
            if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
                malformed_page += 1
                continue
            if not refs:
                pages_without_refs += 1
            for ref in refs:
                resolved = local_path(ref)
                if resolved is None:
                    unresolved_refs += 1
                else:
                    references.add(resolved)
                    if scope_valid:
                        references_by_scope[disk_scope].add(resolved)

            source_hash = frontmatter.get("source_hash")
            if not isinstance(source_hash, str) or not source_hash:
                missing_hash += 1
                if is_source:
                    source_missing_hash += 1
            ingest_hash = frontmatter.get("ingest_hash")
            if is_source:
                source_missing_ingest_receipt += 1
            if not isinstance(source_hash, str) or not source_hash:
                continue
            first = local_path(refs[0]) if refs else None
            if first is None:
                nonlocal_first_source += 1
                if is_source:
                    source_nonlocal += 1
                continue
            try:
                text = Path(first).read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                unreadable_source += 1
                if is_source:
                    source_unreadable += 1
                continue
            if _hash_text(text) == source_hash:
                current_first_sources += 1
                if is_source:
                    source_current += 1
                    if isinstance(ingest_hash, str) and ingest_hash == source_hash:
                        source_ingest_receipt_current += 1
                        source_missing_ingest_receipt -= 1
            else:
                stale += 1
                if is_source:
                    source_stale += 1
    referenced_raw_paths = raw_paths & references
    public_paths = raw_paths & references_by_scope["public"]
    private_paths = raw_paths & references_by_scope["private"]
    private_only = private_paths - references_by_scope["public"]
    source_freshness_complete = (
        bool(source_pages)
        and source_missing_hash == 0
        and source_stale == 0
        and source_unreadable == 0
        and source_nonlocal == 0
        and source_current == source_pages
    )
    coverage = (available and bool(raw_paths) and raw_paths <= references
                and not (unresolved_raw_rows or unresolved_refs or unreadable_page
                         or malformed_page or scope_mismatches))
    return {
        "raw_source_paths": len(raw_paths),
        "raw_rows_with_unresolved_path": unresolved_raw_rows,
        "referenced_raw_source_paths": len(referenced_raw_paths),
        "unreferenced_raw_source_paths": len(raw_paths - references),
        "raw_paths_referenced_by_public_pages": len(public_paths),
        "raw_paths_referenced_by_private_pages": len(private_paths),
        "raw_paths_referenced_only_by_private_pages": len(private_only),
        "raw_paths_referenced_by_unknown_scope_pages": 0,
        "wiki_pages_with_current_first_source_hash": current_first_sources,
        "wiki_pages_root_available": available,
        "wiki_public_pages_root_available": public_root_available,
        "wiki_private_pages_root_available": private_root_available,
        "wiki_pages": all_pages,
        "wiki_public_pages": pages_by_scope["public"],
        "wiki_private_pages": pages_by_scope["private"],
        "wiki_pages_unknown_scope": scope_mismatches,
        "wiki_source_pages": source_pages,
        "wiki_source_pages_public": source_pages_by_scope["public"],
        "wiki_source_pages_private": source_pages_by_scope["private"],
        "wiki_pages_scope_mismatches": scope_mismatches,
        "wiki_pages_without_source_refs": pages_without_refs,
        "wiki_pages_with_unresolved_refs": unresolved_refs,
        "wiki_pages_with_nonlocal_first_source": nonlocal_first_source,
        "wiki_pages_malformed": malformed_page,
        "wiki_pages_unreadable": unreadable_page,
        "wiki_pages_missing_source_hash": missing_hash,
        "wiki_pages_with_stale_first_source": stale,
        "wiki_pages_with_unreadable_first_source": unreadable_source,
        "source_freshness_complete": source_freshness_complete,
        "source_pages_missing_hash": source_missing_hash,
        "source_pages_stale": source_stale,
        "source_pages_unreadable": source_unreadable,
        "source_pages_nonlocal_first_source": source_nonlocal,
        "source_pages_with_current_ingest_receipt": source_ingest_receipt_current,
        "source_pages_missing_current_ingest_receipt": source_missing_ingest_receipt,
        "reference_coverage_complete": coverage,
        "safe_to_retire_raw_index": False,
        "semantic_quality_qualified": False,
        "warning": (
            "Reference coverage, observed source freshness, and successful ingest receipts are "
            "separate checks; none establishes semantic fidelity. This audit never qualifies "
            "raw-index retirement."
        ),
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
