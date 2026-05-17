#!/usr/bin/env bash
# Idempotent: create or update every label in .github/labels.yml on the current repo's GitHub remote.
# Requires: gh (authenticated), python3 with PyYAML (or stdlib tomllib + a JSON dump fallback).
set -euo pipefail

REPO="${REPO:-ccancellieri/mcp-skill-hub}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LABELS_FILE="$HERE/../.github/labels.yml"

# Use Python to parse the YAML (no yq dependency) and emit "name<TAB>color<TAB>description" lines.
python3 - "$LABELS_FILE" <<'PY' | while IFS=$'\t' read -r name color description; do
import sys, re

# Minimal YAML parser — fine for our flat list of dicts.
src = open(sys.argv[1]).read()
items, current = [], None
for raw in src.splitlines():
    if not raw.strip() or raw.lstrip().startswith("#"):
        continue
    if raw.startswith("- "):
        if current is not None:
            items.append(current)
        current = {}
        raw = raw[2:]
    m = re.match(r'\s*([A-Za-z_:-]+):\s*(.+)\s*$', raw)
    if m:
        key, val = m.group(1), m.group(2).strip()
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
        current[key] = val
if current is not None:
    items.append(current)
for it in items:
    print(f"{it['name']}\t{it['color']}\t{it.get('description','')}")
PY
  printf "  %-25s " "$name"
  if gh label create "$name" --repo "$REPO" --color "$color" --description "$description" --force >/dev/null 2>&1; then
    echo "ok"
  else
    echo "FAILED"
  fi
done

echo "Done."
