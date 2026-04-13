"""Random-projection 2D viz for embedding vectors. Stdlib-only.

Caches a seeded D_in x 2 gaussian matrix to disk as plain JSON so every
dashboard session projects vectors consistently.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterable

CACHE_PATH = Path.home() / ".claude" / "mcp-skill-hub" / "state" / "projection.json"


def _gen_matrix(dim_in: int, seed: int) -> list[list[float]]:
    rng = random.Random(seed)
    # Box-Muller for gaussian samples (stdlib only).
    mat: list[list[float]] = []
    for _ in range(dim_in):
        row = []
        for _ in range(2):
            u1 = max(1e-9, rng.random())
            u2 = rng.random()
            z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2 * math.pi * u2)
            row.append(z)
        mat.append(row)
    return mat


def get_projection(dim_in: int = 768, seed: int = 42) -> list[list[float]]:
    try:
        if CACHE_PATH.exists():
            data = json.loads(CACHE_PATH.read_text())
            if (isinstance(data, dict) and data.get("dim_in") == dim_in
                    and data.get("seed") == seed
                    and isinstance(data.get("matrix"), list)
                    and len(data["matrix"]) == dim_in):
                return data["matrix"]
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    mat = _gen_matrix(dim_in, seed)
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(
            {"dim_in": dim_in, "seed": seed, "matrix": mat}))
    except OSError:
        pass
    return mat


def project(vec: list[float], matrix: list[list[float]]) -> tuple[float, float]:
    if not vec or not matrix:
        return 0.0, 0.0
    n = min(len(vec), len(matrix))
    x = 0.0
    y = 0.0
    for i in range(n):
        x += vec[i] * matrix[i][0]
        y += vec[i] * matrix[i][1]
    return x, y


def project_all(rows: Iterable[dict[str, Any]],
                matrix: list[list[float]] | None = None) -> list[dict]:
    """Each input row needs 'id', 'vector' (list[float]); optional 'label','group'.

    Returns list of {id, x, y, label, group}.
    """
    mat = matrix or get_projection()
    out: list[dict] = []
    for row in rows:
        vec = row.get("vector")
        if not vec:
            continue
        x, y = project(vec, mat)
        out.append({
            "id": row.get("id"),
            "x": x,
            "y": y,
            "label": row.get("label", ""),
            "group": row.get("group", ""),
        })
    return out
