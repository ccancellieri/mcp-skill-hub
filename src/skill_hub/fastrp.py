"""Seeded Gaussian random projection for optional embedding compression.

The public name FastRP is retained from the integration proposal. This is a
linear embedding projection, not graph FastRP or an approximate-neighbor index.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import islice
from typing import Iterable

import numpy as np


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


@lru_cache(maxsize=16)
def _random_matrix(n_rows: int, n_cols: int, seed: int = 42,
                   orthogonal: bool = False) -> np.ndarray:
    _positive_int(n_rows, "n_components")
    _positive_int(n_cols, "input dimension")
    if n_rows > n_cols:
        raise ValueError(f"n_components ({n_rows}) exceeds original dimensionality ({n_cols})")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    rng = np.random.Generator(np.random.PCG64(seed))
    if orthogonal:
        q, _ = np.linalg.qr(rng.standard_normal((n_cols, n_rows)))
        matrix = q.T * np.sqrt(n_cols / n_rows)
    else:
        # E[||Rx||²] = ||x||² requires variance 1 / output dimension.
        matrix = rng.standard_normal((n_rows, n_cols)) / np.sqrt(n_rows)
    matrix = matrix.astype(np.float32)
    matrix.flags.writeable = False
    return matrix


def fast_rp_projection(vectors: np.ndarray, n_components: int,
                       seed: int = 42, orthogonal: bool = False) -> np.ndarray:
    """Project a 1-D vector or 2-D matrix using one deterministic transform."""
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim not in (1, 2):
        raise ValueError("vectors must be a 1-D vector or 2-D matrix")
    if not np.isfinite(array).all():
        raise ValueError("vectors must contain only finite values")
    matrix = _random_matrix(n_components, array.shape[-1], seed, orthogonal)
    return array @ matrix.T


project = fast_rp_projection


def fast_rp_batch(vector_iterable: Iterable, n_components: int, seed: int = 42,
                  orthogonal: bool = False, batch_size: int = 1024) -> np.ndarray:
    """Project bounded input batches with a shared matrix; collect output in RAM."""
    _positive_int(n_components, "n_components")
    _positive_int(batch_size, "batch_size")
    iterator = iter(vector_iterable)
    chunks = []
    input_dim = None
    while batch := list(islice(iterator, batch_size)):
        array = np.asarray(batch, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("each batch item must be a 1-D vector")
        if input_dim is None:
            input_dim = array.shape[1]
        elif array.shape[1] != input_dim:
            raise ValueError("all vectors must have the same input dimension")
        chunks.append(fast_rp_projection(array, n_components, seed, orthogonal))
    return np.concatenate(chunks) if chunks else np.empty((0, n_components), dtype=np.float32)


@dataclass(frozen=True)
class ProjectionSpec:
    """Persisted transform identity; incompatible versions fail closed."""

    input_dim: int
    n_components: int = 128
    seed: int = 42
    orthogonal: bool = False
    version: str = "gaussian-pcg64-f32-v1"

    def __post_init__(self):
        _positive_int(self.input_dim, "input dimension")
        _positive_int(self.n_components, "n_components")
        if self.n_components > self.input_dim:
            raise ValueError("n_components exceeds original dimensionality")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(self.orthogonal, bool):
            raise ValueError("orthogonal must be boolean")
        if self.version != "gaussian-pcg64-f32-v1":
            raise ValueError("unsupported projection version")

    def transform(self, vectors):
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim not in (1, 2) or array.shape[-1] != self.input_dim:
            raise ValueError("projection input dimension mismatch")
        return fast_rp_projection(array, self.n_components, self.seed, self.orthogonal)

    def metadata(self) -> dict:
        from dataclasses import asdict
        return asdict(self)

    def model_identity(self, model: str) -> str:
        return f"{model}|fastrp:{self.version}:{self.input_dim}:{self.n_components}:{self.seed}:{int(self.orthogonal)}"
