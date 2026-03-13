from dataclasses import dataclass

import faiss
import numpy as np


@dataclass(frozen=True)
class IndexBuildResult:
    vector_count: int
    dimension: int


def _validate_embeddings(embeddings: np.ndarray) -> None:
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2D array")

    if embeddings.shape[0] == 0:
        raise ValueError("Embeddings must not be empty")


def _prepare_embeddings_for_index(embeddings: np.ndarray) -> np.ndarray:
    _validate_embeddings(embeddings)

    prepared = np.asarray(embeddings, dtype=np.float32).copy()
    faiss.normalize_L2(prepared)

    return prepared


def build_index(embeddings: np.ndarray) -> tuple[faiss.Index, IndexBuildResult]:
    prepared = _prepare_embeddings_for_index(embeddings)
    dimension = prepared.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(prepared)

    result = IndexBuildResult(
        vector_count=prepared.shape[0],
        dimension=dimension,
    )

    return index, result
