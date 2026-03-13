from dataclasses import dataclass

import numpy as np

from src.index.model import VectorIndex


@dataclass(frozen=True)
class IndexBuildResult:
    vector_count: int
    dimension: int


def build_index(embeddings: np.ndarray) -> tuple[VectorIndex, IndexBuildResult]:
    index = VectorIndex(dimension=embeddings.shape[1])
    index.add(embeddings)

    return index, IndexBuildResult(
        vector_count=embeddings.shape[0],
        dimension=embeddings.shape[1],
    )
