from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.utils.files import ensure_parent_dir_exists, require_file_exists


@dataclass(frozen=True)
class IndexConfig:
    metric: str = "cosine"


class VectorIndex:
    def __init__(self, dimension: int, config: IndexConfig | None = None) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required to use VectorIndex. " "Install indexing dependencies first."
            ) from exc

        self._faiss = faiss
        self._config = config or IndexConfig()

        if dimension <= 0:
            raise ValueError("dimension must be positive")

        if self._config.metric != "cosine":
            raise ValueError(f"Unsupported metric: {self._config.metric}")

        self._index = faiss.IndexFlatIP(dimension)

    def _validate_embeddings(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        if embeddings.shape[0] == 0:
            raise ValueError("Embeddings must not be empty")
        if embeddings.shape[1] == 0:
            raise ValueError("Embeddings must have positive dimension")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} "
                f"does not match index dimension {self.dimension}"
            )

    def _prepare_embeddings(
        self,
        embeddings: np.ndarray,
        *,
        allow_single_vector: bool = False,
    ) -> np.ndarray:
        vectors = np.asarray(embeddings, dtype=np.float32).copy()

        if allow_single_vector and vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        self._validate_embeddings(vectors)

        self._faiss.normalize_L2(vectors)
        return vectors

    def add(self, embeddings: np.ndarray) -> None:
        vectors = self._prepare_embeddings(embeddings)
        self._index.add(vectors)

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        vectors = self._prepare_embeddings(query, allow_single_vector=True)
        scores, ids = self._index.search(vectors, top_k)
        return scores, ids

    def save(self, path: Path) -> None:
        ensure_parent_dir_exists(path=path)
        self._faiss.write_index(self._index, str(path))

    @classmethod
    def load(cls, path: Path) -> "VectorIndex":
        try:
            import faiss
        except ImportError as exc:
            raise ImportError(
                "faiss-cpu is required to use VectorIndex. " "Install indexing dependencies first."
            ) from exc

        require_file_exists(path=path)

        index = faiss.read_index(str(path))

        obj = cls.__new__(cls)
        obj._faiss = faiss
        obj._config = IndexConfig()
        obj._index = index
        return obj

    @property
    def size(self) -> int:
        return self._index.ntotal

    @property
    def dimension(self) -> int:
        return self._index.d
