import numpy as np
import pandas as pd
import pytest

from src.search.service import SearchConfig, SearchService


class DummyIndex:
    def __init__(self, *, size: int, scores: np.ndarray, ids: np.ndarray) -> None:
        self.size = size
        self._scores = scores
        self._ids = ids

        self.calls: list[dict] = []

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        self.calls.append(
            {
                "query_vector": query_vector,
                "top_k": top_k,
            }
        )

        return self._scores, self._ids


class DummyEmbedder:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def encode(
        self,
        *,
        texts: list[str],
        batch_size: int,
        normalize_embeddings: bool,
    ) -> np.ndarray:
        self.calls.append(
            {
                "texts": texts,
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
            }
        )

        return np.array([0.1 * (i + 1) for i in range(len(texts))])


@pytest.fixture
def metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": [0, 1, 2],
            "locode": ["KGFRU", "KZALA", "KGFRU"],
            "search_text": [
                "bishkek kyrgyzstan",
                "almaty kazakhstan",
                "frunze kyrgyzstan",
            ],
        }
    )


@pytest.fixture
def embedder() -> DummyEmbedder:
    return DummyEmbedder()


@pytest.fixture
def build_index():
    def _build_index(
        size: int,
        scores: np.ndarray | None = None,
        ids: np.ndarray | None = None,
    ) -> DummyIndex:
        if scores is None:
            scores = np.array([[0.80, 0.95, 0.60]], dtype=np.float32)

        if ids is None:
            ids = np.array([[0, 2, 1]], dtype=np.int64)

        return DummyIndex(size=size, scores=scores, ids=ids)

    return _build_index


@pytest.fixture
def index(
    metadata: pd.DataFrame,
    build_index,
) -> DummyIndex:
    return build_index(size=len(metadata))


@pytest.fixture
def build_service():
    def _build_service(
        *,
        metadata: pd.DataFrame,
        index: DummyIndex,
        embedder: DummyEmbedder | None = None,
        aggregation_alpha: float = 0.05,
    ) -> SearchService:
        service = SearchService(
            index=index,
            metadata=metadata,
            config=SearchConfig(aggregation_alpha=aggregation_alpha),
            embedder=embedder,
        )

        return service

    return _build_service


@pytest.fixture
def service(
    metadata: pd.DataFrame,
    index: DummyIndex,
    embedder: DummyEmbedder,
    build_service,
):
    return build_service(metadata=metadata, index=index, embedder=embedder)
