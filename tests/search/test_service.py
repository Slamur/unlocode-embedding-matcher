import numpy as np
import pandas as pd
import pytest

from src.search.model import SearchRequest
from src.search.service import SearchService


class DummyIndex:
    def __init__(self, *, size: int, scores: np.ndarray, ids: np.ndarray) -> None:
        self.size = size
        self._scores = scores
        self._ids = ids

    def search(self, query_vector: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        return self._scores, self._ids


class DummyEmbedder:
    def __init__(self, embeddings: np.ndarray) -> None:
        self._embeddings = embeddings
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
        return self._embeddings


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


def _build_service(
    monkeypatch: pytest.MonkeyPatch,
    *,
    metadata: pd.DataFrame,
    scores: np.ndarray,
    ids: np.ndarray,
    query_embeddings: np.ndarray | None = None,
) -> tuple[SearchService, DummyEmbedder]:
    index = DummyIndex(size=len(metadata), scores=scores, ids=ids)

    embedder = DummyEmbedder(
        embeddings=(
            query_embeddings
            if query_embeddings is not None
            else np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
        )
    )

    monkeypatch.setattr(
        "src.search.service.TextEmbedder",
        lambda config: embedder,
    )

    service = SearchService(index=index, metadata=metadata)
    return service, embedder


def test_init_raises_if_metadata_missing_required_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = pd.DataFrame(
        {
            "row_id": [0],
            "locode": ["KGFRU"],
        }
    )
    index = DummyIndex(
        size=1,
        scores=np.array([[0.9]], dtype=np.float32),
        ids=np.array([[0]], dtype=np.int64),
    )

    monkeypatch.setattr(
        "src.search.service.TextEmbedder",
        lambda config: DummyEmbedder(np.array([[0.1, 0.2]], dtype=np.float32)),
    )

    with pytest.raises(ValueError, match="Metadata is missing required columns: search_text"):
        SearchService(index=index, metadata=metadata)


def test_init_raises_if_metadata_size_does_not_match_index_size(
    monkeypatch: pytest.MonkeyPatch,
    metadata: pd.DataFrame,
) -> None:
    index = DummyIndex(
        size=len(metadata) + 1,
        scores=np.array([[0.9]], dtype=np.float32),
        ids=np.array([[0]], dtype=np.int64),
    )

    monkeypatch.setattr(
        "src.search.service.TextEmbedder",
        lambda config: DummyEmbedder(np.array([[0.1, 0.2]], dtype=np.float32)),
    )

    with pytest.raises(ValueError, match="does not match index size"):
        SearchService(index=index, metadata=metadata)


def test_search_raises_for_blank_query(
    monkeypatch: pytest.MonkeyPatch,
    metadata: pd.DataFrame,
) -> None:
    service, _ = _build_service(
        monkeypatch,
        metadata=metadata,
        scores=np.array([[0.9]], dtype=np.float32),
        ids=np.array([[0]], dtype=np.int64),
    )

    with pytest.raises(ValueError, match="Query must not be empty"):
        service.search(SearchRequest(query="   ", top_k=5))


def test_search_returns_empty_hits_for_query_with_empty_normalized_variant(
    monkeypatch: pytest.MonkeyPatch,
    metadata: pd.DataFrame,
) -> None:
    service, embedder = _build_service(
        monkeypatch,
        metadata=metadata,
        scores=np.array([[0.9]], dtype=np.float32),
        ids=np.array([[0]], dtype=np.int64),
    )

    response = service.search(SearchRequest(query="!!!", top_k=5))

    assert response.query == "!!!"
    assert response.normalized_query == ""
    assert response.hits == []
    assert embedder.calls == []


def test_search_returns_hits(
    monkeypatch: pytest.MonkeyPatch,
    metadata: pd.DataFrame,
) -> None:
    service, embedder = _build_service(
        monkeypatch,
        metadata=metadata,
        scores=np.array([[0.95, 0.80]], dtype=np.float32),
        ids=np.array([[0, 1]], dtype=np.int64),
    )

    response = service.search(SearchRequest(query="Bishkek, Kyrgyzstan!", top_k=2))

    assert response.query == "Bishkek, Kyrgyzstan!"
    assert response.normalized_query == "bishkek kyrgyzstan"
    assert len(response.hits) == 2

    assert response.hits[0].row_id == 0
    assert response.hits[0].locode == "KGFRU"
    assert response.hits[0].score == pytest.approx(0.95)
    assert response.hits[0].search_text == "bishkek kyrgyzstan"

    assert embedder.calls == [
        {
            "texts": ["bishkek kyrgyzstan"],
            "batch_size": service._config.batch_size,
            "normalize_embeddings": False,
        }
    ]


def test_search_ignores_negative_ids(
    monkeypatch: pytest.MonkeyPatch,
    metadata: pd.DataFrame,
) -> None:
    service, _ = _build_service(
        monkeypatch,
        metadata=metadata,
        scores=np.array([[0.95, 0.70]], dtype=np.float32),
        ids=np.array([[-1, 1]], dtype=np.int64),
    )

    response = service.search(SearchRequest(query="almaty", top_k=2))

    assert len(response.hits) == 1
    assert response.hits[0].locode == "KZALA"


def test_search_deduplicates_by_locode_and_keeps_best_score(
    monkeypatch: pytest.MonkeyPatch,
    metadata: pd.DataFrame,
) -> None:
    service, _ = _build_service(
        monkeypatch,
        metadata=metadata,
        scores=np.array([[0.80, 0.95, 0.60]], dtype=np.float32),
        ids=np.array([[0, 2, 1]], dtype=np.int64),
    )

    response = service.search(SearchRequest(query="frunze", top_k=5))

    assert [hit.locode for hit in response.hits] == ["KGFRU", "KZALA"]
    assert response.hits[0].row_id == 2
    assert response.hits[0].score == pytest.approx(0.95)
    assert response.hits[0].search_text == "frunze kyrgyzstan"


def test_search_respects_top_k_after_deduplication(
    monkeypatch: pytest.MonkeyPatch,
    metadata: pd.DataFrame,
) -> None:
    service, _ = _build_service(
        monkeypatch,
        metadata=metadata,
        scores=np.array([[0.99, 0.98, 0.97]], dtype=np.float32),
        ids=np.array([[0, 1, 2]], dtype=np.int64),
    )

    response = service.search(SearchRequest(query="city", top_k=1))

    assert len(response.hits) == 1
    assert response.hits[0].locode == "KGFRU"
