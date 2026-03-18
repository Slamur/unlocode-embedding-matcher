import numpy as np
import pandas as pd
import pytest

from src.search.model import SearchRequest
from src.search.service import SearchService
from tests.search.service.conftest import DummyEmbedder


def test_search_raises_for_blank_query(
    service: SearchService,
) -> None:
    request = SearchRequest(query="   ", top_k=5)

    with pytest.raises(ValueError, match="Query must not be empty"):
        service.search(request)


def test_search_returns_empty_hits_for_query_with_empty_normalized_variant(
    service: SearchService,
    embedder: DummyEmbedder,
) -> None:
    request = SearchRequest(query="!!!", top_k=5)

    response = service.search(request)

    assert embedder.calls == []

    assert response.query == "!!!"
    assert response.normalized_query == ""
    assert response.hits == []


def test_search_ignores_negative_ids(
    metadata: pd.DataFrame,
    build_index,
    build_service,
) -> None:
    index = build_index(
        size=len(metadata),
        scores=np.array([[0.95, 0.70]], dtype=np.float32),
        ids=np.array([[-1, 1]], dtype=np.int64),
    )

    service = build_service(
        metadata=metadata,
        index=index,
    )

    request = SearchRequest(query="almaty", top_k=2)

    response = service.search(request)

    assert len(response.hits) == 1
    assert response.hits[0].locode == "KZALA"
