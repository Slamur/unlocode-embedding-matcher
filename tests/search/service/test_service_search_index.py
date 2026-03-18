from src.search.model import SearchRequest
from src.search.service import SearchService
from tests.search.service.conftest import DummyIndex


def test_search_index_top_k(
    service: SearchService,
    index: DummyIndex,
) -> None:
    request = SearchRequest(query="city", top_k=2)

    service.search(request)

    assert all(2 == call["top_k"] for call in index.calls)
