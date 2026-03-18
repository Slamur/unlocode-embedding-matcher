from src.search.model import SearchRequest
from src.search.service import SearchService


def test_search_response_queries(
    service: SearchService,
) -> None:
    response = service.search(SearchRequest(query="Bishkek, Kyrgyzstan!", top_k=2))

    assert response.query == "Bishkek, Kyrgyzstan!"
    assert response.normalized_query == "bishkek kyrgyzstan"


def test_search_response_hits(
    service: SearchService,
) -> None:
    response = service.search(SearchRequest(query="Bishkek, Kyrgyzstan!", top_k=2))

    assert len(response.hits) == 2

    assert response.hits[0].row_id == 2
    assert response.hits[0].locode == "KGFRU"
    assert response.hits[0].search_text == "frunze kyrgyzstan"

    # scores are tested separately
    # assert response.hits[0].score == pytest.approx(expected_kgfru_score)

    assert response.hits[1].row_id == 1
    assert response.hits[1].locode == "KZALA"
    assert response.hits[1].search_text == "almaty kazakhstan"

    # scores are tested separately
    # assert response.hits[1].score == pytest.approx(expected_kzala_score)


def test_search_response_top_k(
    service: SearchService,
) -> None:
    response = service.search(SearchRequest(query="Bishkek, Kyrgyzstan!", top_k=1))

    assert len(response.hits) == 1
