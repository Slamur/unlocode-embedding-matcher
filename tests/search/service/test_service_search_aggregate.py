import pytest

from src.search.model import SearchRequest
from src.search.service import SearchService


def test_search_aggregation_keeps_best_hit_metadata(
    service: SearchService,
) -> None:
    request = SearchRequest(query="frunze", top_k=5)

    response = service.search(request)

    expected_variants_count = 2

    expected_best_kgfru_weighted_score = 0.95 * 1.03
    expected_kgfru_hits = 2

    expected_kgfru_score = (
        expected_best_kgfru_weighted_score
        + service._config.aggregation_alpha * expected_kgfru_hits * expected_variants_count
    )
    assert response.hits[0].score == pytest.approx(expected_kgfru_score)

    expected_best_kzala_weighted_score = 0.60 * 1.03
    expected_kzala_hits = 1

    expected_kzala_score = (
        expected_best_kzala_weighted_score
        + service._config.aggregation_alpha * expected_kzala_hits * expected_variants_count
    )
    assert response.hits[1].score == pytest.approx(expected_kzala_score)
