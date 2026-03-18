from src.search.model import SearchRequest
from src.search.service import SearchService
from tests.search.service.conftest import DummyEmbedder


def test_search_embedder_parameters(
    service: SearchService,
    embedder: DummyEmbedder,
):
    request = SearchRequest(query="Bishkek, Kyrgyzstan!", top_k=2)

    service.search(request)

    assert len(embedder.calls) == 1

    assert all(call["batch_size"] == service._config.batch_size for call in embedder.calls)

    assert all(call["normalize_embeddings"] is False for call in embedder.calls)


def test_search_embeds_all_variant_one_word(
    service: SearchService,
    embedder: DummyEmbedder,
):
    request = SearchRequest(query="Bishkek!", top_k=2)

    service.search(request)

    assert all(
        call["texts"]
        == [
            "bishkek",
            "location bishkek",
        ]
        for call in embedder.calls
    )


def test_search_embeds_all_variant_two_words(
    service: SearchService,
    embedder: DummyEmbedder,
):
    request = SearchRequest(query="Bishkek, Kyrgyzstan!", top_k=2)

    service.search(request)

    assert all(
        call["texts"]
        == [
            "bishkek kyrgyzstan",
            "location bishkek",
            "location kyrgyzstan",
            "location kyrgyzstan country bishkek",
            "location bishkek country kyrgyzstan",
        ]
        for call in embedder.calls
    )
