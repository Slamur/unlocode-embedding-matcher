from src.search.variants import build_query_variants


def test_build_query_variants_returns_empty_list_for_empty_input() -> None:
    assert build_query_variants("") == []


def test_build_query_variants_returns_normalized_query_as_single_variant() -> None:
    assert build_query_variants("bishkek kyrgyzstan") == ["bishkek kyrgyzstan"]
