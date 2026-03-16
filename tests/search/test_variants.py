from src.search.variants import QueryVariant, build_query_variants


def test_build_query_variants_returns_empty_list_for_empty_input() -> None:
    assert build_query_variants("") == []


def test_build_query_variants_returns_full_query_variant() -> None:
    assert build_query_variants("bishkek kyrgyzstan") == [
        QueryVariant(
            text="bishkek kyrgyzstan",
            kind="full_query",
            weight=1.00,
        ),
        QueryVariant(
            text="location bishkek",
            kind="reverse_city_second",
            weight=1.07,
        ),
        QueryVariant(
            text="location kyrgyzstan",
            kind="forward_city_second",
            weight=1.07,
        ),
        QueryVariant(
            text="location kyrgyzstan country bishkek",
            kind="forward_city_second_country_first",
            weight=1.10,
        ),
        QueryVariant(
            text="location bishkek country kyrgyzstan",
            kind="reverse_city_second_country_first",
            weight=1.10,
        ),
    ]


def test_build_query_variants_removes_tokens_with_digits() -> None:
    assert build_query_variants("samara, novo sadovaya, 1/2") == [
        QueryVariant(
            text="samara, novo sadovaya, 1/2",
            kind="full_query",
            weight=1.00,
        ),
        QueryVariant(
            text="location samara",
            kind="reverse_city_second",
            weight=1.07,
        ),
        QueryVariant(
            text="location novo sadovaya",
            kind="forward_city_second",
            weight=1.07,
        ),
        QueryVariant(
            text="location novo sadovaya country samara",
            kind="forward_city_second_country_first",
            weight=1.10,
        ),
        QueryVariant(
            text="location samara country novo sadovaya",
            kind="reverse_city_second_country_first",
            weight=1.10,
        ),
    ]
