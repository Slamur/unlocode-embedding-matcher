from src.search.variants import QueryVariant, build_query_variants


def test_build_query_variants_returns_empty_list_for_empty_input() -> None:
    assert build_query_variants("") == []


def test_build_query_variants_returns_full_query_variant() -> None:
    assert build_query_variants("bishkek kyrgyzstan manas") == [
        QueryVariant(
            text="bishkek kyrgyzstan manas",
            kind="full_query",
            weight=1.00,
        ),
        QueryVariant(
            text="location bishkek",
            kind="forward_city_first",
            weight=1.03,
        ),
        QueryVariant(
            text="location manas",
            kind="reverse_city_first",
            weight=1.03,
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
            kind="forward_city_first_country_second",
            weight=1.10,
        ),
        QueryVariant(
            text="location kyrgyzstan country manas",
            kind="reverse_city_second_country_first",
            weight=1.10,
        ),
        QueryVariant(
            text="location manas country kyrgyzstan",
            kind="reverse_city_first_country_second",
            weight=1.10,
        ),
    ]


def test_build_query_variants_removes_tokens_with_digits() -> None:
    assert build_query_variants("samara, novo sadovaya, 1/2, naba") == [
        QueryVariant(
            text="samara, novo sadovaya, 1/2, naba",
            kind="full_query",
            weight=1.00,
        ),
        QueryVariant(
            text="location samara",
            kind="forward_city_first",
            weight=1.03,
        ),
        QueryVariant(
            text="location naba",
            kind="reverse_city_first",
            weight=1.03,
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
            kind="forward_city_first_country_second",
            weight=1.10,
        ),
        QueryVariant(
            text="location novo sadovaya country naba",
            kind="reverse_city_second_country_first",
            weight=1.10,
        ),
        QueryVariant(
            text="location naba country novo sadovaya",
            kind="reverse_city_first_country_second",
            weight=1.10,
        ),
    ]
