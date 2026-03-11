import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.dataset.validation.validate_search_texts import validate_search_texts


def test_validate_search_texts_accepts_valid_dataframe() -> None:
    validated_locations = pd.DataFrame(
        {
            "locode": ["KGFRU", "BEBRU"],
            "country": ["KG", "BE"],
            "code": ["FRU", "BRU"],
            "subdivision_code": [pd.NA, "BRU"],
            "subdivision_name": [pd.NA, "Brussels-Capital Region"],
        }
    )

    search_texts = pd.DataFrame(
        {
            "locode": ["KGFRU", "BEBRU", "BEBRU"],
            "alias_text": ["Bishkek", "Brussel", "Bruxelles"],
            "country": ["KG", "BE", "BE"],
            "subdivision_name": [pd.NA, "Brussels-Capital Region", "Brussels-Capital Region"],
            "search_text_kind": [
                "alias_only",
                "alias_country",
                "structured_with_subdivision",
            ],
            "search_text": [
                "bishkek",
                "brussel be",
                "location bruxelles subdivision brussels capital region country be",
            ],
        }
    )

    result = validate_search_texts(search_texts, validated_locations)

    assert list(result["search_text"]) == [
        "bishkek",
        "brussel be",
        "location bruxelles subdivision brussels capital region country be",
    ]


def test_validate_search_texts_trims_string_values() -> None:
    validated_locations = pd.DataFrame(
        {
            "locode": ["KGFRU"],
            "country": ["KG"],
            "code": ["FRU"],
            "subdivision_code": [pd.NA],
            "subdivision_name": [pd.NA],
        }
    )

    search_texts = pd.DataFrame(
        {
            "locode": [" KGFRU "],
            "alias_text": [" Bishkek "],
            "country": [" KG "],
            "subdivision_name": ["  "],
            "search_text_kind": [" alias_only "],
            "search_text": [" bishkek "],
        }
    )

    result = validate_search_texts(search_texts, validated_locations)

    assert result.iloc[0]["locode"] == "KGFRU"
    assert result.iloc[0]["alias_text"] == "Bishkek"
    assert result.iloc[0]["country"] == "KG"
    assert result.iloc[0]["subdivision_name"] == ""
    assert result.iloc[0]["search_text_kind"] == "alias_only"
    assert result.iloc[0]["search_text"] == "bishkek"


def test_validate_search_texts_rejects_empty_search_text_after_trim() -> None:
    search_texts = pd.DataFrame(
        {
            "locode": ["KGFRU"],
            "alias_text": ["Bishkek"],
            "country": ["KG"],
            "subdivision_name": [pd.NA],
            "search_text_kind": ["alias_only"],
            "search_text": [" "],
        }
    )

    with pytest.raises(
        ValueError,
        match="search_texts.search_text contains 1 empty value",
    ):
        validate_search_texts(search_texts)


def test_validate_search_texts_rejects_empty_search_text_kind_after_trim() -> None:
    search_texts = pd.DataFrame(
        {
            "locode": ["KGFRU"],
            "alias_text": ["Bishkek"],
            "country": ["KG"],
            "subdivision_name": [pd.NA],
            "search_text_kind": [" "],
            "search_text": ["bishkek"],
        }
    )

    with pytest.raises(
        ValueError,
        match="search_texts.search_text_kind contains 1 empty value",
    ):
        validate_search_texts(search_texts)


def test_validate_search_texts_rejects_unknown_locode() -> None:
    validated_locations = pd.DataFrame(
        {
            "locode": ["KGFRU"],
            "country": ["KG"],
            "code": ["FRU"],
            "subdivision_code": [pd.NA],
            "subdivision_name": [pd.NA],
        }
    )

    search_texts = pd.DataFrame(
        {
            "locode": ["KGFRU", "USNYC"],
            "alias_text": ["Bishkek", "New York"],
            "country": ["KG", "US"],
            "subdivision_name": [pd.NA, "New York"],
            "search_text_kind": ["alias_only", "alias_country"],
            "search_text": ["bishkek", "new york us"],
        }
    )

    with pytest.raises(
        ValueError,
        match="search_texts contain locodes absent from locations",
    ):
        validate_search_texts(search_texts, validated_locations)


def test_validate_search_texts_rejects_duplicate_search_text_per_locode() -> None:
    validated_locations = pd.DataFrame(
        {
            "locode": ["KGFRU"],
            "country": ["KG"],
            "code": ["FRU"],
            "subdivision_code": [pd.NA],
            "subdivision_name": [pd.NA],
        }
    )

    search_texts = pd.DataFrame(
        {
            "locode": ["KGFRU", "KGFRU"],
            "alias_text": ["Bishkek", "Bishkek city"],
            "country": ["KG", "KG"],
            "subdivision_name": [pd.NA, pd.NA],
            "search_text_kind": ["alias_only", "structured"],
            "search_text": ["bishkek", "bishkek"],
        }
    )

    with pytest.raises(SchemaErrors):
        validate_search_texts(search_texts, validated_locations)


def test_validate_search_texts_rejects_invalid_locode_format() -> None:
    search_texts = pd.DataFrame(
        {
            "locode": ["KG12"],
            "alias_text": ["Bishkek"],
            "country": ["KG"],
            "subdivision_name": [pd.NA],
            "search_text_kind": ["alias_only"],
            "search_text": ["bishkek"],
        }
    )

    with pytest.raises(SchemaErrors):
        validate_search_texts(search_texts)


def test_validate_search_texts_rejects_empty_alias_text_after_trim() -> None:
    search_texts = pd.DataFrame(
        {
            "locode": ["KGFRU"],
            "alias_text": [" "],
            "country": ["KG"],
            "subdivision_name": [pd.NA],
            "search_text_kind": ["alias_only"],
            "search_text": ["bishkek"],
        }
    )

    with pytest.raises(
        ValueError,
        match="search_texts.alias_text contains 1 empty value",
    ):
        validate_search_texts(search_texts)


def test_validate_search_texts_rejects_empty_country_after_trim() -> None:
    search_texts = pd.DataFrame(
        {
            "locode": ["KGFRU"],
            "alias_text": ["Bishkek"],
            "country": [" "],
            "subdivision_name": [pd.NA],
            "search_text_kind": ["alias_only"],
            "search_text": ["bishkek"],
        }
    )

    with pytest.raises(
        ValueError,
        match="search_texts.country contains 1 empty value",
    ):
        validate_search_texts(search_texts)
