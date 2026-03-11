import pandas as pd

from src.dataset.preparation.search_texts import (
    _build_search_text_rows,
    _build_search_text_variants,
    _prepare_search_texts,
)


def test_build_search_text_variants_without_subdivision() -> None:
    result = _build_search_text_variants(
        alias_text="Bishkek",
        country="KG",
        subdivision_name="",
    )

    assert result == [
        ("alias_only", "Bishkek"),
        ("alias_country", "Bishkek KG"),
        ("structured", "location Bishkek country KG"),
        ("alias_only_ascii_fold", "Bishkek"),
        ("alias_country_ascii_fold", "Bishkek KG"),
        ("structured_ascii_fold", "location Bishkek country KG"),
    ]


def test_build_search_text_variants_with_subdivision() -> None:
    result = _build_search_text_variants(
        alias_text="Brussels",
        country="BE",
        subdivision_name="Brussels-Capital Region",
    )

    assert result == [
        ("alias_only", "Brussels"),
        ("alias_country", "Brussels BE"),
        ("structured", "location Brussels country BE"),
        (
            "alias_subdivision_country",
            "Brussels Brussels-Capital Region BE",
        ),
        (
            "structured_with_subdivision",
            "location Brussels subdivision Brussels-Capital Region country BE",
        ),
        ("alias_only_ascii_fold", "Brussels"),
        ("alias_country_ascii_fold", "Brussels BE"),
        ("structured_ascii_fold", "location Brussels country BE"),
        (
            "alias_subdivision_country_ascii_fold",
            "Brussels Brussels-Capital Region BE",
        ),
        (
            "structured_with_subdivision_ascii_fold",
            "location Brussels subdivision Brussels-Capital Region country BE",
        ),
    ]


def test_build_search_text_variants_ascii_fold_adds_folded_versions() -> None:
    result = _build_search_text_variants(
        alias_text="São Paulo",
        country="BR",
        subdivision_name="",
    )

    assert ("alias_only", "São Paulo") in result
    assert ("alias_only_ascii_fold", "Sao Paulo") in result
    assert ("alias_country_ascii_fold", "Sao Paulo BR") in result
    assert (
        "structured_ascii_fold",
        "location Sao Paulo country BR",
    ) in result


def test_build_search_text_variants_no_subdivision_variants_if_subdivision_empty() -> None:
    result = _build_search_text_variants(
        alias_text="Bishkek",
        country="KG",
        subdivision_name="",
    )

    kinds = [kind for kind, _ in result]

    assert "alias_subdivision_country" not in kinds
    assert "structured_with_subdivision" not in kinds
    assert "alias_subdivision_country_ascii_fold" not in kinds
    assert "structured_with_subdivision_ascii_fold" not in kinds


def test_build_search_text_rows_normalizes_generated_texts() -> None:
    aliases_with_locations = pd.DataFrame(
        [
            ("BRSSZ", "São Paulo", "BR", ""),
        ],
        columns=["locode", "alias_text", "country", "subdivision_name"],
    )

    result = _build_search_text_rows(aliases_with_locations)

    assert result == [
        ("BRSSZ", "São Paulo", "BR", "", "alias_only", "são paulo"),
        ("BRSSZ", "São Paulo", "BR", "", "alias_country", "são paulo br"),
        (
            "BRSSZ",
            "São Paulo",
            "BR",
            "",
            "structured",
            "location são paulo country br",
        ),
        ("BRSSZ", "São Paulo", "BR", "", "alias_only_ascii_fold", "sao paulo"),
        (
            "BRSSZ",
            "São Paulo",
            "BR",
            "",
            "alias_country_ascii_fold",
            "sao paulo br",
        ),
        (
            "BRSSZ",
            "São Paulo",
            "BR",
            "",
            "structured_ascii_fold",
            "location sao paulo country br",
        ),
    ]


def test_prepare_search_texts_trims_filters_and_deduplicates() -> None:
    search_texts = pd.DataFrame(
        [
            ("BRSSZ", " São Paulo ", " BR ", "", " alias_only ", " são paulo "),
            ("BRSSZ", "Sao Paulo", "BR", "", "alias_only_ascii_fold", "sao paulo"),
            ("BRSSZ", "Another Alias", "BR", "", "alias_country", "   "),
            ("BRSSZ", "Duplicate Alias", "BR", "", "structured", "são paulo"),
        ],
        columns=[
            "locode",
            "alias_text",
            "country",
            "subdivision_name",
            "search_text_kind",
            "search_text",
        ],
    )

    result = _prepare_search_texts(search_texts)

    assert result.to_dict(orient="records") == [
        {
            "locode": "BRSSZ",
            "alias_text": "São Paulo",
            "country": "BR",
            "subdivision_name": "",
            "search_text_kind": "alias_only",
            "search_text": "são paulo",
        },
        {
            "locode": "BRSSZ",
            "alias_text": "Sao Paulo",
            "country": "BR",
            "subdivision_name": "",
            "search_text_kind": "alias_only_ascii_fold",
            "search_text": "sao paulo",
        },
    ]
