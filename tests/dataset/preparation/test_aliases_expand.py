import pandas as pd

from src.dataset.preparation.aliases_expand import (
    _split_parenthesized_name_with_labels,
    build_expanded_rows,
)


def test_split_parenthesized_name_with_labels_without_parentheses() -> None:
    result = _split_parenthesized_name_with_labels("Bishkek")

    assert result == [("Bishkek", "full")]


def test_split_parenthesized_name_with_labels_with_parentheses() -> None:
    result = _split_parenthesized_name_with_labels("Brussel (Bruxelles)")

    assert result == [
        ("Brussel (Bruxelles)", "full"),
        ("Brussel", "paren_left"),
        ("Bruxelles", "paren_inner"),
    ]


def test_split_parenthesized_name_with_labels_strips_outer_spaces() -> None:
    result = _split_parenthesized_name_with_labels("  Foo (Bar)  ")

    assert result == [
        ("Foo (Bar)", "full"),
        ("Foo", "paren_left"),
        ("Bar", "paren_inner"),
    ]


def test_split_parenthesized_name_with_labels_strips_inner_spaces() -> None:
    result = _split_parenthesized_name_with_labels("Foo   (  Bar  )")

    assert result == [
        ("Foo   (  Bar  )", "full"),
        ("Foo", "paren_left"),
        ("Bar", "paren_inner"),
    ]


def test_split_parenthesized_name_with_labels_empty_string_returns_empty_list() -> None:
    result = _split_parenthesized_name_with_labels("   ")

    assert result == []


def test_split_parenthesized_name_with_labels_unbalanced_parentheses_keeps_original_only() -> None:
    result = _split_parenthesized_name_with_labels("Foo (Bar")

    assert result == [("Foo (Bar", "full")]


def test_build_expanded_rows_expands_name_and_name_wo_diacritics() -> None:
    aliases = pd.DataFrame(
        [
            ("BEBRU", "Brussel (Bruxelles)", "Brussel (Bruxelles)"),
        ],
        columns=["locode", "name", "name_wo_diacritics"],
    )

    result = build_expanded_rows(aliases)

    assert result == [
        ("BEBRU", "name", "Brussel (Bruxelles)", "full"),
        ("BEBRU", "name", "Brussel", "paren_left"),
        ("BEBRU", "name", "Bruxelles", "paren_inner"),
        ("BEBRU", "name_wo_diacritics", "Brussel (Bruxelles)", "full"),
        ("BEBRU", "name_wo_diacritics", "Brussel", "paren_left"),
        ("BEBRU", "name_wo_diacritics", "Bruxelles", "paren_inner"),
    ]


def test_build_expanded_rows_keeps_non_parenthesized_values_as_full_only() -> None:
    aliases = pd.DataFrame(
        [
            ("KGFRU", "Frunze", "Frunze"),
        ],
        columns=["locode", "name", "name_wo_diacritics"],
    )

    result = build_expanded_rows(aliases)

    assert result == [
        ("KGFRU", "name", "Frunze", "full"),
        ("KGFRU", "name_wo_diacritics", "Frunze", "full"),
    ]
