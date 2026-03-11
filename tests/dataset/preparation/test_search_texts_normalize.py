import pytest

from src.dataset.preparation.search_texts import _normalize_search_text


@pytest.mark.parametrize(
    ("raw_text", "expected"),
    [
        ("Bishkek", "bishkek"),
        ("  Bishkek  ", "bishkek"),
        ("New   York", "new york"),
        ("Bishkek, KG", "bishkek kg"),
        ("St. Petersburg", "st petersburg"),
        ("foo/bar", "foo bar"),
        ("foo-bar", "foo bar"),
        ("foo_bar", "foo_bar"),  # underscore are saved because of \\w
        ("Brussel (Bruxelles)", "brussel bruxelles"),
        ("Санкт-Петербург", "санкт петербург"),
        ("München", "münchen"),
        ("São Paulo", "são paulo"),
        ("Aşgabat", "aşgabat"),
        ("北京市", "北京市"),
        ("東京", "東京"),
        ("Airport 2", "airport 2"),
        ("  , . / - ( )  ", ""),
        ("", ""),
    ],
)
def test_normalize_search_text(raw_text: str, expected: str) -> None:
    assert _normalize_search_text(raw_text) == expected


def test_normalize_search_text_collapses_multiple_separators_to_single_space() -> None:
    assert _normalize_search_text("foo, - / bar") == "foo bar"


def test_normalize_search_text_is_idempotent() -> None:
    raw_text = "  Brussel (Bruxelles), BE  "

    normalized_once = _normalize_search_text(raw_text)
    normalized_twice = _normalize_search_text(normalized_once)

    assert normalized_twice == normalized_once
