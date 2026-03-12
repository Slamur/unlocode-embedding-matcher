import pandas as pd
import pytest

from src.embeddings.metadata import generate_metadata


def test_generate_metadata_builds_expected_dataframe() -> None:
    search_texts = pd.DataFrame(
        {
            "locode": ["KG FRU", "JP TYO"],
            "search_text": ["bishkek", "tokyo"],
        }
    )

    result = generate_metadata(search_texts)

    assert list(result.columns) == ["row_id", "locode", "search_text"]
    assert result["row_id"].tolist() == [0, 1]
    assert result["locode"].tolist() == ["KG FRU", "JP TYO"]
    assert result["search_text"].tolist() == ["bishkek", "tokyo"]


def test_generate_metadata_raises_on_missing_required_columns() -> None:
    search_texts = pd.DataFrame(
        {
            "locode": ["KG FRU"],
        }
    )

    with pytest.raises(ValueError, match="Missing required columns"):
        generate_metadata(search_texts)


def test_generate_metadata_raises_on_empty_dataframe() -> None:
    search_texts = pd.DataFrame(columns=["locode", "search_text"])

    with pytest.raises(ValueError, match="Input search_texts dataframe is empty"):
        generate_metadata(search_texts)


def test_generate_metadata_raises_on_null_locode() -> None:
    search_texts = pd.DataFrame(
        {
            "locode": [None],
            "search_text": ["bishkek"],
        }
    )

    with pytest.raises(ValueError, match="Column 'locode' contains null values"):
        generate_metadata(search_texts)


def test_generate_metadata_raises_on_null_search_text() -> None:
    search_texts = pd.DataFrame(
        {
            "locode": ["KG FRU"],
            "search_text": [None],
        }
    )

    with pytest.raises(ValueError, match="Column 'search_text' contains null values"):
        generate_metadata(search_texts)


def test_generate_metadata_raises_on_blank_search_text() -> None:
    search_texts = pd.DataFrame(
        {
            "locode": ["KG FRU"],
            "search_text": ["   "],
        }
    )

    with pytest.raises(
        ValueError, match="Column 'search_text' contains empty or whitespace-only values"
    ):
        generate_metadata(search_texts)


def test_generate_metadata_raises_on_non_string_search_text() -> None:
    search_texts = pd.DataFrame(
        {
            "locode": ["KG FRU"],
            "search_text": [123],
        }
    )

    with pytest.raises(ValueError, match="search_text"):
        generate_metadata(search_texts)
