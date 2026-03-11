import pandas as pd

from src.dataset.validation.schemas import SEARCH_TEXTS_SCHEMA
from src.dataset.validation.validate import (
    ensure_no_empty_strings,
    ensure_same_locodes,
    normalize_string_columns,
    validate_with_pandera,
)


def validate_search_texts(
    search_texts: pd.DataFrame,
    locations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = normalize_string_columns(
        search_texts,
        [
            "locode",
            "alias_text",
            "country",
            "subdivision_name",
            "search_text_kind",
            "search_text",
        ],
    )

    ensure_no_empty_strings(
        result,
        [
            "locode",
            "alias_text",
            "country",
            "search_text_kind",
            "search_text",
        ],
        df_name="search_texts",
    )

    result = validate_with_pandera(SEARCH_TEXTS_SCHEMA, result)

    if locations is not None:
        ensure_same_locodes(result, locations, df_name="search_texts")

    return result
