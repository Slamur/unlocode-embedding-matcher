import pandas as pd

from src.dataset.validation.schemas import ALIASES_SCHEMA
from src.dataset.validation.validate import (
    _ensure_no_empty_strings,
    _ensure_same_locodes,
    _normalize_string_columns,
    _validate_with_pandera,
)


def validate_aliases(
    aliases: pd.DataFrame,
    locations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = _normalize_string_columns(
        aliases,
        [
            "locode",
            "alias_text",
        ],
    )

    _ensure_no_empty_strings(
        result,
        ["locode", "alias_text"],
        df_name="aliases",
    )

    result = _validate_with_pandera(ALIASES_SCHEMA, result)

    if locations is not None:
        _ensure_same_locodes(result, locations, df_name="aliases")

    return result
