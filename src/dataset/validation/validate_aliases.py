import pandas as pd

from src.dataset.validation.schemas import ALIASES_SCHEMA
from src.dataset.validation.validate import (
    ensure_no_empty_strings,
    ensure_same_locodes,
    normalize_string_columns,
    validate_with_pandera,
)


def validate_aliases(
    aliases: pd.DataFrame,
    locations: pd.DataFrame | None = None,
) -> pd.DataFrame:
    result = normalize_string_columns(
        aliases,
        [
            "locode",
            "alias_text",
        ],
    )

    ensure_no_empty_strings(
        result,
        ["locode", "alias_text"],
        df_name="aliases",
    )

    result = validate_with_pandera(ALIASES_SCHEMA, result)

    if locations is not None:
        ensure_same_locodes(result, locations, df_name="aliases")

    return result
