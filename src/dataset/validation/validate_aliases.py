import pandas as pd

from src.dataset.validation.schemas import ALIASES_SCHEMA
from src.dataset.validation.validate import (
    ensure_no_empty_strings,
    normalize_string_columns,
    validate_with_pandera,
)


def _ensure_same_locodes(aliases: pd.DataFrame, locations: pd.DataFrame) -> None:
    location_locodes = set(locations["locode"])
    missing_mask = ~aliases["locode"].isin(location_locodes)
    missing_mask = missing_mask.fillna(False)

    if missing_mask.any():
        missing_codes = sorted(aliases.loc[missing_mask, "locode"].dropna().unique())
        preview = missing_codes[:10]
        raise ValueError("aliases contain locodes absent from locations; " f"examples: {preview}")


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
        _ensure_same_locodes(result, locations)

    return result
