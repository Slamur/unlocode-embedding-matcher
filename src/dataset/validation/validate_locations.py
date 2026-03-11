import pandas as pd

from src.dataset.validation.schemas import LOCATIONS_SCHEMA
from src.dataset.validation.validate import (
    _ensure_no_empty_strings,
    _normalize_string_columns,
    _validate_with_pandera,
)


def _ensure_valid_locodes(locations: pd.DataFrame, df_name: str) -> None:
    expected_locode = locations["country"] + locations["code"]
    invalid_mask = locations["locode"] != expected_locode
    invalid_mask = invalid_mask.fillna(False)

    if invalid_mask.any():
        invalid_rows = locations.loc[invalid_mask, ["locode", "country", "code"]].head(10)
        preview = invalid_rows.to_dict(orient="records")
        raise ValueError(
            f"{df_name}.locode must be equal to country + code; " f"examples: {preview}"
        )


def validate_locations(locations: pd.DataFrame) -> pd.DataFrame:
    result = _normalize_string_columns(
        locations,
        [
            "locode",
            "country",
            "code",
            "subdivision_code",
            "subdivision_name",
        ],
    )

    _ensure_no_empty_strings(
        result,
        ["locode", "country", "code"],
        df_name="locations",
    )

    result = _validate_with_pandera(LOCATIONS_SCHEMA, result)

    _ensure_valid_locodes(result, df_name="locations")

    return result
