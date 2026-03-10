from collections.abc import Iterable

import pandas as pd
import pandera.pandas as pa

from src.dataset.validation.schemas import ALIASES_SCHEMA, LOCATIONS_SCHEMA


def _normalize_string_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        return

    series = df[column]

    # non-string -> na
    normalized = series.astype("string")
    normalized = normalized.str.strip()

    df[column] = normalized


def _normalize_string_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        _normalize_string_column(result, column)
    return result


def _ensure_no_empty_strings(df: pd.DataFrame, columns: Iterable[str], *, df_name: str) -> None:
    errors: list[str] = []

    for column in columns:
        if column not in df.columns:
            continue

        series = df[column]
        # null == "" -> na
        empty_mask = series.eq("").fillna(False)
        if empty_mask.any():
            count = int(empty_mask.sum())
            errors.append(f"{df_name}.{column} contains {count} empty value(s) after trimming")

    if errors:
        raise ValueError("; ".join(errors))


def _validate_with_pandera(schema: pa.DataFrameSchema, df: pd.DataFrame) -> pd.DataFrame:
    return schema.validate(df, lazy=True)


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
            "name",
            "name_wo_diacritics",
            "subdivision_code",
            "subdivision_name",
        ],
    )

    _ensure_no_empty_strings(
        result,
        ["locode", "country", "code", "name", "name_wo_diacritics"],
        df_name="locations",
    )

    result = _validate_with_pandera(LOCATIONS_SCHEMA, result)

    _ensure_valid_locodes(result, df_name="locations")

    return result


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
        _ensure_same_locodes(result, locations)

    return result
