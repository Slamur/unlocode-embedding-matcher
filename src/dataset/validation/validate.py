from collections.abc import Iterable

import pandas as pd
import pandera.pandas as pa


def normalize_string_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        return

    series = df[column]

    # non-string -> na
    normalized = series.astype("string")
    normalized = normalized.str.strip()

    df[column] = normalized


def normalize_string_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    result = df.copy()
    for column in columns:
        normalize_string_column(result, column)
    return result


def ensure_no_empty_strings(df: pd.DataFrame, columns: Iterable[str], *, df_name: str) -> None:
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


def validate_with_pandera(schema: pa.DataFrameSchema, df: pd.DataFrame) -> pd.DataFrame:
    return schema.validate(df, lazy=True)


def ensure_same_locodes(df: pd.DataFrame, locations: pd.DataFrame, *, df_name: str) -> None:
    location_locodes = set(locations["locode"])
    missing_mask = ~df["locode"].isin(location_locodes)
    missing_mask = missing_mask.fillna(False)

    if missing_mask.any():
        missing_codes = sorted(df.loc[missing_mask, "locode"].dropna().unique())
        preview = missing_codes[:10]
        raise ValueError(f"{df_name} contain locodes absent from locations; examples: {preview}")
