import pandas as pd

from src.dataset.inspect import inspect_df_info

def _merge_codes_with_subdivisions(
    codes: pd.DataFrame, 
    subdivisions: pd.DataFrame,
) -> pd.DataFrame:
    merged_codes = codes.merge(
        subdivisions,
        how="left",
        left_on=["country", "subdivision"],
        right_on=["country", "subdivision_code"],
        suffixes=("", "_subdiv"),
    )

    return merged_codes


def _prepare_merged_codes(
    merged_codes: pd.DataFrame,
) -> pd.DataFrame:
    prepared_merged_codes = merged_codes.copy()

    for col in ["subdivision_code", "subdivision_name", "subdivision_type"]:
        prepared_merged_codes[col] = prepared_merged_codes[col].fillna("")

    return prepared_merged_codes


def build_merged_table(
    codes: pd.DataFrame, 
    subdivisions: pd.DataFrame, 
    verbose: bool = False,
) -> pd.DataFrame:
    merged_codes = _merge_codes_with_subdivisions(codes=codes, subdivisions=subdivisions)
    prepared_merged_codes = _prepare_merged_codes(merged_codes=merged_codes)

    inspect_df_info(merged_codes, "Merged DataFrame", verbose=verbose)

    return prepared_merged_codes
