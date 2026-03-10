import pandas as pd

from src.dataset.logging import log_df_info

def _merge_codes_with_subdivisions(
    codes_df: pd.DataFrame, 
    subdivisions_df: pd.DataFrame,
) -> pd.DataFrame:
    merged_df = codes_df.merge(
        subdivisions_df,
        how="left",
        left_on=["country", "subdivision"],
        right_on=["country", "subdivision_code"],
        suffixes=("", "_subdiv"),
    )

    return merged_df


def _prepare_merged_df(
    merged_df: pd.DataFrame,
) -> pd.DataFrame:
    prepared_merged_df = merged_df.copy()

    for col in ["subdivision_code", "subdivision_name", "subdivision_type"]:
        prepared_merged_df[col] = prepared_merged_df[col].fillna("")

    return prepared_merged_df


def merge_and_prepare(
    codes_df: pd.DataFrame, 
    subdivisions_df: pd.DataFrame, 
    verbose: bool = False,
) -> pd.DataFrame:
    merged_df = _merge_codes_with_subdivisions(codes_df, subdivisions_df)
    prepared_merged_df = _prepare_merged_df(merged_df)

    log_df_info(merged_df, "Merged DataFrame", verbose=verbose)

    return prepared_merged_df
