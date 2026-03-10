import pandas as pd

from src.dataset.inspect import inspect_df_info


def _build_locations(
    merged_codes: pd.DataFrame,
) -> pd.DataFrame:
    locations = merged_codes.drop(columns=["name", "name_wo_diacritics"]).copy()

    return locations


def _prepare_locations(
    locations: pd.DataFrame,
) -> pd.DataFrame:
    prepared_locations = locations.copy()

    # remove full locode duplicates
    prepared_locations = prepared_locations.drop_duplicates(subset=["locode"])

    return prepared_locations


def build_locations_table(
    merged_codes: pd.DataFrame, 
    verbose: bool = False,
) -> pd.DataFrame:
    locations = _build_locations(merged_codes)

    inspect_df_info(locations, "Locations DataFrame", verbose=verbose)

    prepared_locations = _prepare_locations(locations)

    inspect_df_info(prepared_locations, "Prepared Locations DataFrame", verbose=verbose)

    return prepared_locations
