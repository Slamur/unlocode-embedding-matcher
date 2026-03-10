import pandas as pd

from src.dataset.logging import log_df_info


def _build_locations(
    merged: pd.DataFrame,
) -> pd.DataFrame:
    locations = merged.drop(columns=["name", "name_wo_diacritics"]).copy()

    return locations


def _prepare_locations(
    locations: pd.DataFrame,
) -> pd.DataFrame:
    prepared_locations = locations.copy()

    # remove full locode duplicates
    prepared_locations = prepared_locations.drop_duplicates(subset=["locode"])

    return prepared_locations


def resolve_locations(
    merged: pd.DataFrame, 
    verbose: bool = False,
) -> pd.DataFrame:
    locations = _build_locations(merged)

    log_df_info(locations, "Locations DataFrame", verbose=verbose)

    prepared_locations = _prepare_locations(locations)

    log_df_info(prepared_locations, "Prepared Locations DataFrame", verbose=verbose)

    return prepared_locations
