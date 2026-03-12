import numpy as np
import pandas as pd

from src.dataset.inspect import inspect_df_info

_REQUIRED_COLUMNS = ["locode", "search_text"]


def _validate_search_texts(search_texts: pd.DataFrame) -> None:
    missing_columns = [column for column in _REQUIRED_COLUMNS if column not in search_texts.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if search_texts.empty:
        raise ValueError("Input search_texts dataframe is empty")

    if search_texts["locode"].isna().any():
        raise ValueError("Column 'locode' contains null values")

    if search_texts["search_text"].isna().any():
        raise ValueError("Column 'search_text' contains null values")

    non_string_mask = ~search_texts["search_text"].map(lambda value: isinstance(value, str))
    if non_string_mask.any():
        raise ValueError("Column 'search_text' contains non-string values")

    empty_mask = search_texts["search_text"].str.strip().eq("")
    if empty_mask.any():
        raise ValueError("Column 'search_text' contains empty or whitespace-only values")


def _build_metadata(search_texts: pd.DataFrame) -> pd.DataFrame:
    metadata = search_texts[["locode", "search_text"]].copy()
    metadata = metadata.reset_index(drop=True)
    metadata.insert(0, "row_id", np.arange(len(metadata), dtype=np.int64))
    return metadata


def generate_metadata(
    search_texts: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    _validate_search_texts(search_texts)

    metadata = _build_metadata(search_texts=search_texts)

    inspect_df_info(df=metadata, name="Embeddings Metadata", verbose=verbose)

    return metadata
