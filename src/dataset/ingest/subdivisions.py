from pathlib import Path

import pandas as pd

from src.dataset.inspect import inspect_df_info
from src.dataset.io.csv import find_csv_files, read_csv_file


SUBDIVISION_COLUMNS = [
    "country",
    "subdivision_code",
    "subdivision_name",
    "subdivision_type",
]


def _read_subdivisions(
    csv_dir: Path, 
    filename_substring: str,
) -> pd.DataFrame:
    subdivision_files = find_csv_files(directory=csv_dir, filename_substring=filename_substring)

    return read_csv_file(path=subdivision_files[0], column_names=SUBDIVISION_COLUMNS)


def _prepare_subdivisions(
    subdivisions: pd.DataFrame,
) -> pd.DataFrame:
    prepared_subdivisions = subdivisions.copy()

    # pure string columns
    for col in ("country", "subdivision_code", "subdivision_name", "subdivision_type"):
        prepared_subdivisions[col] = prepared_subdivisions[col].str.strip()

    return prepared_subdivisions


def read_subdivisions_table(
    csv_dir: Path,
    filename_substring: str,
    verbose: bool = False,
) -> pd.DataFrame:
    subdivisions = _read_subdivisions(csv_dir=csv_dir, filename_substring=filename_substring)

    inspect_df_info(subdivisions, "Subdivisions", verbose=verbose)

    prepared_subdivisions = _prepare_subdivisions(subdivisions)

    inspect_df_info(prepared_subdivisions, "Prepared Subdivisions", verbose=verbose)

    return prepared_subdivisions
