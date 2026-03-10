from collections.abc import Sequence
from pathlib import Path

import pandas as pd


DEFAULT_ENCODING = "cp1252"


def find_csv_files(
    directory: Path, 
    filename_substring: str,
) -> Sequence[Path]:
    files = sorted(directory.glob(f"*{filename_substring}*.csv"))

    if not files:
        raise RuntimeError(f"No files matching substring '{filename_substring}' found in {directory}")
    
    return files


def read_csv_file(
    path: Path, 
    column_names: list[str] | None = None, 
    encoding: str = DEFAULT_ENCODING,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        names=column_names,
        dtype=str,
        keep_default_na=False,
        encoding=encoding,
    )
