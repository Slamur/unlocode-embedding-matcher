from collections.abc import Sequence
from pathlib import Path

import pandas as pd

_DEFAULT_ENCODING = "cp1252"


def find_csv_files(
    directory: Path,
    filename_substring: str,
    fail_on_empty: bool = True,
) -> Sequence[Path]:
    files = sorted(directory.glob(f"*{filename_substring}*.csv"))

    if fail_on_empty and not files:
        raise RuntimeError(
            f"No files matching substring '{filename_substring}' found in {directory}"
        )

    return files


def read_csv_file(
    path: Path,
    column_names: list[str] | None = None,
    encoding: str = _DEFAULT_ENCODING,
) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        names=column_names,
        dtype=str,
        keep_default_na=False,
        encoding=encoding,
    )
