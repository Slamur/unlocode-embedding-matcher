from pathlib import Path

import pandas as pd

from src.dataset.logging import log_df_info
from src.dataset.io import find_csv_files, read_csv_file


UNLOCODE_COLUMNS = [
    "change",
    "country",
    "code",
    "name",
    "name_wo_diacritics",
    "subdivision",
    "function",
    "status",
    "date",
    "iata",
    "coordinates",
    "remarks",
]


def _read_codes(csv_dir: Path, part_substring: str) -> pd.DataFrame:
    part_files = find_csv_files(csv_dir, part_substring)

    parts = [read_csv_file(path, column_names=UNLOCODE_COLUMNS) for path in part_files]
    return pd.concat(parts, ignore_index=True)


def _prepare_codes(codes: pd.DataFrame) -> pd.DataFrame:
    prepared_codes = codes.copy()

    # pure string columns
    for col in ("country", "code", "name", "name_wo_diacritics", "subdivision"):
        prepared_codes[col] = prepared_codes[col].str.strip()

    # filter columns without country and/or code
    prepared_codes = prepared_codes[(prepared_codes["country"] != "") & (prepared_codes["code"] != "")].copy()

    # generate locode columns
    prepared_codes["locode"] = prepared_codes["country"] + prepared_codes["code"]
    prepared_codes["locode_display"] = prepared_codes["country"] + " " + prepared_codes["code"]

    return prepared_codes


def read_prepared_codes(csv_dir: Path, part_substring: str, verbose: bool = False) -> pd.DataFrame:
    codes = _read_codes(csv_dir, part_substring)

    log_df_info(codes, "Codes", verbose=verbose)

    prepared_codes = _prepare_codes(codes)

    log_df_info(prepared_codes, "Prepared Codes", verbose=verbose)

    return prepared_codes
