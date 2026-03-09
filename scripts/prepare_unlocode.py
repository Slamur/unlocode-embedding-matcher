from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from src.config.paths import RAW_DIR


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

SUBDIVISION_COLUMNS = [
    "country",
    "subdivision_code",
    "subdivision_name",
    "subdivision_type",
]


def _find_csv_files(directory: Path, substring: str) -> Sequence[Path]:
    files = sorted(directory.glob(f"*{substring}*.csv"))

    if not files:
        raise RuntimeError(f"No files matching substring '{substring}' found in {directory}")
    
    return files


def _read_csv_file(path: Path, column_names: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        names=column_names,
        dtype=str,
        keep_default_na=False,
        encoding="cp1252",
    )


def _read_codes(csv_dir: Path = RAW_DIR) -> pd.DataFrame:
    part_files = _find_csv_files(csv_dir, "CodeListPart")

    parts = [_read_csv_file(path, column_names=UNLOCODE_COLUMNS) for path in part_files]
    return pd.concat(parts, ignore_index=True)


def _read_subdivisions(csv_dir: Path = RAW_DIR) -> pd.DataFrame:
    subdivision_files = _find_csv_files(csv_dir, "Subdivision")
    if not subdivision_files:
        raise RuntimeError(f"No subdivision CSV found in {csv_dir}")

    return _read_csv_file(subdivision_files[0], column_names=SUBDIVISION_COLUMNS)


def main() -> None:
    codes_df = _read_codes()
    subdivisions_df = _read_subdivisions()

    print(codes_df.shape)
    print(codes_df.head().to_string())

    print(subdivisions_df.shape)
    print(subdivisions_df.head().to_string())


if __name__ == "__main__":
    main()