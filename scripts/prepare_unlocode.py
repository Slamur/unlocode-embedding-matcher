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


def _print_df_info(df: pd.DataFrame, name: str, verbose: bool = False) -> None:
    if not verbose:
        return

    print(f"{name}:")
    print(f"Shape: {df.shape}")
    print(df.head().to_string())
    print()


class _CodesPreparer:

    @staticmethod
    def _read_codes(csv_dir: Path = RAW_DIR) -> pd.DataFrame:
        part_files = _find_csv_files(csv_dir, "CodeListPart")

        parts = [_read_csv_file(path, column_names=UNLOCODE_COLUMNS) for path in part_files]
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _prepare_codes_df(codes_df: pd.DataFrame) -> pd.DataFrame:
        prepared_codes_df = codes_df.copy()

        # pure string columns
        for col in ("country", "code", "name", "name_wo_diacritics", "subdivision"):
            prepared_codes_df[col] = prepared_codes_df[col].str.strip()

        # filter columns without country and/or code
        prepared_codes_df = prepared_codes_df[(prepared_codes_df["country"] != "") & (prepared_codes_df["code"] != "")].copy()

        # generate locode columns
        prepared_codes_df["locode"] = prepared_codes_df["country"] + prepared_codes_df["code"]
        prepared_codes_df["locode_display"] = prepared_codes_df["country"] + " " + prepared_codes_df["code"]

        return prepared_codes_df

    @staticmethod
    def read_prepared_codes_df(verbose: bool = False) -> pd.DataFrame:
        codes_df = _CodesPreparer._read_codes()

        _print_df_info(codes_df, "Codes", verbose=verbose)

        prepared_codes_df = _CodesPreparer._prepare_codes_df(codes_df)

        _print_df_info(prepared_codes_df, "Prepared Codes", verbose=verbose)

        return prepared_codes_df


class _SubdivisionsPreparer:

    @staticmethod
    def _read_subdivisions(csv_dir: Path = RAW_DIR) -> pd.DataFrame:
        subdivision_files = _find_csv_files(csv_dir, "Subdivision")
        if not subdivision_files:
            raise RuntimeError(f"No subdivision CSV found in {csv_dir}")

        return _read_csv_file(subdivision_files[0], column_names=SUBDIVISION_COLUMNS)

    @staticmethod
    def _prepare_subdivisions(
        subdivisions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        prepared_subdivisions_df = subdivisions_df.copy()

        # pure string columns
        for col in ("country", "subdivision_code", "subdivision_name", "subdivision_type"):
            prepared_subdivisions_df[col] = prepared_subdivisions_df[col].str.strip()

        return prepared_subdivisions_df
    
    @staticmethod
    def read_prepared_subdivisions_df(verbose: bool = False) -> pd.DataFrame:
        subdivisions_df = _SubdivisionsPreparer._read_subdivisions()

        _print_df_info(subdivisions_df, "Subdivisions", verbose=verbose)

        prepared_subdivisions_df = _SubdivisionsPreparer._prepare_subdivisions(subdivisions_df)

        _print_df_info(prepared_subdivisions_df, "Prepared Subdivisions", verbose=verbose)

        return prepared_subdivisions_df


def _merge_codes_with_subdivisions(codes_df: pd.DataFrame, subdivisions_df: pd.DataFrame) -> pd.DataFrame:
    merged_df = codes_df.merge(
        subdivisions_df,
        how="left",
        left_on=["country", "subdivision"],
        right_on=["country", "subdivision_code"],
        suffixes=("", "_subdiv"),
    )

    return merged_df


def main() -> None:

    prepared_codes_df = _CodesPreparer.read_prepared_codes_df()
    prepared_subdivisions_df = _SubdivisionsPreparer.read_prepared_subdivisions_df()

    merged_df = _merge_codes_with_subdivisions(prepared_codes_df, prepared_subdivisions_df)

    _print_df_info(merged_df, "Merged DataFrame", verbose=True)

    print("Unique locodes:", merged_df["locode"].nunique())
    print("Rows:", len(merged_df))


if __name__ == "__main__":
    main()