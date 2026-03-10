from pathlib import Path

import pandas as pd

from src.config.paths import RAW_DIR, INTERIM_DIR
from src.dataset.logging import log_df_info
from src.dataset.io.csv import *
from src.dataset.codes import read_prepared_codes

SUBDIVISION_COLUMNS = [
    "country",
    "subdivision_code",
    "subdivision_name",
    "subdivision_type",
]


class _SubdivisionsPreparer:

    @staticmethod
    def _read_subdivisions(csv_dir: Path = RAW_DIR) -> pd.DataFrame:
        subdivision_files = find_csv_files(csv_dir, "Subdivision")
        if not subdivision_files:
            raise RuntimeError(f"No subdivision CSV found in {csv_dir}")

        return read_csv_file(subdivision_files[0], column_names=SUBDIVISION_COLUMNS)

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

        log_df_info(subdivisions_df, "Subdivisions", verbose=verbose)

        prepared_subdivisions_df = _SubdivisionsPreparer._prepare_subdivisions(subdivisions_df)

        log_df_info(prepared_subdivisions_df, "Prepared Subdivisions", verbose=verbose)

        return prepared_subdivisions_df


class _DataMerger:

    @staticmethod
    def _merge_codes_with_subdivisions(codes_df: pd.DataFrame, subdivisions_df: pd.DataFrame) -> pd.DataFrame:
        merged_df = codes_df.merge(
            subdivisions_df,
            how="left",
            left_on=["country", "subdivision"],
            right_on=["country", "subdivision_code"],
            suffixes=("", "_subdiv"),
        )

        return merged_df
    
    @staticmethod
    def _prepare_merged_df(merged_df: pd.DataFrame) -> pd.DataFrame:
        prepared_merged_df = merged_df.copy()

        for col in ["subdivision_code", "subdivision_name", "subdivision_type"]:
            prepared_merged_df[col] = prepared_merged_df[col].fillna("")

        return prepared_merged_df
    
    @staticmethod
    def merge_and_prepare(codes_df: pd.DataFrame, subdivisions_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        merged_df = _DataMerger._merge_codes_with_subdivisions(codes_df, subdivisions_df)
        prepared_merged_df = _DataMerger._prepare_merged_df(merged_df)

        log_df_info(merged_df, "Merged DataFrame", verbose=verbose)

        return prepared_merged_df


def main() -> None:

    codes_df = read_prepared_codes(RAW_DIR, 'CodeListPart')
    subdivisions_df = _SubdivisionsPreparer.read_prepared_subdivisions_df()

    merged_df = _DataMerger.merge_and_prepare(codes_df, subdivisions_df)

    merged_df_path = INTERIM_DIR / "merged_codes.parquet"
    merged_df.to_parquet(merged_df_path, index=False)

    print(f"Saved merged codes to: {merged_df_path}")
    print(f"Shape: {merged_df.shape}")


if __name__ == "__main__":
    main()
