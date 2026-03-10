import pandas as pd

from src.config.paths import RAW_DIR, INTERIM_DIR
from src.dataset.logging import log_df_info
from src.dataset.ingest.codes import read_prepared_codes
from src.dataset.ingest.subdivisions import read_prepared_subdivisions

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

    codes_df = read_prepared_codes(RAW_DIR, "CodeListPart")
    subdivisions_df = read_prepared_subdivisions(RAW_DIR, "Subdivision")

    merged_df = _DataMerger.merge_and_prepare(codes_df, subdivisions_df)

    merged_df_path = INTERIM_DIR / "merged_codes.parquet"
    merged_df.to_parquet(merged_df_path, index=False)

    print(f"Saved merged codes to: {merged_df_path}")
    print(f"Shape: {merged_df.shape}")


if __name__ == "__main__":
    main()
