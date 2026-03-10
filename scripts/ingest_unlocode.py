from src.config.paths import RAW_DIR, INTERIM_DIR
from src.dataset.ingest.codes import read_prepared_codes
from src.dataset.ingest.subdivisions import read_prepared_subdivisions
from src.dataset.ingest.merge import merge_and_prepare


def main() -> None:

    codes = read_prepared_codes(csv_dir=RAW_DIR, filename_substring="CodeListPart")
    subdivisions = read_prepared_subdivisions(csv_dir=RAW_DIR, filename_substring="Subdivision")

    merged_codes = merge_and_prepare(codes=codes, subdivisions=subdivisions)

    merged_codes_path = INTERIM_DIR / "merged_codes.parquet"
    merged_codes.to_parquet(merged_codes_path, index=False)

    print(f"Saved merged codes to: {merged_codes_path}")
    print(f"Shape: {merged_codes.shape}")


if __name__ == "__main__":
    main()
