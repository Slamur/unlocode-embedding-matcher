from src.config.paths import INTERIM_DIR, RAW_DIR
from src.dataset.ingest.codes import read_codes_table
from src.dataset.ingest.merge import build_merged_table
from src.dataset.ingest.subdivisions import read_subdivisions_table


def main() -> None:

    codes = read_codes_table(csv_dir=RAW_DIR, filename_substring="CodeListPart")
    subdivisions = read_subdivisions_table(csv_dir=RAW_DIR, filename_substring="Subdivision")

    merged_codes = build_merged_table(codes=codes, subdivisions=subdivisions)

    merged_codes_path = INTERIM_DIR / "merged_codes.parquet"
    merged_codes.to_parquet(merged_codes_path, index=False)

    print(f"Saved merged codes to: {merged_codes_path}")
    print(f"Shape: {merged_codes.shape}")


if __name__ == "__main__":
    main()
