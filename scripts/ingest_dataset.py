from src.config.paths import INTERIM_DIR, RAW_DIR
from src.dataset.ingestion.codes import read_codes_table
from src.dataset.ingestion.merge import build_merged_table
from src.dataset.ingestion.subdivisions import read_subdivisions_table
from src.utils.paths import ensure_parent_dir_exists


def main() -> None:

    csv_dir = RAW_DIR

    codes = read_codes_table(csv_dir=csv_dir, filename_substring="CodeListPart")
    subdivisions = read_subdivisions_table(csv_dir=csv_dir, filename_substring="Subdivision")

    merged_codes = build_merged_table(codes=codes, subdivisions=subdivisions)

    merged_codes_path = INTERIM_DIR / "merged_codes.parquet"

    ensure_parent_dir_exists(path=merged_codes_path)
    merged_codes.to_parquet(merged_codes_path, index=False)

    print(f"Saved merged codes to: {merged_codes_path}")
    print(f"Shape: {merged_codes.shape}")


if __name__ == "__main__":
    main()
