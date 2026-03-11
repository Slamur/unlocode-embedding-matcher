from src.config.paths import MERGED_CODES_PATH, RAW_DIR
from src.dataset.ingestion.codes import read_codes_table
from src.dataset.ingestion.merge import build_merged_table
from src.dataset.ingestion.subdivisions import read_subdivisions_table
from src.utils.files import save_parquet


def main() -> None:

    csv_dir = RAW_DIR

    codes = read_codes_table(csv_dir=csv_dir, filename_substring="CodeListPart")
    subdivisions = read_subdivisions_table(csv_dir=csv_dir, filename_substring="Subdivision")

    merged_codes = build_merged_table(codes=codes, subdivisions=subdivisions)

    merged_codes_path = MERGED_CODES_PATH

    save_parquet(df=merged_codes, path=merged_codes_path)

    print(f"Saved merged codes to: {merged_codes_path}")
    print(f"Shape: {merged_codes.shape}")


if __name__ == "__main__":
    main()
