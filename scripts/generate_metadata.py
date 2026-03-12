from pathlib import Path

from src.config.paths import (
    SEARCH_TEXT_METADATA_PATH,
    SEARCH_TEXTS_PATH,
)
from src.dataset.inspect import inspect_df_info
from src.embeddings.metadata import generate_metadata
from src.utils.files import read_parquet, save_parquet


def _show_existing_metadata(
    metadata_path: Path = SEARCH_TEXT_METADATA_PATH,
) -> None:
    print(f"Metadata file already exists at: {metadata_path}")

    print("Loading existing metadata...")

    metadata = read_parquet(path=SEARCH_TEXT_METADATA_PATH)
    inspect_df_info(df=metadata, name="Existing Embeddings Metadata", verbose=True)


def _generate_and_save_metadata(
    metadata_path: Path = SEARCH_TEXT_METADATA_PATH,
    search_texts_path: Path = SEARCH_TEXTS_PATH,
) -> None:
    search_texts = read_parquet(path=search_texts_path)
    metadata = generate_metadata(search_texts=search_texts)

    save_parquet(df=metadata, path=metadata_path)

    print(f"Embeddings Metadata saved to: {SEARCH_TEXT_METADATA_PATH}")
    print(f"Shape: {metadata.shape}")


def main() -> None:
    metadata_path = SEARCH_TEXT_METADATA_PATH

    if metadata_path.exists():
        _show_existing_metadata(metadata_path=metadata_path)
    else:
        _generate_and_save_metadata(metadata_path=metadata_path)


if __name__ == "__main__":
    main()
