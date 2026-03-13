from src.config.paths import (
    SEARCH_TEXTS_EMBEDDINGS_MANIFEST_PATH,
    SEARCH_TEXTS_EMBEDDINGS_PATH,
    SEARCH_TEXTS_METADATA_PATH,
    SEARCH_TEXTS_PATH,
)
from src.embeddings.artifacts import read_and_validate_artifacts
from src.embeddings.generate import generate_embeddings
from src.embeddings.io import save_embeddings, save_manifest
from src.embeddings.metadata import generate_metadata
from src.utils.files import read_parquet, save_parquet


def main() -> None:
    metadata_path = SEARCH_TEXTS_METADATA_PATH
    embeddings_path = SEARCH_TEXTS_EMBEDDINGS_PATH
    manifest_path = SEARCH_TEXTS_EMBEDDINGS_MANIFEST_PATH

    existing_artifacts = read_and_validate_artifacts(
        metadata_path=metadata_path,
        embeddings_path=embeddings_path,
        manifest_path=manifest_path,
    )

    if existing_artifacts is not None:
        print("Skipping generation: all artifacts are valid and exist.")
        return

    search_texts = read_parquet(path=SEARCH_TEXTS_PATH)
    metadata = generate_metadata(search_texts=search_texts)

    result = generate_embeddings(
        metadata=metadata,
        normalize_embeddings=False,
    )

    save_parquet(df=metadata, path=metadata_path)

    print(f"Embeddings Metadata saved to: {metadata_path}")
    print(f"Shape: {metadata.shape}")

    save_embeddings(embeddings=result.embeddings, path=embeddings_path)

    print(f"Embeddings saved to: {embeddings_path}")
    print(f"Shape: {result.embeddings.shape}")

    save_manifest(manifest=result.manifest, path=manifest_path)

    print(f"Embeddings manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
