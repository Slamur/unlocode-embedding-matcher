import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.config.paths import (
    SEARCH_TEXT_EMBEDDINGS_MANIFEST_PATH,
    SEARCH_TEXT_EMBEDDINGS_PATH,
    SEARCH_TEXT_METADATA_PATH,
    SEARCH_TEXTS_PATH,
)
from src.embeddings.generate import EmbeddingBuildInfo, generate_embeddings
from src.embeddings.metadata import generate_metadata
from src.utils.files import ensure_parent_dir_exists, read_parquet, save_parquet


def _validate_existing_artifacts(
    metadata_path: Path,
    embeddings_path: Path,
    manifest_path: Path,
) -> None:
    embeddings = np.load(embeddings_path, mmap_mode="r")
    metadata = read_parquet(path=metadata_path)

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Artifacts mismatch: embeddings rows = {embeddings.shape[0]}, "
            f"metadata rows = {len(metadata)}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if manifest["row_count"] != len(metadata):
        raise ValueError("Manifest row_count does not match metadata")
    if manifest["embedding_dim"] != embeddings.shape[1]:
        raise ValueError("Manifest embedding_dim does not match embeddings")


def _valid_artifacts_exist(
    metadata_path: Path = SEARCH_TEXT_METADATA_PATH,
    embeddings_path: Path = SEARCH_TEXT_EMBEDDINGS_PATH,
    manifest_path: Path = SEARCH_TEXT_EMBEDDINGS_MANIFEST_PATH,
) -> bool:
    artifact_exists = [
        metadata_path.exists(),
        embeddings_path.exists(),
        manifest_path.exists(),
    ]

    if not all(artifact_exists):
        if any(artifact_exists):
            raise RuntimeError(
                "Found incomplete embedding artifacts." "Remove them manually and rerun the script."
            )

        return False

    _validate_existing_artifacts(
        metadata_path=metadata_path, embeddings_path=embeddings_path, manifest_path=manifest_path
    )

    return True


def _save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    ensure_parent_dir_exists(path=path)
    np.save(path, embeddings)


def _save_manifest(path: Path, info: EmbeddingBuildInfo) -> None:
    ensure_parent_dir_exists(path=path)

    path.write_text(
        json.dumps(asdict(info), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    metadata_path = SEARCH_TEXT_METADATA_PATH
    embeddings_path = SEARCH_TEXT_EMBEDDINGS_PATH
    manifest_path = SEARCH_TEXT_EMBEDDINGS_MANIFEST_PATH

    if _valid_artifacts_exist(
        metadata_path=metadata_path, embeddings_path=embeddings_path, manifest_path=manifest_path
    ):
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

    _save_embeddings(path=embeddings_path, embeddings=result.embeddings)

    print(f"Embeddings saved to: {embeddings_path}")
    print(f"Shape: {result.embeddings.shape}")

    _save_manifest(path=manifest_path, info=result.manifest)

    print(f"Embeddings manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
