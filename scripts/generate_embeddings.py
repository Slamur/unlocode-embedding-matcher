import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.config.paths import (
    SEARCH_TEXT_EMBEDDINGS_MANIFEST_PATH,
    SEARCH_TEXT_EMBEDDINGS_PATH,
    SEARCH_TEXT_METADATA_PATH,
)
from src.embeddings.generate import EmbeddingBuildInfo, generate_embeddings
from src.utils.files import ensure_parent_dir_exists, read_parquet


def _load_texts(metadata_path: Path) -> list[str]:
    metadata = read_parquet(path=metadata_path)

    return metadata["search_text"].tolist()


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
    texts = _load_texts(metadata_path=SEARCH_TEXT_METADATA_PATH)

    result = generate_embeddings(
        texts=texts,
        normalize_embeddings=False,
    )

    _save_embeddings(path=SEARCH_TEXT_EMBEDDINGS_PATH, embeddings=result.embeddings)

    print(f"Embeddings saved to: {SEARCH_TEXT_EMBEDDINGS_PATH}")
    print(f"Shape: {result.embeddings.shape}")

    _save_manifest(path=SEARCH_TEXT_EMBEDDINGS_MANIFEST_PATH, info=result.manifest)

    print(f"Embeddings manifest saved to: {SEARCH_TEXT_EMBEDDINGS_MANIFEST_PATH}")


if __name__ == "__main__":
    main()
