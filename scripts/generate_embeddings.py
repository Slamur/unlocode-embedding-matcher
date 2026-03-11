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
from src.utils.files import ensure_parent_dir_exists, read_parquet, save_parquet


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
    search_texts = read_parquet(path=SEARCH_TEXTS_PATH)

    result = generate_embeddings(
        search_texts=search_texts,
        normalize_embeddings=False,
    )

    save_parquet(df=result.metadata, path=SEARCH_TEXT_METADATA_PATH)

    _save_embeddings(path=SEARCH_TEXT_EMBEDDINGS_PATH, embeddings=result.embeddings)
    _save_manifest(path=SEARCH_TEXT_EMBEDDINGS_MANIFEST_PATH, info=result.manifest)


if __name__ == "__main__":
    main()
