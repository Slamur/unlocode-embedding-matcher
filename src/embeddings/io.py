import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.embeddings.generate import EmbeddingBuildInfo
from src.utils.files import ensure_parent_dir_exists, require_file_exists

_DEFAULT_MANIFEST_ENCODING = "utf-8"


def save_embeddings(embeddings: np.ndarray, path: Path) -> None:
    ensure_parent_dir_exists(path=path)
    np.save(path, embeddings)


def read_embeddings(path: Path) -> np.ndarray:
    require_file_exists(path=path)

    return np.load(path, mmap_mode="r")


def save_manifest(
    manifest: EmbeddingBuildInfo, path: Path, encoding: str = _DEFAULT_MANIFEST_ENCODING
) -> None:
    ensure_parent_dir_exists(path=path)

    path.write_text(
        json.dumps(asdict(manifest), ensure_ascii=False, indent=2),
        encoding=encoding,
    )


def read_manifest(path: Path, encoding: str = _DEFAULT_MANIFEST_ENCODING) -> EmbeddingBuildInfo:
    dict_from_json = json.loads(path.read_text(encoding=encoding))

    return EmbeddingBuildInfo(**dict_from_json)
