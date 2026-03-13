from pathlib import Path

import faiss

from src.utils.files import ensure_parent_dir_exists, require_file_exists


def save_index(index: faiss.Index, path: Path) -> None:
    ensure_parent_dir_exists(path=path)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    require_file_exists(path=path)
    return faiss.read_index(str(path))
