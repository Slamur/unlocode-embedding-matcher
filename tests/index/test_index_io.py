import numpy as np
import pytest

from src.index.build import build_index
from src.index.io import load_index, save_index


def test_save_and_load_index_roundtrip(tmp_path) -> None:
    embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )

    index, _ = build_index(embeddings=embeddings)

    path = tmp_path / "index" / "faiss.index"
    save_index(index=index, path=path)

    loaded_index = load_index(path=path)

    query = np.array([[0.6, 0.8]], dtype=np.float32)
    scores, ids = loaded_index.search(query, 2)

    assert loaded_index.ntotal == 3
    assert ids[0].tolist() == [2, 1]
    assert scores[0][0] >= scores[0][1]


def test_load_index_raises_when_file_does_not_exist(tmp_path) -> None:
    path = tmp_path / "missing.index"

    with pytest.raises(FileNotFoundError):
        load_index(path=path)
