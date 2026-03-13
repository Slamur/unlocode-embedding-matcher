import numpy as np
import pytest

from src.embeddings.generate import EmbeddingBuildInfo
from src.embeddings.io import (
    read_embeddings,
    read_manifest,
    save_embeddings,
    save_manifest,
)


def test_save_and_read_embeddings_roundtrip(tmp_path) -> None:
    path = tmp_path / "embeddings.npy"
    expected = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    save_embeddings(expected, path)
    actual = read_embeddings(path)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected)


def test_read_embeddings_raises_when_file_missing(tmp_path) -> None:
    path = tmp_path / "missing.npy"

    with pytest.raises(FileNotFoundError):
        read_embeddings(path)


def test_save_and_read_manifest_roundtrip(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    expected = EmbeddingBuildInfo(
        model_name="test-model",
        row_count=2,
        embedding_dim=3,
        batch_size=64,
        normalize_embeddings=False,
        dtype="float32",
        device="cpu",
    )

    save_manifest(expected, path)
    actual = read_manifest(path)

    assert actual == expected


def test_read_manifest_raises_when_file_missing(tmp_path) -> None:
    path = tmp_path / "missing.json"

    with pytest.raises(FileNotFoundError):
        read_manifest(path)
