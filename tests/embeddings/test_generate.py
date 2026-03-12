import numpy as np
import pandas as pd
import pytest

from src.embeddings.generate import generate_embeddings


class FakeTextEmbedder:
    def __init__(self, config) -> None:
        self.config = config
        self.device = "cpu"

    def encode(self, texts, batch_size, normalize_embeddings=False):
        assert texts == ["bishkek", "tokyo"]
        assert batch_size == 16
        assert normalize_embeddings is True

        return np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
            ],
            dtype=np.float32,
        )


def test_generate_embeddings_returns_embeddings_and_manifest(monkeypatch) -> None:
    metadata = pd.DataFrame(
        {
            "row_id": [0, 1],
            "locode": ["KG FRU", "JP TYO"],
            "search_text": ["bishkek", "tokyo"],
        }
    )

    monkeypatch.setattr("src.embeddings.generate.TextEmbedder", FakeTextEmbedder)

    result = generate_embeddings(
        metadata=metadata,
        batch_size=16,
        normalize_embeddings=True,
    )

    assert result.embeddings.shape == (2, 2)
    np.testing.assert_allclose(
        result.embeddings,
        np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )

    assert result.manifest.model_name
    assert result.manifest.row_count == 2
    assert result.manifest.embedding_dim == 2
    assert result.manifest.batch_size == 16
    assert result.manifest.normalize_embeddings is True
    assert result.manifest.dtype == "float32"
    assert result.manifest.device == "cpu"


class FakeTextEmbedderWithMismatch:
    def __init__(self, config) -> None:
        self.device = "cpu"

    def encode(self, texts, batch_size, normalize_embeddings=False):
        return np.array([[1.0, 2.0]], dtype=np.float32)


def test_generate_embeddings_raises_on_embeddings_count_mismatch(monkeypatch) -> None:
    metadata = pd.DataFrame(
        {
            "row_id": [0, 1],
            "locode": ["KG FRU", "JP TYO"],
            "search_text": ["bishkek", "tokyo"],
        }
    )

    monkeypatch.setattr("src.embeddings.generate.TextEmbedder", FakeTextEmbedderWithMismatch)

    with pytest.raises(ValueError, match="Embeddings count mismatch"):
        generate_embeddings(metadata=metadata)


def test_generate_embeddings_raises_when_search_text_column_missing() -> None:
    metadata = pd.DataFrame(
        {
            "row_id": [0, 1],
            "locode": ["KG FRU", "JP TYO"],
            # intentionally missing "search_text"
        }
    )

    with pytest.raises(ValueError, match="Metadata must contain 'search_text' column"):
        generate_embeddings(metadata=metadata)
