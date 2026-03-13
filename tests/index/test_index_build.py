from unittest.mock import MagicMock

import numpy as np

from src.index.build import build_index


def test_build_index_creates_vector_index(monkeypatch) -> None:
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    mock_index = MagicMock()
    mock_cls = MagicMock(return_value=mock_index)

    monkeypatch.setattr("src.index.build.VectorIndex", mock_cls)

    index, result = build_index(embeddings)

    mock_cls.assert_called_once_with(dimension=2)
    mock_index.add.assert_called_once()

    assert index is mock_index
    assert result.vector_count == 2
    assert result.dimension == 2
