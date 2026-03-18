import pandas as pd
import pytest

from src.search.service import SearchService
from tests.search.service.conftest import DummyEmbedder


def test_init_raises_if_metadata_missing_required_columns(
    embedder: DummyEmbedder,
    build_index,
) -> None:
    metadata = pd.DataFrame(
        {
            "row_id": [0],
            "locode": ["KGFRU"],
        }
    )

    index = build_index(size=len(metadata))

    with pytest.raises(ValueError, match="Metadata is missing required columns: search_text"):
        SearchService(index=index, metadata=metadata, embedder=embedder)


def test_init_raises_if_metadata_size_does_not_match_index_size(
    metadata: pd.DataFrame,
    embedder: DummyEmbedder,
    build_index,
) -> None:
    index = build_index(size=len(metadata) + 1)

    with pytest.raises(ValueError, match="does not match index size"):
        SearchService(index=index, metadata=metadata, embedder=embedder)
