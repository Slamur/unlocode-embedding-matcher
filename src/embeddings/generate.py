from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.embeddings import (
    DEFAULT_BATCH_SIZE,
    MODEL_NAME,
)
from src.embeddings.model import EmbedderConfig, TextEmbedder


@dataclass(frozen=True)
class EmbeddingBuildInfo:
    model_name: str
    row_count: int
    embedding_dim: int
    batch_size: int
    normalize_embeddings: bool
    dtype: str
    device: str


@dataclass(frozen=True)
class EmbeddingBuildResult:
    embeddings: np.ndarray
    metadata: pd.DataFrame
    manifest: EmbeddingBuildInfo


_REQUIRED_COLUMNS = ["locode", "search_text"]


def _validate_search_texts(search_texts: pd.DataFrame) -> None:
    missing_columns = [column for column in _REQUIRED_COLUMNS if column not in search_texts.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if search_texts.empty:
        raise ValueError("Input search_texts dataframe is empty")

    if search_texts["locode"].isna().any():
        raise ValueError("Column 'locode' contains null values")

    if search_texts["search_text"].isna().any():
        raise ValueError("Column 'search_text' contains null values")

    if not pd.api.types.is_string_dtype(search_texts["search_text"]):
        raise ValueError("Column 'search_text' must have string dtype")

    empty_mask = search_texts["search_text"].str.strip().eq("")
    if empty_mask.any():
        raise ValueError("Column 'search_text' contains empty or whitespace-only values")


def _build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    metadata = df[["locode", "search_text"]].copy()
    metadata = metadata.reset_index(drop=True)
    metadata.insert(0, "row_id", np.arange(len(metadata), dtype=np.int64))
    return metadata


def generate_embeddings(
    search_texts: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize_embeddings: bool = False,
) -> EmbeddingBuildResult:
    _validate_search_texts(search_texts)

    metadata = _build_metadata(search_texts)

    embedder = TextEmbedder(
        EmbedderConfig(model_name=MODEL_NAME),
    )

    texts = metadata["search_text"].tolist()

    embeddings = embedder.encode(
        texts=texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )

    if len(embeddings) != len(metadata):
        raise ValueError(f"Embeddings count mismatch: {len(embeddings)} != {len(metadata)}")

    build_info = EmbeddingBuildInfo(
        model_name=MODEL_NAME,
        row_count=len(metadata),
        embedding_dim=int(embeddings.shape[1]),
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        dtype=str(embeddings.dtype),
        device=embedder.device,
    )

    return EmbeddingBuildResult(embeddings=embeddings, metadata=metadata, manifest=build_info)
