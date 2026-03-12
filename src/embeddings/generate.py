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
    manifest: EmbeddingBuildInfo


def generate_embeddings(
    metadata: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize_embeddings: bool = False,
) -> EmbeddingBuildResult:
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

    return EmbeddingBuildResult(embeddings=embeddings, manifest=build_info)
