import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.embeddings import (
    DEFAULT_BATCH_SIZE,
    MODEL_NAME,
)
from src.config.paths import (
    EMBEDDINGS_DIR,
    SEARCH_TEXT_EMBEDDINGS_MANIFEST_PATH,
    SEARCH_TEXT_EMBEDDINGS_PATH,
    SEARCH_TEXT_METADATA_PATH,
    SEARCH_TEXTS_PATH,
)
from src.embeddings.model import EmbedderConfig, TextEmbedder

REQUIRED_COLUMNS = ["locode", "search_text"]


@dataclass(frozen=True)
class EmbeddingBuildInfo:
    model_name: str
    input_path: str
    row_count: int
    embedding_dim: int
    batch_size: int
    normalize_embeddings: bool
    dtype: str
    device: str


def read_search_texts(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    return df


def validate_search_texts(df: pd.DataFrame) -> None:
    if df.empty:
        raise ValueError("Input search_texts dataframe is empty")

    if df["locode"].isna().any():
        raise ValueError("Column 'locode' contains null values")

    if df["search_text"].isna().any():
        raise ValueError("Column 'search_text' contains null values")

    if not pd.api.types.is_string_dtype(df["search_text"]):
        raise ValueError("Column 'search_text' must have string dtype")

    empty_mask = df["search_text"].str.strip().eq("")
    if empty_mask.any():
        raise ValueError("Column 'search_text' contains empty or whitespace-only values")


def build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    metadata = df[["locode", "search_text"]].copy()
    metadata = metadata.reset_index(drop=True)
    metadata.insert(0, "row_id", np.arange(len(metadata), dtype=np.int64))
    return metadata


def save_manifest(path: Path, info: EmbeddingBuildInfo) -> None:
    path.write_text(
        json.dumps(asdict(info), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def generate_embeddings(
    input_path: Path = SEARCH_TEXTS_PATH,
    embeddings_path: Path = SEARCH_TEXT_EMBEDDINGS_PATH,
    metadata_path: Path = SEARCH_TEXT_METADATA_PATH,
    manifest_path: Path = SEARCH_TEXT_EMBEDDINGS_MANIFEST_PATH,
    batch_size: int = DEFAULT_BATCH_SIZE,
    normalize_embeddings: bool = False,
) -> None:
    df = read_search_texts(input_path)
    validate_search_texts(df)

    metadata = build_metadata(df)
    texts = metadata["search_text"].tolist()

    embedder = TextEmbedder(
        EmbedderConfig(model_name=MODEL_NAME),
    )
    embeddings = embedder.encode(
        texts=texts,
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
    )

    if len(embeddings) != len(metadata):
        raise ValueError(f"Embeddings count mismatch: {len(embeddings)} != {len(metadata)}")

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    np.save(embeddings_path, embeddings)
    metadata.to_parquet(metadata_path, index=False)

    build_info = EmbeddingBuildInfo(
        model_name=MODEL_NAME,
        input_path=str(input_path),
        row_count=len(metadata),
        embedding_dim=int(embeddings.shape[1]),
        batch_size=batch_size,
        normalize_embeddings=normalize_embeddings,
        dtype=str(embeddings.dtype),
        device=embedder.device,
    )
    save_manifest(manifest_path, build_info)


if __name__ == "__main__":
    generate_embeddings()
