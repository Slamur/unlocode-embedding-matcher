from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.embeddings import DEFAULT_BATCH_SIZE, MODEL_NAME
from src.embeddings.model import EmbedderConfig, TextEmbedder
from src.index.model import VectorIndex
from src.search.model import SearchHit, SearchRequest, SearchResponse
from src.search.normalize import build_query_variants


@dataclass(frozen=True)
class SearchConfig:
    model_name: str = MODEL_NAME
    batch_size: int = DEFAULT_BATCH_SIZE


class SearchService:
    def __init__(
        self,
        *,
        index: VectorIndex,
        metadata: pd.DataFrame,
        config: SearchConfig | None = None,
    ) -> None:
        self._index = index
        self._metadata = metadata.reset_index(drop=True).copy()
        self._config = config or SearchConfig()
        self._embedder = TextEmbedder(EmbedderConfig(model_name=self._config.model_name))

        self._validate_metadata()

    def _validate_metadata(self) -> None:
        required_columns = {
            "locode",
            "alias_text",
            "country",
            "subdivision_name",
            "search_text_kind",
            "search_text",
        }

        missing = required_columns - set(self._metadata.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Metadata is missing required columns: {missing_list}")

        if len(self._metadata) != self._index.size:
            raise ValueError(
                f"Metadata row count {len(self._metadata)} "
                f"does not match index size {self._index.size}"
            )

    def search(self, request: SearchRequest) -> SearchResponse:
        if not request.query.strip():
            raise ValueError("Query must not be empty")

        variants = build_query_variants(request.query)
        normalized_query = variants[0] if variants else ""

        if not variants:
            return SearchResponse(
                query=request.query,
                normalized_query=normalized_query,
                hits=[],
            )

        query_embeddings = self._embedder.encode(
            texts=variants,
            batch_size=self._config.batch_size,
            normalize_embeddings=False,
        )

        raw_hits: list[SearchHit] = []

        for query_vector in query_embeddings:
            variant_hits = self._build_variant_hits(query_vector=query_vector, top_k=request.top_k)
            raw_hits.extend(variant_hits)

        merged_hits = self._deduplicate_hits(hits=raw_hits, top_k=request.top_k)

        return SearchResponse(
            query=request.query,
            normalized_query=normalized_query,
            hits=merged_hits,
        )

    def _build_variant_hits(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        scores, ids = self._index.search(query_vector, top_k=top_k)

        variant_hits: list[SearchHit] = [
            self._build_hit(score, idx)
            for score, idx in zip(scores[0], ids[0], strict=True)
            if idx >= 0
        ]

        return variant_hits

    def _build_hit(self, score: float, idx: int) -> SearchHit:
        row = self._metadata.iloc[int(idx)]

        return SearchHit(
            locode=str(row["locode"]),
            score=float(score),
            search_text=str(row["search_text"]),
            alias_text=str(row["alias_text"]),
            country=str(row["country"]),
            subdivision_name=str(row["subdivision_name"]),
            search_text_kind=str(row["search_text_kind"]),
        )

    def _deduplicate_hits(
        self,
        hits: list[SearchHit],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        best_by_locode: dict[str, SearchHit] = {}

        for hit in hits:
            existing = best_by_locode.get(hit.locode)
            if existing is None or hit.score > existing.score:
                best_by_locode[hit.locode] = hit

        merged = sorted(
            best_by_locode.values(),
            key=lambda hit: hit.score,
            reverse=True,
        )

        return merged[:top_k]
