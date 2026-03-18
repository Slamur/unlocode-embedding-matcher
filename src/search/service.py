from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config.embeddings import DEFAULT_BATCH_SIZE, MODEL_NAME
from src.embeddings.model import EmbedderConfig, TextEmbedder
from src.index.model import VectorIndex
from src.search.model import SearchHit, SearchRequest, SearchResponse
from src.search.variants import QueryVariant, build_query_variants
from src.text.normalize import normalize_text


@dataclass(frozen=True)
class SearchConfig:
    model_name: str = MODEL_NAME
    batch_size: int = DEFAULT_BATCH_SIZE
    aggregation_alpha: float = 0.05


@dataclass(frozen=True)
class VariantSearchHit:
    row_id: int
    locode: str
    raw_score: float
    score: float
    search_text: str
    variant_text: str
    variant_kind: str
    variant_weight: float


class SearchService:
    def __init__(
        self,
        *,
        index: VectorIndex,
        metadata: pd.DataFrame,
        config: SearchConfig | None = None,
        embedder: TextEmbedder | None = None,
    ) -> None:
        self._index = index
        self._metadata = metadata.reset_index(drop=True).copy()
        self._config = config or SearchConfig()
        self._embedder = embedder or TextEmbedder(
            EmbedderConfig(model_name=self._config.model_name)
        )

        self._validate_metadata()

    def _validate_metadata(self) -> None:
        required_columns = {
            "row_id",
            "locode",
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

        normalized_query = normalize_text(text=request.query)
        variants = build_query_variants(normalized_query=normalized_query)

        if not variants:
            return SearchResponse(
                query=request.query,
                normalized_query=normalized_query,
                hits=[],
            )

        query_embeddings = self._embedder.encode(
            texts=[variant.text for variant in variants],
            batch_size=self._config.batch_size,
            normalize_embeddings=False,
        )

        raw_hits: list[VariantSearchHit] = []

        for variant, query_vector in zip(variants, query_embeddings, strict=True):
            variant_hits = self._build_variant_hits(
                query_vector=query_vector,
                variant=variant,
                top_k=request.top_k,
            )
            raw_hits.extend(variant_hits)

        aggregated_hits = self._aggregate_hits(hits=raw_hits)
        top_hits = self._top_hits(hits=aggregated_hits, top_k=request.top_k)

        return SearchResponse(
            query=request.query,
            normalized_query=normalized_query,
            hits=top_hits,
        )

    def _build_variant_hits(
        self,
        query_vector: np.ndarray,
        *,
        variant: QueryVariant,
        top_k: int,
    ) -> list[VariantSearchHit]:
        scores, ids = self._index.search(query_vector, top_k=top_k)

        variant_hits: list[VariantSearchHit] = [
            self._build_hit(score=score, idx=idx, variant=variant)
            for score, idx in zip(scores[0], ids[0], strict=True)
            if idx >= 0
        ]

        return variant_hits

    def _build_hit(
        self,
        score: float,
        idx: int,
        *,
        variant: QueryVariant,
    ) -> VariantSearchHit:
        row = self._metadata.iloc[int(idx)]
        weighted_score = float(score) * variant.weight

        return VariantSearchHit(
            row_id=int(row["row_id"]),
            locode=str(row["locode"]),
            raw_score=float(score),
            score=weighted_score,
            search_text=str(row["search_text"]),
            variant_text=variant.text,
            variant_kind=variant.kind,
            variant_weight=variant.weight,
        )

    def _aggregate_hits(
        self,
        hits: list[VariantSearchHit],
    ) -> list[SearchHit]:
        hits_by_locode: dict[str, list[VariantSearchHit]] = {}

        for hit in hits:
            hits_by_locode.setdefault(hit.locode, []).append(hit)

        aggregated_hits: list[SearchHit] = []

        for locode, locode_hits in hits_by_locode.items():
            best_hit = max(locode_hits, key=lambda hit: hit.score)
            vote_count = len(locode_hits)
            aggregated_score = best_hit.score + self._config.aggregation_alpha * vote_count

            aggregated_hits.append(
                SearchHit(
                    row_id=best_hit.row_id,
                    locode=locode,
                    score=aggregated_score,
                    search_text=best_hit.search_text,
                )
            )

        return aggregated_hits

    def _top_hits(
        self,
        hits: list[SearchHit],
        *,
        top_k: int,
    ) -> list[SearchHit]:
        hits.sort(key=lambda hit: hit.score, reverse=True)
        return hits[:top_k]
