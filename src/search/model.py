from dataclasses import dataclass


@dataclass(frozen=True)
class SearchRequest:
    query: str
    top_k: int


@dataclass(frozen=True)
class SearchHit:
    locode: str
    score: float
    search_text: str
    alias_text: str
    country: str
    subdivision_name: str
    search_text_kind: str


@dataclass(frozen=True)
class SearchResponse:
    query: str
    normalized_query: str
    hits: list[SearchHit]
