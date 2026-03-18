from dataclasses import dataclass


@dataclass(frozen=True)
class SearchRequest:
    query: str
    top_k: int


@dataclass(frozen=True)
class SearchHit:
    row_id: int
    locode: str
    score: float
    search_text: str

    def __str__(self) -> str:
        return f"[{self.locode}] score={self.score:.4f}\n" f"  {self.search_text}"


@dataclass(frozen=True)
class SearchResponse:
    query: str
    normalized_query: str
    hits: list[SearchHit]
