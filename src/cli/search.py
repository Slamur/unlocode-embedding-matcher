from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.config.paths import FAISS_INDEX_PATH, SEARCH_TEXTS_METADATA_PATH
from src.index.model import VectorIndex
from src.search.model import SearchRequest
from src.search.service import SearchService
from src.utils.files import read_parquet


def _read_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("query", type=str)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--index-path",
        type=Path,
        default=FAISS_INDEX_PATH,
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=SEARCH_TEXTS_METADATA_PATH,
    )

    return parser.parse_args()


def main() -> None:
    args = _read_args()

    index = VectorIndex.load(args.index_path)
    metadata = read_parquet(args.metadata_path)

    service = SearchService(index=index, metadata=metadata)
    response = service.search(request=SearchRequest(query=args.query, top_k=args.top_k))

    print(f"query: {response.query}")
    print(f"normalized_query: {response.normalized_query}")
    print()

    for i, hit in enumerate(response.hits, start=1):
        print(f"{i}. {hit.locode}  score={hit.score:.4f}")
        print(f"   alias_text: {hit.alias_text}")
        print(f"   search_text: {hit.search_text}")
        print(f"   kind: {hit.search_text_kind}")
        print(f"   country: {hit.country}")
        print(f"   subdivision: {hit.subdivision_name}")
        print()


if __name__ == "__main__":
    main()
