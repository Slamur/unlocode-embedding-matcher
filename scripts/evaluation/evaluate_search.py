import argparse
from dataclasses import dataclass
from pathlib import Path

import yaml

from src.config.paths import FAISS_INDEX_PATH, SEARCH_TEXTS_METADATA_PATH
from src.index.model import VectorIndex
from src.search.model import SearchHit, SearchRequest
from src.search.service import SearchService
from src.utils.files import read_parquet


@dataclass(frozen=True)
class EvalCase:
    query: str
    expected_locodes: list[str]


@dataclass(frozen=True)
class EvalResult:
    query: str
    expected_locodes: list[str]
    actual_hits: list[SearchHit]
    found_rank: int | None

    @property
    def actual_locodes(self) -> list[str]:
        return [hit.locode for hit in self.actual_hits]

    def status(self) -> str:
        if self.found_rank == 1:
            return "OK"
        if self.found_rank is None:
            return "MISS"
        return "PARTIAL"

    def to_pretty_string(self, index: int | None = None) -> str:
        lines: list[str] = []

        prefix = f"{index}. " if index is not None else ""
        lines.append(f"{prefix}[{self.status()}] {self.query}")
        lines.append(f"   expected: {', '.join(self.expected_locodes)}")

        if self.found_rank is None:
            lines.append("   found at: not found")
        else:
            lines.append(f"   found at: rank {self.found_rank}")

        if not self.actual_hits:
            lines.append("   actual: <no hits>")
            return "\n".join(lines)

        expected_set = set(self.expected_locodes)

        lines.append("   actual:")
        for rank, hit in enumerate(self.actual_hits, start=1):
            marker = " *" if hit.locode in expected_set else ""
            lines.append(f"     {rank}. {hit.locode}  score={hit.score:.4f}{marker}")
            lines.append(f"        search_text: {hit.search_text}")

        return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate search quality on a small labeled dataset."
    )
    parser.add_argument(
        "--cases",
        required=True,
        help="Path to YAML file with eval cases.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="How many search hits to request for each query.",
    )
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


def _load_cases(path: str) -> list[EvalCase]:
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError("Eval file must contain a top-level mapping")

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError("Eval file must contain 'cases' list")

    cases: list[EvalCase] = []

    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise ValueError(f"Case #{index} must be a mapping")

        query = raw_case.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Case #{index} must contain non-empty 'query'")

        expected_locodes = raw_case.get("expected_locodes")
        expected_locode = raw_case.get("expected_locode")

        if expected_locodes is not None and expected_locode is not None:
            raise ValueError(
                f"Case #{index} must contain either "
                f"'expected_locode' or 'expected_locodes', not both"
            )

        if expected_locodes is None and expected_locode is None:
            raise ValueError(f"Case #{index} must contain 'expected_locode' or 'expected_locodes'")

        if expected_locode is not None:
            if not isinstance(expected_locode, str) or not expected_locode.strip():
                raise ValueError(f"Case #{index} has invalid 'expected_locode'")
            normalized_expected_locodes = [expected_locode.strip()]
        else:
            if not isinstance(expected_locodes, list) or not expected_locodes:
                raise ValueError(f"Case #{index} has invalid 'expected_locodes'")

            normalized_expected_locodes = []
            for locode in expected_locodes:
                if not isinstance(locode, str) or not locode.strip():
                    raise ValueError(f"Case #{index} has invalid locode in 'expected_locodes'")
                normalized_expected_locodes.append(locode.strip())

        cases.append(
            EvalCase(
                query=query.strip(),
                expected_locodes=normalized_expected_locodes,
            )
        )

    return cases


def _find_rank(actual_locodes: list[str], expected_locodes: list[str]) -> int | None:
    expected_set = set(expected_locodes)

    for index, locode in enumerate(actual_locodes, start=1):
        if locode in expected_set:
            return index

    return None


def _evaluate_case(
    search_service: SearchService,
    case: EvalCase,
    *,
    top_k: int,
) -> EvalResult:
    response = search_service.search(
        SearchRequest(
            query=case.query,
            top_k=top_k,
        )
    )

    actual_locodes = [hit.locode for hit in response.hits]
    found_rank = _find_rank(
        actual_locodes=actual_locodes,
        expected_locodes=case.expected_locodes,
    )

    return EvalResult(
        query=case.query,
        expected_locodes=case.expected_locodes,
        actual_hits=response.hits,
        found_rank=found_rank,
    )


def _format_expected(expected_locodes: list[str]) -> str:
    return ", ".join(expected_locodes)


def _extract_status(result: EvalResult) -> str:
    if result.found_rank == 1:
        return "OK"
    elif result.found_rank is None:
        return "MISS"
    else:
        return "PARTIAL"


def _print_summary(results: list[EvalResult]) -> None:
    total = len(results)
    top1 = sum(1 for result in results if result.found_rank == 1)
    top3 = sum(1 for result in results if result.found_rank is not None and result.found_rank <= 3)
    top5 = sum(1 for result in results if result.found_rank is not None and result.found_rank <= 5)
    missed = sum(1 for result in results if result.found_rank is None)

    print("Evaluation summary")
    print("------------------")
    print(f"Total cases: {total}")
    print(f"Top-1: {top1}/{total} = {top1 / total:.2%}")
    print(f"Top-3: {top3}/{total} = {top3 / total:.2%}")
    print(f"Top-5: {top5}/{total} = {top5 / total:.2%}")
    print(f"Missed: {missed}/{total} = {missed / total:.2%}")
    print()

    print("Per-case results")
    print("----------------")

    for index, result in enumerate(results, start=1):
        print(result.to_pretty_string(index=index))
        print()


def main() -> None:
    args = _parse_args()

    index = VectorIndex.load(args.index_path)
    metadata = read_parquet(args.metadata_path)

    search_service = SearchService(index=index, metadata=metadata)

    cases = _load_cases(args.cases)

    results = [
        _evaluate_case(
            search_service=search_service,
            case=case,
            top_k=args.top_k,
        )
        for case in cases
    ]

    _print_summary(results)


if __name__ == "__main__":
    main()
