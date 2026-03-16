import re
from dataclasses import dataclass

_PART_SPLIT_PATTERN = re.compile(r"[,;/]+")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")
_DIGIT_TOKEN_PATTERN = re.compile(r"\S*\d\S*")


@dataclass(frozen=True)
class QueryVariant:
    text: str
    kind: str
    weight: float


def _clean_text(text: str) -> str:
    return _MULTI_SPACE_PATTERN.sub(" ", text).strip()


def _remove_tokens_with_digits(text: str) -> str:
    tokens = text.split()
    tokens = [token for token in tokens if not _DIGIT_TOKEN_PATTERN.fullmatch(token)]
    return " ".join(tokens)


def _deduplicate_variants(variants: list[QueryVariant]) -> list[QueryVariant]:
    best_by_text: dict[str, QueryVariant] = {}
    order: list[str] = []

    for variant in variants:
        if not variant.text:
            continue

        existing = best_by_text.get(variant.text)

        if existing is None:
            best_by_text[variant.text] = variant
            order.append(variant.text)
            continue

        if variant.weight > existing.weight:
            best_by_text[variant.text] = variant

    return [best_by_text[text] for text in order]


def _split_query_parts(normalized_query: str) -> list[str]:
    has_punctuation_split = bool(_PART_SPLIT_PATTERN.search(normalized_query))

    if has_punctuation_split:
        raw_parts = _PART_SPLIT_PATTERN.split(normalized_query)
    else:
        raw_parts = normalized_query.split()

    parts: list[str] = []

    for part in raw_parts:
        cleaned_part = _clean_text(part)
        cleaned_part = _remove_tokens_with_digits(cleaned_part)
        cleaned_part = _clean_text(cleaned_part)

        if cleaned_part:
            parts.append(cleaned_part)

    return parts


def _build_variants_from_parts(parts: list[str]) -> list[QueryVariant]:
    variants: list[QueryVariant] = []

    if not parts:
        return variants

    first = parts[0]
    last = parts[-1]

    variants.append(
        QueryVariant(
            text=f"location {first}",
            kind="forward_city_first",
            weight=1.03,
        )
    )
    variants.append(
        QueryVariant(
            text=f"location {last}",
            kind="reverse_city_first",
            weight=1.03,
        )
    )

    if len(parts) > 1:
        second = parts[1]
        penultimate = parts[-2]

        variants.append(
            QueryVariant(
                text=f"location {second}",
                kind="forward_city_second",
                weight=1.07,
            )
        )
        variants.append(
            QueryVariant(
                text=f"location {second} country {first}",
                kind="forward_city_second_country_first",
                weight=1.10,
            )
        )

        variants.append(
            QueryVariant(
                text=f"location {penultimate}",
                kind="reverse_city_second",
                weight=1.07,
            )
        )
        variants.append(
            QueryVariant(
                text=f"location {penultimate} country {last}",
                kind="reverse_city_second_country_first",
                weight=1.10,
            )
        )

    return variants


def build_query_variants(normalized_query: str) -> list[QueryVariant]:
    normalized_query = _clean_text(normalized_query)

    if not normalized_query:
        return []

    variants: list[QueryVariant] = [
        QueryVariant(
            text=normalized_query,
            kind="full_query",
            weight=1.00,
        )
    ]

    parts = _split_query_parts(normalized_query)
    variants.extend(_build_variants_from_parts(parts=parts))

    return _deduplicate_variants(variants)
