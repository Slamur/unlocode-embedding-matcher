import re

_PART_SPLIT_PATTERN = re.compile(r"[,;/]+")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")
_DIGIT_TOKEN_PATTERN = re.compile(r"\S*\d\S*")


def _clean_text(text: str) -> str:
    return _MULTI_SPACE_PATTERN.sub(" ", text).strip()


def _remove_tokens_with_digits(text: str) -> str:
    tokens = text.split()
    tokens = [token for token in tokens if not _DIGIT_TOKEN_PATTERN.fullmatch(token)]
    return " ".join(tokens)


def _deduplicate_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for item in items:
        if not item or item in seen:
            continue

        seen.add(item)
        result.append(item)

    return result


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


def build_query_variants(normalized_query: str) -> list[str]:
    normalized_query = _clean_text(normalized_query)

    if not normalized_query:
        return []

    variants: list[str] = [normalized_query]

    parts = _split_query_parts(normalized_query)

    if len(parts) == 1:
        variants.append(f"location {parts[0]}")
    else:
        first = parts[0]
        second = parts[1]
        penultimate = parts[-2]
        last = parts[-1]

        variants.append(f"location {first}")
        variants.append(f"location {second}")
        variants.append(f"location {second} country {first}")
        variants.append(f"location {penultimate}")
        variants.append(f"location {penultimate} country {last}")
        variants.append(f"location {last}")

    return _deduplicate_preserve_order(variants)
