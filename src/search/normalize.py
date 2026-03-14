import re
import unicodedata

_NON_WORD_PATTERN = re.compile(r"[^\w]+", flags=re.UNICODE)
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


def normalize_query(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.lower().strip()
    normalized = _NON_WORD_PATTERN.sub(" ", normalized)
    normalized = _MULTI_SPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def build_query_variants(query: str) -> list[str]:
    variants = []

    normalized = normalize_query(query)
    if normalized:
        variants.append(normalized)

    return variants
