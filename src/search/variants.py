def build_query_variants(normalized_query: str) -> list[str]:
    variants = []

    if normalized_query:
        variants.append(normalized_query)

    return variants
