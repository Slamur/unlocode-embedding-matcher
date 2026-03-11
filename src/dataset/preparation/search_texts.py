import re

import pandas as pd

from src.dataset.inspect import inspect_df_info

_NON_ALNUM_PATTERN = re.compile(r"[^0-9a-z]+")
_MULTI_SPACE_PATTERN = re.compile(r"\s+")


def _normalize_search_text(text: str) -> str:
    normalized = text.lower().strip()
    normalized = _NON_ALNUM_PATTERN.sub(" ", normalized)
    normalized = _MULTI_SPACE_PATTERN.sub(" ", normalized)
    return normalized.strip()


def _build_search_text_variants(
    *,
    alias_text: str,
    country: str,
    subdivision_name: str,
) -> list[tuple[str, str]]:
    variants: list[tuple[str, str]] = []

    variants.append(("alias_only", alias_text))
    variants.append(("alias_country", f"{alias_text} {country}"))
    variants.append(("structured", f"location {alias_text} country {country}"))

    if subdivision_name:
        variants.append(
            (
                "alias_subdivision_country",
                f"{alias_text} {subdivision_name} {country}",
            )
        )
        variants.append(
            (
                "structured_with_subdivision",
                f"location {alias_text} subdivision {subdivision_name} country {country}",
            )
        )

    return variants


def _build_search_text_rows(
    aliases_with_locations: pd.DataFrame,
) -> list[tuple[str, str, str, str, str, str]]:
    rows: list[tuple[str, str, str, str, str, str]] = []

    for (
        locode,
        alias_text,
        country,
        subdivision_name,
    ) in aliases_with_locations.itertuples(index=False, name=None):
        variants = _build_search_text_variants(
            alias_text=alias_text,
            country=country,
            subdivision_name=subdivision_name,
        )

        for search_text_kind, raw_search_text in variants:
            search_text = _normalize_search_text(raw_search_text)
            rows.append(
                (
                    locode,
                    alias_text,
                    country,
                    subdivision_name,
                    search_text_kind,
                    search_text,
                )
            )

    return rows


def _join_aliases_with_locations(
    aliases: pd.DataFrame,
    locations: pd.DataFrame,
) -> pd.DataFrame:
    location_context = locations[["locode", "country", "subdivision_name"]].copy()

    joined = aliases.merge(
        location_context,
        how="inner",
        on="locode",
    )

    return joined


def _prepare_search_texts(
    search_texts: pd.DataFrame,
) -> pd.DataFrame:
    prepared = search_texts.copy()

    # strip all columns
    prepared["alias_text"] = prepared["alias_text"].str.strip()

    # TODO: add country name to/instead of country code
    prepared["country"] = prepared["country"].str.strip()

    prepared["subdivision_name"] = prepared["subdivision_name"].fillna("").str.strip()
    prepared["search_text_kind"] = prepared["search_text_kind"].str.strip()
    prepared["search_text"] = prepared["search_text"].str.strip()

    # remove rows without search text
    prepared = prepared[prepared["search_text"] != ""].copy()

    # any alias_text is taken after drop
    # retrieval-only table, so it's not a problem
    prepared = prepared.drop_duplicates(subset=["locode", "search_text"]).reset_index(drop=True)

    return prepared


def build_search_texts_table(
    *,
    aliases: pd.DataFrame,
    locations: pd.DataFrame,
    verbose: bool = False,
) -> pd.DataFrame:
    aliases_with_locations = _join_aliases_with_locations(
        aliases=aliases,
        locations=locations,
    )
    inspect_df_info(
        aliases_with_locations,
        "Aliases With Locations DataFrame",
        verbose=verbose,
    )

    rows = _build_search_text_rows(aliases_with_locations)
    search_texts = pd.DataFrame(
        rows,
        columns=[
            "locode",
            "alias_text",
            "country",
            "subdivision_name",
            "search_text_kind",
            "search_text",
        ],
    )
    inspect_df_info(search_texts, "Search Texts DataFrame", verbose=verbose)

    prepared_search_texts = _prepare_search_texts(search_texts)
    inspect_df_info(
        prepared_search_texts,
        "Prepared Search Texts DataFrame",
        verbose=verbose,
    )

    return prepared_search_texts
