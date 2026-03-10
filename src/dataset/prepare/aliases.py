import pandas as pd

from src.dataset.logging import log_df_info
from src.dataset.prepare.split import build_splitted_rows


def _build_aliases(merged_codes: pd.DataFrame) -> pd.DataFrame:
    aliases = merged_codes[["locode", "name", "name_wo_diacritics"]].copy()

    return aliases


def _split_parenthesized_aliases(aliases: pd.DataFrame) -> pd.DataFrame:
    splitted_rows = build_splitted_rows(aliases)

    splitted_aliases = pd.DataFrame(
        splitted_rows,
        columns=["locode", "source_field", "alias_text", "alias_kind"],
    )

    return splitted_aliases


def _prepare_aliases(splitted_aliases: pd.DataFrame) -> pd.DataFrame:
    prepared_aliases = splitted_aliases[["locode", "alias_text"]].copy()

    # strip names
    prepared_aliases["alias_text"] = prepared_aliases["alias_text"].str.strip()

    # remove empty names
    prepared_aliases = prepared_aliases[prepared_aliases["alias_text"] != ""].copy()

    # remove duplicates
    prepared_aliases = prepared_aliases.drop_duplicates(
        subset=["locode", "alias_text"]
    ).reset_index(drop=True)

    return prepared_aliases


def resolve_aliases(
    merged_codes: pd.DataFrame, 
    verbose: bool = False,
) -> pd.DataFrame:

    aliases = _build_aliases(merged_codes=merged_codes)

    log_df_info(aliases, "Aliases DataFrame", verbose=verbose)

    splitted_aliases = _split_parenthesized_aliases(aliases=aliases)

    log_df_info(splitted_aliases, "Splitted Aliases DataFrame", verbose=verbose)

    prepared_splitted_aliases = _prepare_aliases(splitted_aliases=splitted_aliases)

    log_df_info(prepared_splitted_aliases, "Prepared Splitted Aliases DataFrame", verbose=verbose)

    return prepared_splitted_aliases
