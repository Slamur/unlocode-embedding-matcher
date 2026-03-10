import pandas as pd

from src.dataset.inspect import inspect_df_info
from src.dataset.prepare.expand import build_expanded_rows


def _build_aliases(
    merged_codes: pd.DataFrame,
) -> pd.DataFrame:
    aliases = merged_codes[["locode", "name", "name_wo_diacritics"]].copy()

    return aliases


def _expand_aliases(
    aliases: pd.DataFrame,
) -> pd.DataFrame:
    expanded_rows = build_expanded_rows(aliases)

    expanded_aliases = pd.DataFrame(
        expanded_rows,
        columns=["locode", "source_field", "alias_text", "alias_kind"],
    )

    return expanded_aliases


def _prepare_aliases(
    expanded_aliases: pd.DataFrame,
) -> pd.DataFrame:
    prepared_aliases = expanded_aliases[["locode", "alias_text"]].copy()

    # strip names
    prepared_aliases["alias_text"] = prepared_aliases["alias_text"].str.strip()

    # remove empty names
    prepared_aliases = prepared_aliases[prepared_aliases["alias_text"] != ""].copy()

    # remove duplicates
    prepared_aliases = prepared_aliases.drop_duplicates(
        subset=["locode", "alias_text"]
    ).reset_index(drop=True)

    return prepared_aliases


def build_aliases_table(
    merged_codes: pd.DataFrame, 
    verbose: bool = False,
) -> pd.DataFrame:

    aliases = _build_aliases(merged_codes=merged_codes)

    inspect_df_info(aliases, "Aliases DataFrame", verbose=verbose)

    expanded_aliases = _expand_aliases(aliases=aliases)

    inspect_df_info(expanded_aliases, "Expanded Aliases DataFrame", verbose=verbose)

    prepared_aliases = _prepare_aliases(expanded_aliases=expanded_aliases)

    inspect_df_info(prepared_aliases, "Prepared Aliases DataFrame", verbose=verbose)

    return prepared_aliases
