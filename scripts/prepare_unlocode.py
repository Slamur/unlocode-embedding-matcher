from collections.abc import Sequence
from pathlib import Path

import re
import pandas as pd

from src.config.paths import INTERIM_DIR
from src.dataset.utils import print_df_info

class _AliasesResolver:

    @staticmethod
    def _build_aliases_df(merged_df: pd.DataFrame) -> pd.DataFrame:
        aliases_df = merged_df[["locode", "name", "name_wo_diacritics"]].copy()

        return aliases_df

    @staticmethod
    def _prepare_aliases_df(aliases_df: pd.DataFrame) -> pd.DataFrame:
        prepared_aliases_df = aliases_df.copy()

        return prepared_aliases_df

    @staticmethod
    def resolve_aliases(merged_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:

        aliases_df = _AliasesResolver._build_aliases_df(merged_df)

        print_df_info(aliases_df, "Aliases DataFrame", verbose=verbose)

        prepared_aliases_df = _AliasesResolver._prepare_aliases_df(aliases_df)

        print_df_info(prepared_aliases_df, "Prepared Aliases DataFrame", verbose=verbose)

        return prepared_aliases_df
    

class _AliasesSplitter:

    PARENTHESIZED_NAME_PATTERN = re.compile(r"^\s*(.*?)\s*\((.*?)\)\s*$")

    @staticmethod
    def _split_parenthesized_name_with_labels(name: str) -> list[tuple[str, str]]:
        name = name.strip()
        if not name:
            return []

        result: list[tuple[str, str]] = [(name, "full")]

        match = _AliasesSplitter.PARENTHESIZED_NAME_PATTERN.match(name)
        if not match:
            return result

        left = match.group(1).strip()
        inner = match.group(2).strip()

        if left:
            result.append((left, "paren_left"))

        if inner:
            result.append((inner, "paren_inner"))

        return result

    @staticmethod
    def _build_splitted_rows(aliases_df: pd.DataFrame) -> list[tuple[str, str, str, str]]:
        splitted_rows: list[tuple[str, str, str, str]] = []

        def add_rows_for(locode: str, source_field: str, field_value: str) -> None:
            for alias_text, alias_kind in _AliasesSplitter._split_parenthesized_name_with_labels(field_value):
                splitted_rows.append((locode, source_field, alias_text, alias_kind))

        for locode, name, name_wo_diacritics in aliases_df.itertuples(index=False, name=None):
            add_rows_for(locode, "name", name)
            add_rows_for(locode, "name_wo_diacritics", name_wo_diacritics)

        return splitted_rows
    
    @staticmethod
    def _split_parenthesized_aliases(aliases_df: pd.DataFrame) -> pd.DataFrame:
        splitted_rows = _AliasesSplitter._build_splitted_rows(aliases_df)

        splitted_aliases_df = pd.DataFrame(
            splitted_rows,
            columns=["locode", "source_field", "alias_text", "alias_kind"],
        )

        return splitted_aliases_df

    @staticmethod
    def _prepare_aliases_df(splitted_aliases_df: pd.DataFrame) -> pd.DataFrame:
        prepared_aliases_df = splitted_aliases_df.copy()

        # strip names
        prepared_aliases_df["alias_text"] = prepared_aliases_df["alias_text"].str.strip()

        # remove empty names
        prepared_aliases_df = prepared_aliases_df[prepared_aliases_df["alias_text"] != ""].copy()

        # remove duplicates
        prepared_aliases_df = prepared_aliases_df.drop_duplicates(
            subset=["locode", "source_field", "alias_text", "alias_kind"]
        ).reset_index(drop=True)

        return prepared_aliases_df

    @staticmethod
    def split_aliases(aliases_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:

        splitted_aliases_df = _AliasesSplitter._split_parenthesized_aliases(aliases_df)

        print_df_info(splitted_aliases_df, "Splitted Aliases DataFrame", verbose=verbose)

        prepared_splitted_aliases_df = _AliasesSplitter._prepare_aliases_df(splitted_aliases_df)

        print_df_info(prepared_splitted_aliases_df, "Prepared Splitted Aliases DataFrame", verbose=verbose)

        return prepared_splitted_aliases_df


class _LocationsResolver:

    @staticmethod
    def _build_locations_df(merged_df: pd.DataFrame) -> pd.DataFrame:
        locations_df = merged_df.drop(columns=["name", "name_wo_diacritics"]).copy()

        return locations_df

    @staticmethod
    def _prepare_locations_df(locations_df: pd.DataFrame) -> pd.DataFrame:
        prepared_locations_df = locations_df.copy()

        # remove full locode duplicates
        prepared_locations_df = prepared_locations_df.drop_duplicates(subset=["locode"])

        return prepared_locations_df

    @staticmethod
    def resolve_locations(merged_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        locations_df = _LocationsResolver._build_locations_df(merged_df)

        print_df_info(locations_df, "Locations DataFrame", verbose=verbose)

        prepared_locations_df = _LocationsResolver._prepare_locations_df(locations_df)

        print_df_info(prepared_locations_df, "Prepared Locations DataFrame", verbose=verbose)

        return prepared_locations_df



def main() -> None:

    merged_codes_path = INTERIM_DIR / "merged_codes.parquet"
    merged_codes_df = pd.read_parquet(merged_codes_path)

    locations_df = _LocationsResolver.resolve_locations(merged_codes_df)

    aliases_df = _AliasesResolver.resolve_aliases(merged_codes_df)
    splitted_aliases_df = _AliasesSplitter.split_aliases(aliases_df)

    print("\nAliases for BEBRU:")
    print(aliases_df[aliases_df["locode"] == "BEBRU"].to_string())

    print("\nSplitted Aliases for BEBRU:")
    print(splitted_aliases_df[splitted_aliases_df["locode"] == "BEBRU"].to_string())


if __name__ == "__main__":
    main()