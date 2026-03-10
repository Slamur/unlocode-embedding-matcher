import re
import pandas as pd

from src.config.paths import INTERIM_DIR, PROCESSED_DIR
from src.dataset.logging import log_df_info
from src.dataset.prepare.locations import resolve_locations

class _AliasesResolver:

    @staticmethod
    def _build_aliases(merged_df: pd.DataFrame) -> pd.DataFrame:
        aliases = merged_df[["locode", "name", "name_wo_diacritics"]].copy()

        return aliases

    @staticmethod
    def _prepare_aliases(aliases: pd.DataFrame) -> pd.DataFrame:
        prepared_aliases = aliases.copy()

        return prepared_aliases

    @staticmethod
    def resolve_aliases(merged_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:

        aliases = _AliasesResolver._build_aliases(merged_df)

        log_df_info(aliases, "Aliases DataFrame", verbose=verbose)

        prepared_aliases = _AliasesResolver._prepare_aliases(aliases)

        log_df_info(prepared_aliases, "Prepared Aliases DataFrame", verbose=verbose)

        return prepared_aliases
    

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
    def _build_splitted_rows(aliases: pd.DataFrame) -> list[tuple[str, str, str, str]]:
        splitted_rows: list[tuple[str, str, str, str]] = []

        def add_rows_for(locode: str, source_field: str, field_value: str) -> None:
            for alias_text, alias_kind in _AliasesSplitter._split_parenthesized_name_with_labels(field_value):
                splitted_rows.append((locode, source_field, alias_text, alias_kind))

        for locode, name, name_wo_diacritics in aliases.itertuples(index=False, name=None):
            add_rows_for(locode, "name", name)
            add_rows_for(locode, "name_wo_diacritics", name_wo_diacritics)

        return splitted_rows
    
    @staticmethod
    def _split_parenthesized_aliases(aliases: pd.DataFrame) -> pd.DataFrame:
        splitted_rows = _AliasesSplitter._build_splitted_rows(aliases)

        splitted_aliases = pd.DataFrame(
            splitted_rows,
            columns=["locode", "source_field", "alias_text", "alias_kind"],
        )

        return splitted_aliases

    @staticmethod
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

    @staticmethod
    def split_aliases(aliases: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:

        splitted_aliases = _AliasesSplitter._split_parenthesized_aliases(aliases)

        log_df_info(splitted_aliases, "Splitted Aliases DataFrame", verbose=verbose)

        prepared_splitted_aliases = _AliasesSplitter._prepare_aliases(splitted_aliases)

        log_df_info(prepared_splitted_aliases, "Prepared Splitted Aliases DataFrame", verbose=verbose)

        return prepared_splitted_aliases


def main() -> None:

    merged_codes_path = INTERIM_DIR / "merged_codes.parquet"
    merged_codes = pd.read_parquet(merged_codes_path)

    locations = resolve_locations(merged_codes)

    aliases = _AliasesResolver.resolve_aliases(merged_codes)
    splitted_aliases = _AliasesSplitter.split_aliases(aliases)

    locations_path = PROCESSED_DIR / "unlocode_locations.parquet"
    aliases_path = PROCESSED_DIR / "unlocode_aliases.parquet"

    locations.to_parquet(locations_path, index=False)
    splitted_aliases.to_parquet(aliases_path, index=False)

    print(f"Saved locations to: {locations_path}")
    print(f"Saved aliases to: {aliases_path}")


if __name__ == "__main__":
    main()
