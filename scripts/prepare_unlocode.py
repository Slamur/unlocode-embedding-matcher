from collections.abc import Sequence
from pathlib import Path

import re
import pandas as pd

from src.config.paths import RAW_DIR


UNLOCODE_COLUMNS = [
    "change",
    "country",
    "code",
    "name",
    "name_wo_diacritics",
    "subdivision",
    "function",
    "status",
    "date",
    "iata",
    "coordinates",
    "remarks",
]

SUBDIVISION_COLUMNS = [
    "country",
    "subdivision_code",
    "subdivision_name",
    "subdivision_type",
]


def _find_csv_files(directory: Path, substring: str) -> Sequence[Path]:
    files = sorted(directory.glob(f"*{substring}*.csv"))

    if not files:
        raise RuntimeError(f"No files matching substring '{substring}' found in {directory}")
    
    return files


def _read_csv_file(path: Path, column_names: list[str] | None = None) -> pd.DataFrame:
    return pd.read_csv(
        path,
        header=None,
        names=column_names,
        dtype=str,
        keep_default_na=False,
        encoding="cp1252",
    )


def _print_df_info(df: pd.DataFrame, name: str, verbose: bool = False) -> None:
    if not verbose:
        return

    print(f"{name}:")
    print(f"Shape: {df.shape}")
    print(df.head().to_string())
    print()


class _CodesPreparer:

    @staticmethod
    def _read_codes(csv_dir: Path = RAW_DIR) -> pd.DataFrame:
        part_files = _find_csv_files(csv_dir, "CodeListPart")

        parts = [_read_csv_file(path, column_names=UNLOCODE_COLUMNS) for path in part_files]
        return pd.concat(parts, ignore_index=True)

    @staticmethod
    def _prepare_codes_df(codes_df: pd.DataFrame) -> pd.DataFrame:
        prepared_codes_df = codes_df.copy()

        # pure string columns
        for col in ("country", "code", "name", "name_wo_diacritics", "subdivision"):
            prepared_codes_df[col] = prepared_codes_df[col].str.strip()

        # filter columns without country and/or code
        prepared_codes_df = prepared_codes_df[(prepared_codes_df["country"] != "") & (prepared_codes_df["code"] != "")].copy()

        # generate locode columns
        prepared_codes_df["locode"] = prepared_codes_df["country"] + prepared_codes_df["code"]
        prepared_codes_df["locode_display"] = prepared_codes_df["country"] + " " + prepared_codes_df["code"]

        return prepared_codes_df

    @staticmethod
    def read_prepared_codes_df(verbose: bool = False) -> pd.DataFrame:
        codes_df = _CodesPreparer._read_codes()

        _print_df_info(codes_df, "Codes", verbose=verbose)

        prepared_codes_df = _CodesPreparer._prepare_codes_df(codes_df)

        _print_df_info(prepared_codes_df, "Prepared Codes", verbose=verbose)

        return prepared_codes_df


class _SubdivisionsPreparer:

    @staticmethod
    def _read_subdivisions(csv_dir: Path = RAW_DIR) -> pd.DataFrame:
        subdivision_files = _find_csv_files(csv_dir, "Subdivision")
        if not subdivision_files:
            raise RuntimeError(f"No subdivision CSV found in {csv_dir}")

        return _read_csv_file(subdivision_files[0], column_names=SUBDIVISION_COLUMNS)

    @staticmethod
    def _prepare_subdivisions(
        subdivisions_df: pd.DataFrame,
    ) -> pd.DataFrame:
        prepared_subdivisions_df = subdivisions_df.copy()

        # pure string columns
        for col in ("country", "subdivision_code", "subdivision_name", "subdivision_type"):
            prepared_subdivisions_df[col] = prepared_subdivisions_df[col].str.strip()

        return prepared_subdivisions_df
    
    @staticmethod
    def read_prepared_subdivisions_df(verbose: bool = False) -> pd.DataFrame:
        subdivisions_df = _SubdivisionsPreparer._read_subdivisions()

        _print_df_info(subdivisions_df, "Subdivisions", verbose=verbose)

        prepared_subdivisions_df = _SubdivisionsPreparer._prepare_subdivisions(subdivisions_df)

        _print_df_info(prepared_subdivisions_df, "Prepared Subdivisions", verbose=verbose)

        return prepared_subdivisions_df


class _DataMerger:

    @staticmethod
    def _merge_codes_with_subdivisions(codes_df: pd.DataFrame, subdivisions_df: pd.DataFrame) -> pd.DataFrame:
        merged_df = codes_df.merge(
            subdivisions_df,
            how="left",
            left_on=["country", "subdivision"],
            right_on=["country", "subdivision_code"],
            suffixes=("", "_subdiv"),
        )

        return merged_df
    
    @staticmethod
    def _prepare_merged_df(merged_df: pd.DataFrame) -> pd.DataFrame:
        prepared_merged_df = merged_df.copy()

        for col in ["subdivision_code", "subdivision_name", "subdivision_type"]:
            prepared_merged_df[col] = prepared_merged_df[col].fillna("")

        return prepared_merged_df
    
    @staticmethod
    def merge_and_prepare(codes_df: pd.DataFrame, subdivisions_df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        merged_df = _DataMerger._merge_codes_with_subdivisions(codes_df, subdivisions_df)
        prepared_merged_df = _DataMerger._prepare_merged_df(merged_df)

        _print_df_info(merged_df, "Merged DataFrame", verbose=verbose)

        return prepared_merged_df


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

        _print_df_info(aliases_df, "Aliases DataFrame", verbose=verbose)

        prepared_aliases_df = _AliasesResolver._prepare_aliases_df(aliases_df)

        _print_df_info(prepared_aliases_df, "Prepared Aliases DataFrame", verbose=verbose)

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

        _print_df_info(splitted_aliases_df, "Splitted Aliases DataFrame", verbose=verbose)

        prepared_splitted_aliases_df = _AliasesSplitter._prepare_aliases_df(splitted_aliases_df)

        _print_df_info(prepared_splitted_aliases_df, "Prepared Splitted Aliases DataFrame", verbose=verbose)

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

        _print_df_info(locations_df, "Locations DataFrame", verbose=verbose)

        prepared_locations_df = _LocationsResolver._prepare_locations_df(locations_df)

        _print_df_info(prepared_locations_df, "Prepared Locations DataFrame", verbose=verbose)

        return prepared_locations_df



def main() -> None:

    codes_df = _CodesPreparer.read_prepared_codes_df()
    subdivisions_df = _SubdivisionsPreparer.read_prepared_subdivisions_df()

    merged_df = _DataMerger.merge_and_prepare(codes_df, subdivisions_df)

    locations_df = _LocationsResolver.resolve_locations(merged_df)

    aliases_df = _AliasesResolver.resolve_aliases(merged_df, verbose=True)
    splitted_aliases_df = _AliasesSplitter.split_aliases(aliases_df, verbose=True)

    print("\nAliases for BEBRU:")
    print(aliases_df[aliases_df["locode"] == "BEBRU"].to_string())

    print("\nSplitted Aliases for BEBRU:")
    print(splitted_aliases_df[splitted_aliases_df["locode"] == "BEBRU"].to_string())


if __name__ == "__main__":
    main()