import re
import pandas as pd


_PARENTHESIZED_NAME_PATTERN = re.compile(r"^\s*(.*?)\s*\((.*?)\)\s*$")


def _split_parenthesized_name_with_labels(
    name: str,
) -> list[tuple[str, str]]:
    name = name.strip()
    if not name:
        return []

    result: list[tuple[str, str]] = [(name, "full")]

    match = _PARENTHESIZED_NAME_PATTERN.match(name)
    if not match:
        return result

    left = match.group(1).strip()
    inner = match.group(2).strip()

    if left:
        result.append((left, "paren_left"))

    if inner:
        result.append((inner, "paren_inner"))

    return result


def build_expanded_rows(
    aliases: pd.DataFrame,
) -> list[tuple[str, str, str, str]]:
    expanded_rows: list[tuple[str, str, str, str]] = []

    def add_rows_for(locode: str, source_field: str, field_value: str) -> None:
        for alias_text, alias_kind in _split_parenthesized_name_with_labels(field_value):
            expanded_rows.append((locode, source_field, alias_text, alias_kind))

    for locode, name, name_wo_diacritics in aliases.itertuples(index=False, name=None):
        add_rows_for(locode, "name", name)
        add_rows_for(locode, "name_wo_diacritics", name_wo_diacritics)

    return expanded_rows
