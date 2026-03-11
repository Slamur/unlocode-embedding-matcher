import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.dataset.validation.validate_aliases import validate_aliases


def test_validate_aliases_accepts_valid_dataframe() -> None:
    validated_locations = pd.DataFrame(
        {
            "locode": ["KGFRU", "BEBRU"],
            "country": ["KG", "BE"],
            "code": ["FRU", "BRU"],
            "subdivision_code": [pd.NA, "BRU"],
            "subdivision_name": [pd.NA, "Brussels-Capital Region"],
        }
    )
    aliases = pd.DataFrame(
        {
            "locode": ["KGFRU", "BEBRU", "BEBRU"],
            "alias_text": ["Bishkek", "Brussel", "Bruxelles"],
        }
    )

    result = validate_aliases(aliases, validated_locations)

    assert list(result["alias_text"]) == ["Bishkek", "Brussel", "Bruxelles"]


def test_validate_aliases_trims_string_values() -> None:
    validated_locations = pd.DataFrame(
        {
            "locode": ["KGFRU"],
            "country": ["KG"],
            "code": ["FRU"],
            "subdivision_code": [pd.NA],
            "subdivision_name": [pd.NA],
        }
    )
    aliases = pd.DataFrame(
        {
            "locode": ["  KGFRU  "],
            "alias_text": ["  Bishkek  "],
        }
    )

    result = validate_aliases(aliases, validated_locations)

    assert result.iloc[0]["locode"] == "KGFRU"
    assert result.iloc[0]["alias_text"] == "Bishkek"


def test_validate_aliases_rejects_empty_alias_after_trim() -> None:
    aliases = pd.DataFrame(
        {
            "locode": ["KGFRU"],
            "alias_text": ["   "],
        }
    )

    with pytest.raises(ValueError, match="aliases.alias_text contains 1 empty value"):
        validate_aliases(aliases)


def test_validate_aliases_rejects_unknown_locode() -> None:
    validated_locations = pd.DataFrame(
        {
            "locode": ["KGFRU"],
            "country": ["KG"],
            "code": ["FRU"],
            "subdivision_code": [pd.NA],
            "subdivision_name": [pd.NA],
        }
    )
    aliases = pd.DataFrame(
        {
            "locode": ["KGFRU", "USNYC"],
            "alias_text": ["Bishkek", "New York"],
        }
    )

    with pytest.raises(ValueError, match="aliases contain locodes absent from locations"):
        validate_aliases(aliases, validated_locations)


def test_validate_aliases_rejects_duplicate_alias() -> None:
    validated_locations = pd.DataFrame(
        {
            "locode": ["KGFRU"],
            "country": ["KG"],
            "code": ["FRU"],
            "subdivision_code": [pd.NA],
            "subdivision_name": [pd.NA],
        }
    )

    aliases = pd.DataFrame(
        {
            "locode": ["KGFRU", "KGFRU"],
            "alias_text": ["Bishkek", "Bishkek"],
        }
    )

    with pytest.raises(SchemaErrors):
        validate_aliases(aliases, validated_locations)
