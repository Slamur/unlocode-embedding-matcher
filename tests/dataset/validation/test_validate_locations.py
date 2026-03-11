import pandas as pd
import pytest
from pandera.errors import SchemaErrors

from src.dataset.validation.validate_locations import validate_locations


def test_validate_locations_accepts_valid_dataframe() -> None:
    locations = pd.DataFrame(
        {
            "locode": ["KGFRU", "BEBRU"],
            "country": ["KG", "BE"],
            "code": ["FRU", "BRU"],
            "subdivision_code": [pd.NA, "BRU"],
            "subdivision_name": [pd.NA, "Brussels-Capital Region"],
        }
    )

    result = validate_locations(locations)

    assert list(result["locode"]) == ["KGFRU", "BEBRU"]


def test_validate_locations_rejects_duplicate_locode() -> None:
    locations = pd.DataFrame(
        {
            "locode": ["KGFRU", "KGFRU"],
            "country": ["KG", "KG"],
            "code": ["FRU", "FRU"],
            "subdivision_code": [pd.NA, pd.NA],
            "subdivision_name": [pd.NA, pd.NA],
        }
    )

    with pytest.raises(SchemaErrors):
        validate_locations(locations)


def test_validate_locations_rejects_invalid_locode_format() -> None:
    locations = pd.DataFrame(
        {
            "locode": ["KG12"],
            "country": ["KG"],
            "code": ["12"],
            "subdivision_code": [pd.NA],
            "subdivision_name": [pd.NA],
        }
    )

    with pytest.raises(SchemaErrors):
        validate_locations(locations)


def test_validate_locations_rejects_locode_not_equal_country_plus_code() -> None:
    locations = pd.DataFrame(
        {
            "locode": ["KGOSS"],
            "country": ["KG"],
            "code": ["FRU"],
            "subdivision_code": [pd.NA],
            "subdivision_name": [pd.NA],
        }
    )

    with pytest.raises(ValueError, match="locations.locode must be equal to country \\+ code"):
        validate_locations(locations)
