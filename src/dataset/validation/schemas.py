import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

_LOCODE_PATTERN = r"^[A-Z]{2}[A-Z0-9]{3}$"


def required_string_column(*, min_length: int = 1) -> Column:
    return Column(
        pa.String,
        nullable=False,
        checks=[
            Check.str_length(min_value=min_length),
        ],
        coerce=True,
    )


def optional_string_column() -> Column:
    return Column(
        pa.String,
        nullable=True,
        required=False,
        coerce=True,
    )


def locode_column() -> Column:
    return Column(
        pa.String,
        nullable=False,
        checks=[
            Check.str_length(min_value=5, max_value=5),
            Check.str_matches(_LOCODE_PATTERN),
        ],
        coerce=True,
    )


LOCATIONS_SCHEMA = DataFrameSchema(
    columns={
        "locode": locode_column(),
        "country": required_string_column(),
        "code": required_string_column(),
        "subdivision_code": optional_string_column(),
        "subdivision_name": optional_string_column(),
    },
    unique=["locode"],
    strict=False,
    coerce=True,
)


ALIASES_SCHEMA = DataFrameSchema(
    columns={
        "locode": locode_column(),
        "alias_text": required_string_column(),
    },
    unique=["locode", "alias_text"],
    strict=False,
    coerce=True,
)

SEARCH_TEXTS_SCHEMA = DataFrameSchema(
    columns={
        "locode": locode_column(),
        "alias_text": required_string_column(),
        "country": required_string_column(),
        "subdivision_name": optional_string_column(),
        "search_text_kind": required_string_column(),
        "search_text": required_string_column(),
    },
    unique=["locode", "search_text"],
    strict=False,
    coerce=True,
)
