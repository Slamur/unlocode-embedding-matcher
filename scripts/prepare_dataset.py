import pandas as pd

from src.config.paths import INTERIM_DIR, PROCESSED_DIR
from src.dataset.preparation.aliases import build_aliases_table
from src.dataset.preparation.locations import build_locations_table
from src.dataset.preparation.search_texts import build_search_texts_table
from src.dataset.validation.validate_aliases import validate_aliases
from src.dataset.validation.validate_locations import validate_locations
from src.dataset.validation.validate_search_texts import validate_search_texts


def main() -> None:
    merged_codes_path = INTERIM_DIR / "merged_codes.parquet"
    if not merged_codes_path.exists():
        raise FileNotFoundError(f"Expected merged codes file not found at: {merged_codes_path}")

    merged_codes = pd.read_parquet(merged_codes_path)

    locations = build_locations_table(merged_codes=merged_codes)
    aliases = build_aliases_table(merged_codes=merged_codes)
    search_texts = build_search_texts_table(
        aliases=aliases,
        locations=locations,
    )

    locations = validate_locations(locations=locations)
    aliases = validate_aliases(aliases=aliases, locations=locations)
    search_texts = validate_search_texts(
        search_texts=search_texts,
        locations=locations,
    )

    locations_path = PROCESSED_DIR / "unlocode_locations.parquet"
    aliases_path = PROCESSED_DIR / "unlocode_aliases.parquet"
    search_texts_path = PROCESSED_DIR / "unlocode_search_texts.parquet"

    locations.to_parquet(locations_path, index=False)
    aliases.to_parquet(aliases_path, index=False)
    search_texts.to_parquet(search_texts_path, index=False)

    print(f"Saved locations to: {locations_path}")
    print(f"Shape: {locations.shape}")

    print(f"Saved aliases to: {aliases_path}")
    print(f"Shape: {aliases.shape}")

    print(f"Saved search texts to: {search_texts_path}")
    print(f"Shape: {search_texts.shape}")


if __name__ == "__main__":
    main()
