from src.config.paths import ALIASES_PATH, LOCATIONS_PATH, MERGED_CODES_PATH, SEARCH_TEXTS_PATH
from src.dataset.preparation.aliases import build_aliases_table
from src.dataset.preparation.locations import build_locations_table
from src.dataset.preparation.search_texts import build_search_texts_table
from src.dataset.validation.validate_aliases import validate_aliases
from src.dataset.validation.validate_locations import validate_locations
from src.dataset.validation.validate_search_texts import validate_search_texts
from src.utils.files import read_parquet, save_parquet


def main() -> None:
    merged_codes = read_parquet(path=MERGED_CODES_PATH)

    locations = build_locations_table(merged_codes=merged_codes)
    locations = validate_locations(locations=locations)

    aliases = build_aliases_table(merged_codes=merged_codes)
    aliases = validate_aliases(aliases=aliases, locations=locations)

    search_texts = build_search_texts_table(
        aliases=aliases,
        locations=locations,
    )

    search_texts = validate_search_texts(
        search_texts=search_texts,
        locations=locations,
    )

    locations_path = LOCATIONS_PATH
    aliases_path = ALIASES_PATH
    search_texts_path = SEARCH_TEXTS_PATH

    save_parquet(df=locations, path=locations_path)

    print(f"Saved locations to: {locations_path}")
    print(f"Shape: {locations.shape}")

    save_parquet(df=aliases, path=aliases_path)

    print(f"Saved aliases to: {aliases_path}")
    print(f"Shape: {aliases.shape}")

    save_parquet(df=search_texts, path=search_texts_path)

    print(f"Saved search texts to: {search_texts_path}")
    print(f"Shape: {search_texts.shape}")


if __name__ == "__main__":
    main()
