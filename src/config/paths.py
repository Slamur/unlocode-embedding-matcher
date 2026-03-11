from pathlib import Path

ROOT_FILES = ["requirements.txt", ".git"]


def _find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if any((path / root_file).exists() for root_file in ROOT_FILES):
            return path

    raise RuntimeError("Could not determine project root")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())

# Dataset paths

DATA_DIR = PROJECT_ROOT / "data"

RAW_DIR = DATA_DIR / "raw"

INTERIM_DIR = DATA_DIR / "interim"
MERGED_CODES_PATH = INTERIM_DIR / "merged_codes.parquet"

PROCESSED_DIR = DATA_DIR / "processed"

LOCATIONS_PATH = PROCESSED_DIR / "unlocode_locations.parquet"
ALIASES_PATH = PROCESSED_DIR / "unlocode_aliases.parquet"
SEARCH_TEXTS_PATH = PROCESSED_DIR / "unlocode_search_texts.parquet"
