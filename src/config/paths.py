from pathlib import Path

ROOT_FILES = ["requirements.txt", ".git"]


def _find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if any((path / root_file).exists() for root_file in ROOT_FILES):
            return path

    raise RuntimeError("Could not determine project root")


def _ensure_dir_exists(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())

DATA_DIR = _ensure_dir_exists(PROJECT_ROOT / "data")
RAW_DIR = _ensure_dir_exists(DATA_DIR / "raw")
INTERIM_DIR = _ensure_dir_exists(DATA_DIR / "interim")
PROCESSED_DIR = _ensure_dir_exists(DATA_DIR / "processed")
