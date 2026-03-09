from pathlib import Path


ROOT_FILES = ["requirements.txt", ".git"]


def _find_project_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if any((path / root_file).exists() for root_file in ROOT_FILES):
            return path

    raise RuntimeError("Could not determine project root")


def _get_existing_dir_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


PROJECT_ROOT = _find_project_root(Path(__file__).resolve())

DATA_DIR = _get_existing_dir_path(PROJECT_ROOT / "data")
RAW_DIR = _get_existing_dir_path(DATA_DIR / "raw")
PROCESSED_DIR = _get_existing_dir_path(DATA_DIR / "processed")
