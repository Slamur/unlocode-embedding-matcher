from pathlib import Path


def ensure_dir_exists(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir_exists(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
