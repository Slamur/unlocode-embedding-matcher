from pathlib import Path

import pandas as pd


def ensure_dir_exists(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir_exists(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def require_file_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found at: {path}")

    if not path.is_file():
        raise FileNotFoundError(f"Expected a file but found a directory at: {path}")

    return path


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir_exists(path=path)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    require_file_exists(path=path)

    return pd.read_parquet(path)
