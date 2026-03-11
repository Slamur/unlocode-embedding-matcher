from pathlib import Path

import pandas as pd


def ensure_dir_exists(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir_exists(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    ensure_parent_dir_exists(path=path)
    df.to_parquet(path, index=False)


def read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found at: {path}")
    return pd.read_parquet(path)
