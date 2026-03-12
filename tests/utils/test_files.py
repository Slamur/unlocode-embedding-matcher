from pathlib import Path

import pandas as pd
import pytest

from src.utils.files import (
    ensure_dir_exists,
    ensure_parent_dir_exists,
    read_parquet,
    require_file_exists,
    save_parquet,
)


def test_ensure_dir_exists_creates_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir"

    result = ensure_dir_exists(path)

    assert path.exists()
    assert path.is_dir()
    assert result == path


def test_ensure_parent_dir_exists_creates_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "file.parquet"

    result = ensure_parent_dir_exists(path)

    assert path.parent.exists()
    assert path.parent.is_dir()
    assert result == path


def test_require_file_exists_returns_path_for_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "file.txt"
    path.write_text("hello", encoding="utf-8")

    result = require_file_exists(path)

    assert result == path


def test_require_file_exists_raises_when_file_missing(tmp_path: Path) -> None:
    path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError, match="Required file not found"):
        require_file_exists(path)


def test_require_file_exists_raises_when_path_is_directory(tmp_path: Path) -> None:
    path = tmp_path / "directory"
    path.mkdir()

    with pytest.raises(FileNotFoundError, match="Expected file, found non-file path"):
        require_file_exists(path)


def test_save_and_read_parquet_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "data.parquet"
    expected = pd.DataFrame(
        {
            "locode": ["KG FRU", "JP TYO"],
            "search_text": ["bishkek", "tokyo"],
        }
    )

    save_parquet(expected, path)
    actual = read_parquet(path)

    pd.testing.assert_frame_equal(actual, expected)


def test_read_parquet_raises_when_file_missing(tmp_path: Path) -> None:
    path = tmp_path / "missing.parquet"

    with pytest.raises(FileNotFoundError, match="Required file not found"):
        read_parquet(path)
