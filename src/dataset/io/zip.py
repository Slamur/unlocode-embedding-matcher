import zipfile
from pathlib import Path

from src.utils.paths import ensure_dir_exists


def unzip(
    source: Path,
    destination_dir: Path | None = None,
) -> None:
    if not source.exists():
        print(f"Can't unzip from {source}: file not found")
        return

    if destination_dir is None:
        destination_dir = source.parent

    ensure_dir_exists(path=destination_dir)

    print(f"Extracting dataset from {source} to {destination_dir}")

    with zipfile.ZipFile(source, "r") as z:
        z.extractall(destination_dir)
