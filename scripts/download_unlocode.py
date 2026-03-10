
import zipfile
from pathlib import Path

from src.config.paths import RAW_DIR
from src.dataset.download import download_zip


_URL = "https://service.unece.org/trade/locode/loc242csv.zip"

_ZIP_FILENAME = "unlocode.zip"




def _unzip(source: Path, dest: Path = RAW_DIR):
    print(f"Extracting dataset from {source} to {dest}")

    with zipfile.ZipFile(source, "r") as z:
        z.extractall(dest)


def main():
    destination_dir = RAW_DIR

    zip_filename = download_zip(_URL, destination_dir, _ZIP_FILENAME)
    _unzip(source=zip_filename, dest=destination_dir)

    print("Dataset ready in", destination_dir)


if __name__ == "__main__":
    main()
