import requests
import zipfile
from pathlib import Path


URL = "https://service.unece.org/trade/locode/loc242csv.zip"

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"

ZIP_FILENAME = "unlocode.zip"

STREAM_CHUNK_SIZE = 8192


def _download_stream(response: requests.Response, destination: Path, chunk_size: int = STREAM_CHUNK_SIZE):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)


def _download_zip(url: str = URL, destination_dir: Path = RAW_DIR) -> Path:
    print(f"Downloading UN/LOCODE dataset from {url} to {destination_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / ZIP_FILENAME

    if destination.exists():
        print(f"File {destination} already exists. Skipping download.")
        return destination

    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        _download_stream(response, destination, chunk_size=STREAM_CHUNK_SIZE)

    return destination


def _unzip(source: Path, dest: Path = RAW_DIR):
    print(f"Extracting dataset from {source} to {dest}")

    with zipfile.ZipFile(source, "r") as z:
        z.extractall(dest)


def main():
    destination_dir = RAW_DIR

    zip_filename = _download_zip(destination_dir=destination_dir)
    _unzip(source=zip_filename, dest=destination_dir)

    print("Dataset ready in", destination_dir)


if __name__ == "__main__":
    main()