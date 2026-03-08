import requests
import zipfile
from pathlib import Path


URL = "https://service.unece.org/trade/locode/loc242csv.zip"

DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "unlocode.zip"

STREAM_CHUNK_SIZE = 8192


def _download_stream(response: requests.Response, destination: Path = ZIP_PATH, chunk_size: int = STREAM_CHUNK_SIZE):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)


def _download_zip(url: str = URL, destination: Path = ZIP_PATH):
    print("Downloading UN/LOCODE dataset from", url)

    destination.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        _download_stream(response, destination, chunk_size=STREAM_CHUNK_SIZE)


def _unzip(src: Path = ZIP_PATH, dest: Path = DATA_DIR):
    print("Extracting dataset from", src)
    
    with zipfile.ZipFile(src, "r") as z:
        z.extractall(dest)


def main():
    _download_zip()
    _unzip()

    print("Dataset ready in", DATA_DIR)


if __name__ == "__main__":
    main()