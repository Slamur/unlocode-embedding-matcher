import requests
from pathlib import Path


_STREAM_CHUNK_SIZE = 8192


def _download_stream(response: requests.Response, destination: Path, chunk_size: int = _STREAM_CHUNK_SIZE):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)


def download(url: str, destination_dir: Path, destination_filename: str) -> Path:
    print(f"Downloading {destination_filename} from {url} to {destination_dir}")

    destination_dir.mkdir(parents=True, exist_ok=True)

    destination = destination_dir / destination_filename

    if destination.exists():
        print(f"File {destination} already exists. Skipping download.")
        return destination

    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        _download_stream(response, destination, chunk_size=_STREAM_CHUNK_SIZE)

    return destination
