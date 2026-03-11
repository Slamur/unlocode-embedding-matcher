from pathlib import Path

import requests

from src.utils.files import ensure_parent_dir_exists

_STREAM_CHUNK_SIZE = 8192


def _download_stream(
    response: requests.Response,
    destination: Path,
    chunk_size: int = _STREAM_CHUNK_SIZE,
) -> None:
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            f.write(chunk)


def download(
    url: str,
    destination: Path,
) -> None:
    print(f"Downloading from {url} to {destination}")

    if destination.exists():
        print(f"Skipping download: file {destination} already exists.")
        return

    ensure_parent_dir_exists(path=destination)

    with requests.get(url, stream=True) as response:
        response.raise_for_status()

        _download_stream(response=response, destination=destination)
