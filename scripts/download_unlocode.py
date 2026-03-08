import requests
import zipfile
from pathlib import Path


URL = "https://service.unece.org/trade/locode/loc242csv.zip"

DATA_DIR = Path("data")
ZIP_PATH = DATA_DIR / "unlocode.zip"


def download():
    DATA_DIR.mkdir(exist_ok=True)

    r = requests.get(URL)
    r.raise_for_status()

    with open(ZIP_PATH, "wb") as f:
        f.write(r.content)


def unzip():
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        z.extractall(DATA_DIR)


if __name__ == "__main__":
    download()
    unzip()

    print("Dataset ready in", DATA_DIR)