from src.config.paths import RAW_DIR
from src.dataset.io.download import download
from src.dataset.io.zip import unzip

_URL = "https://service.unece.org/trade/locode/loc242csv.zip"

_ZIP_FILENAME = "unlocode.zip"


def main():
    destination_dir = RAW_DIR
    zip_path = destination_dir / _ZIP_FILENAME

    download(url=_URL, destination=zip_path)
    unzip(source=zip_path)

    print("Dataset ready in", destination_dir)


if __name__ == "__main__":
    main()
