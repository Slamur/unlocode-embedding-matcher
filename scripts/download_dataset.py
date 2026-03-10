from src.config.paths import RAW_DIR
from src.dataset.io.download import download
from src.dataset.io.zip import unzip

_URL = "https://service.unece.org/trade/locode/loc242csv.zip"

_ZIP_FILENAME = "unlocode.zip"


def main():
    destination_dir = RAW_DIR

    zip_path = download(url=_URL, destination_dir=RAW_DIR, destination_filename=_ZIP_FILENAME)
    unzip(source=zip_path, dest=RAW_DIR)

    print("Dataset ready in", destination_dir)


if __name__ == "__main__":
    main()
