import zipfile
from pathlib import Path

def unzip(source: Path, dest: Path):
    print(f"Extracting dataset from {source} to {dest}")

    with zipfile.ZipFile(source, "r") as z:
        z.extractall(dest)
