# "Address to UN/LOCODE" matcher

Exercise to get familiar with embedding search.

## Initialization

### Python + venv

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download UN/LOCODE raw data

```sh
python  -m scripts.download_unlocode
```

### Prepare UN/LOCODE parquet files

```sh
python  -m scripts.ingest_unlocode
python -m scripts.prepare_unlocode
```
