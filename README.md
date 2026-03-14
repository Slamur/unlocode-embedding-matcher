# "Address to UN/LOCODE" matcher

Exercise to get familiar with embedding search.

## Initialization

### Python + venv (development)

```sh
make install-dev
```

### Download and prepare UN/LOCODE dataset

```sh
make build-dataset
```

### Embedding / index dependencies

```sh
make install-embed
```

### Generate embeddings

```sh
make generate-embeddings
```

### Build FAISS index

```sh
make build-index
```

## Usage

### CLI Search

```sh
make search q="..."
```
