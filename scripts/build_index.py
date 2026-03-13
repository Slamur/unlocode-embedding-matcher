from src.config.paths import (
    FAISS_INDEX_PATH,
    SEARCH_TEXTS_EMBEDDINGS_MANIFEST_PATH,
    SEARCH_TEXTS_EMBEDDINGS_PATH,
    SEARCH_TEXTS_METADATA_PATH,
)
from src.embeddings.artifacts import read_and_validate_artifacts
from src.index.build import build_index
from src.index.io import save_index


def main() -> None:
    artifacts = read_and_validate_artifacts(
        metadata_path=SEARCH_TEXTS_METADATA_PATH,
        embeddings_path=SEARCH_TEXTS_EMBEDDINGS_PATH,
        manifest_path=SEARCH_TEXTS_EMBEDDINGS_MANIFEST_PATH,
    )

    if artifacts is None:
        print("No valid artifacts found. Please run the embedding build script first.")
        return

    index, build_result = build_index(embeddings=artifacts.embeddings)
    save_index(index=index, path=FAISS_INDEX_PATH)

    print(
        f"Built FAISS index: " f"vectors={build_result.vector_count}, dim={build_result.dimension}"
    )


if __name__ == "__main__":
    main()
