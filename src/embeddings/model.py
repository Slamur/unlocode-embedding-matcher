from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmbedderConfig:
    model_name: str
    device: str | None = None


class TextEmbedder:
    def __init__(self, config: EmbedderConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required to use TextEmbedder. "
                "Install embedding dependencies first."
            ) from exc

        self._config = config
        self._model = SentenceTransformer(
            model_name_or_path=config.model_name,
            device=config.device,
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        embeddings = self._model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=True,
        )

        return embeddings.astype(np.float32, copy=False)

    @property
    def embedding_dim(self) -> int:
        return self._model.get_sentence_embedding_dimension()

    @property
    def device(self) -> str:
        if not hasattr(self._model, "device"):
            return "unknown"

        device = self._model.device
        return str(device)
