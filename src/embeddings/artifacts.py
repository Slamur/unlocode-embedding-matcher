from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.embeddings.generate import EmbeddingBuildInfo
from src.embeddings.io import read_embeddings, read_manifest
from src.utils.files import read_parquet


@dataclass(frozen=True)
class EmbeddingArtifacts:
    metadata: pd.DataFrame
    embeddings: np.ndarray
    manifest: EmbeddingBuildInfo


def _artifacts_exist(
    metadata_path: Path,
    embeddings_path: Path,
    manifest_path: Path,
) -> bool:
    artifact_exists = [
        metadata_path.exists(),
        embeddings_path.exists(),
        manifest_path.exists(),
    ]

    if not all(artifact_exists):
        if any(artifact_exists):
            raise RuntimeError(
                "Found incomplete embedding artifacts. "
                "Remove them manually and rerun the build script."
            )

        return False

    return True


def _validate_artifacts(
    artifacts: EmbeddingArtifacts,
) -> None:
    metadata, embeddings, manifest = artifacts.metadata, artifacts.embeddings, artifacts.manifest

    if embeddings.shape[0] != len(metadata):
        raise ValueError(
            f"Artifacts mismatch: embeddings rows = {embeddings.shape[0]}, "
            f"metadata rows = {len(metadata)}"
        )

    if manifest.row_count != len(metadata):
        raise ValueError("Manifest row_count does not match metadata")
    if manifest.embedding_dim != embeddings.shape[1]:
        raise ValueError("Manifest embedding_dim does not match embeddings")


def read_and_validate_artifacts(
    metadata_path: Path,
    embeddings_path: Path,
    manifest_path: Path,
) -> EmbeddingArtifacts | None:

    if not _artifacts_exist(
        metadata_path=metadata_path, embeddings_path=embeddings_path, manifest_path=manifest_path
    ):
        return None

    artifacts = EmbeddingArtifacts(
        metadata=read_parquet(path=metadata_path),
        embeddings=read_embeddings(path=embeddings_path),
        manifest=read_manifest(path=manifest_path),
    )

    _validate_artifacts(artifacts=artifacts)

    return artifacts
