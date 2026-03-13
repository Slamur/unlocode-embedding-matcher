from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.embeddings.artifacts import EmbeddingArtifacts, read_and_validate_artifacts


def test_read_and_validate_artifacts_returns_none_when_no_files_exist(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.parquet"
    embeddings_path = tmp_path / "embeddings.npy"
    manifest_path = tmp_path / "manifest.json"

    artifacts = read_and_validate_artifacts(
        metadata_path=metadata_path,
        embeddings_path=embeddings_path,
        manifest_path=manifest_path,
    )

    assert artifacts is None


def test_read_and_validate_artifacts_raises_on_incomplete_artifacts(tmp_path) -> None:
    metadata_path = tmp_path / "metadata.parquet"
    embeddings_path = tmp_path / "embeddings.npy"
    manifest_path = tmp_path / "manifest.json"

    metadata_path.touch()

    with pytest.raises(
        RuntimeError,
        match="Found incomplete embedding artifacts",
    ):
        read_and_validate_artifacts(
            metadata_path=metadata_path,
            embeddings_path=embeddings_path,
            manifest_path=manifest_path,
        )


def test_read_and_validate_artifacts_raises_on_embeddings_metadata_row_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    metadata_path = tmp_path / "metadata.parquet"
    embeddings_path = tmp_path / "embeddings.npy"
    manifest_path = tmp_path / "manifest.json"

    metadata_path.touch()
    embeddings_path.touch()
    manifest_path.touch()

    monkeypatch.setattr(
        "src.embeddings.artifacts.read_parquet",
        lambda path: pd.DataFrame({"locode": ["AA001", "AA002"]}),
    )
    monkeypatch.setattr(
        "src.embeddings.artifacts.read_embeddings",
        lambda path: np.array([[1.0, 0.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.embeddings.artifacts.read_manifest",
        lambda path: SimpleNamespace(row_count=1, embedding_dim=2),
    )

    with pytest.raises(ValueError, match="Artifacts mismatch"):
        read_and_validate_artifacts(
            metadata_path=metadata_path,
            embeddings_path=embeddings_path,
            manifest_path=manifest_path,
        )


def test_read_and_validate_artifacts_raises_on_manifest_row_count_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    metadata_path = tmp_path / "metadata.parquet"
    embeddings_path = tmp_path / "embeddings.npy"
    manifest_path = tmp_path / "manifest.json"

    metadata_path.touch()
    embeddings_path.touch()
    manifest_path.touch()

    monkeypatch.setattr(
        "src.embeddings.artifacts.read_parquet",
        lambda path: pd.DataFrame({"locode": ["AA001", "AA002"]}),
    )
    monkeypatch.setattr(
        "src.embeddings.artifacts.read_embeddings",
        lambda path: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.embeddings.artifacts.read_manifest",
        lambda path: SimpleNamespace(row_count=3, embedding_dim=2),
    )

    with pytest.raises(ValueError, match="Manifest row_count does not match metadata"):
        read_and_validate_artifacts(
            metadata_path=metadata_path,
            embeddings_path=embeddings_path,
            manifest_path=manifest_path,
        )


def test_read_and_validate_artifacts_raises_on_manifest_embedding_dim_mismatch(
    tmp_path,
    monkeypatch,
) -> None:
    metadata_path = tmp_path / "metadata.parquet"
    embeddings_path = tmp_path / "embeddings.npy"
    manifest_path = tmp_path / "manifest.json"

    metadata_path.touch()
    embeddings_path.touch()
    manifest_path.touch()

    monkeypatch.setattr(
        "src.embeddings.artifacts.read_parquet",
        lambda path: pd.DataFrame({"locode": ["AA001", "AA002"]}),
    )
    monkeypatch.setattr(
        "src.embeddings.artifacts.read_embeddings",
        lambda path: np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
    )
    monkeypatch.setattr(
        "src.embeddings.artifacts.read_manifest",
        lambda path: SimpleNamespace(row_count=2, embedding_dim=3),
    )

    with pytest.raises(ValueError, match="Manifest embedding_dim does not match embeddings"):
        read_and_validate_artifacts(
            metadata_path=metadata_path,
            embeddings_path=embeddings_path,
            manifest_path=manifest_path,
        )


def test_read_and_validate_artifacts_returns_artifacts_when_valid(
    tmp_path,
    monkeypatch,
) -> None:
    metadata_path = tmp_path / "metadata.parquet"
    embeddings_path = tmp_path / "embeddings.npy"
    manifest_path = tmp_path / "manifest.json"

    metadata_path.touch()
    embeddings_path.touch()
    manifest_path.touch()

    metadata = pd.DataFrame({"locode": ["AA001", "AA002"]})
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    manifest = SimpleNamespace(row_count=2, embedding_dim=2)

    monkeypatch.setattr("src.embeddings.artifacts.read_parquet", lambda path: metadata)
    monkeypatch.setattr("src.embeddings.artifacts.read_embeddings", lambda path: embeddings)
    monkeypatch.setattr("src.embeddings.artifacts.read_manifest", lambda path: manifest)

    artifacts = read_and_validate_artifacts(
        metadata_path=metadata_path,
        embeddings_path=embeddings_path,
        manifest_path=manifest_path,
    )

    assert isinstance(artifacts, EmbeddingArtifacts)
    assert artifacts.metadata is metadata
    assert artifacts.embeddings is embeddings
    assert artifacts.manifest is manifest
