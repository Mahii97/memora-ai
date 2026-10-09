import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np

from app.embeddings.trainer import EmbeddingBuilder
from app.retrieval.bm25 import BM25Index
from app.timeline.timeline_index import TimelineIndexer
from app.vector.vector_index import VectorIndex
from app.tokenizer.dataset_builder import DatasetBuilder


BACKEND_ROOT = Path(__file__).resolve().parents[2]


class IndexPipeline:
    DATA_DIR = BACKEND_ROOT / "data"
    ARTIFACTS = (
        "vocabulary.json",
        "tokenized_chunks.json",
        "embeddings.npy",
        "vector_index.pkl",
        "bm25_index.json",
        "timeline_index.json",
    )

    @classmethod
    def rebuild(cls, chunks, data_dir=None):
        data_dir = Path(data_dir or cls.DATA_DIR)
        data_dir.mkdir(parents=True, exist_ok=True)
        stage = Path(tempfile.mkdtemp(prefix=".index-build-", dir=data_dir))

        try:
            dataset_builder = DatasetBuilder()
            dataset = dataset_builder.build(chunks)
            dataset_builder.save_vocabulary(stage / "vocabulary.json")
            dataset_builder.save_dataset(dataset, stage / "tokenized_chunks.json")

            embeddings_path = stage / "embeddings.npy"
            embeddings = EmbeddingBuilder().build(
                dataset_path=stage / "tokenized_chunks.json",
                embeddings_path=embeddings_path,
            )
            if len(embeddings) != len(dataset):
                raise RuntimeError("Embedding count does not match canonical dataset")

            vector_index = VectorIndex()
            vector_index.build(embeddings, dataset)
            vector_index.save(stage / "vector_index.pkl")

            bm25_index = BM25Index()
            bm25_index.build(dataset)
            bm25_index.save(stage / "bm25_index.json")

            timeline = TimelineIndexer().build(dataset)
            TimelineIndexer().save(timeline, stage / "timeline_index.json")

            cls._validate_bundle(dataset, embeddings, vector_index, bm25_index, timeline)
            cls._promote(stage, data_dir)
            return {
                "chunk_count": len(dataset),
                "embedding_dimension": int(np.asarray(embeddings).shape[1]),
            }
        finally:
            if stage.exists():
                shutil.rmtree(stage, ignore_errors=True)

    @staticmethod
    def _validate_bundle(dataset, embeddings, vector_index, bm25_index, timeline):
        dataset_ids = [str(chunk["chunk_id"]) for chunk in dataset]
        if len(dataset_ids) != len(set(dataset_ids)):
            raise RuntimeError("Canonical dataset contains duplicate chunk IDs")
        expected = set(dataset_ids)
        vector_ids = {str(chunk["chunk_id"]) for chunk in vector_index.metadata}
        bm25_ids = {str(chunk["chunk_id"]) for chunk in bm25_index.documents}
        timeline_ids = {str(chunk["chunk_id"]) for chunk in timeline}
        if not (expected == vector_ids == bm25_ids == timeline_ids):
            raise RuntimeError("Staged search indexes do not contain identical chunk IDs")
        if np.asarray(embeddings).shape[0] != len(expected):
            raise RuntimeError("Staged embedding count does not match canonical chunk IDs")

    @classmethod
    def _promote(cls, stage: Path, data_dir: Path):
        backup = stage / "backup"
        backup.mkdir()
        backed_up = []
        installed = []

        try:
            for artifact in cls.ARTIFACTS:
                current = data_dir / artifact
                if current.exists():
                    os.replace(current, backup / artifact)
                    backed_up.append(artifact)

            for artifact in cls.ARTIFACTS:
                os.replace(stage / artifact, data_dir / artifact)
                installed.append(artifact)
        except Exception:
            for artifact in reversed(installed):
                current = data_dir / artifact
                if current.exists():
                    current.unlink()
            for artifact in reversed(backed_up):
                previous = backup / artifact
                if previous.exists():
                    os.replace(previous, data_dir / artifact)
            raise
