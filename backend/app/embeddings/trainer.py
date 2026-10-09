import json
import numpy as np
from pathlib import Path

from app.embeddings.embedding_service import EmbeddingService


class EmbeddingBuilder:

    def build(self, dataset_path=None, embeddings_path=None):
        backend = Path(__file__).resolve().parents[2]
        dataset_path = Path(dataset_path or backend / "data" / "tokenized_chunks.json")
        embeddings_path = Path(embeddings_path or backend / "data" / "embeddings.npy")

        with dataset_path.open(encoding="utf-8") as f:
            dataset = json.load(f)

        service = EmbeddingService()
        embeddings = service.embed_documents([chunk["text"] for chunk in dataset])

        if len(embeddings) != len(dataset):
            raise RuntimeError("Embedding count does not match indexed chunk count")

        np.save(embeddings_path, embeddings)

        print(
            f"Created {len(embeddings)} embeddings with {service.model_name}"
        )
        print(f"Shape: {embeddings.shape}")
        print(f"Dimension: {service.dimension}")
        return embeddings