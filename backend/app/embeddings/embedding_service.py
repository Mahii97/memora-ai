import os
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_ROOT / ".env")

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=2)
def _load_model(model_name: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(model_name, device=device)


class EmbeddingService:
    """Local sentence embeddings backed by a cached SentenceTransformer model."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", DEFAULT_MODEL)
        self.device = device or os.getenv("EMBEDDING_DEVICE", "cpu")

    @property
    def model(self) -> SentenceTransformer:
        return _load_model(self.model_name, self.device)

    @property
    def dimension(self) -> int:
        dimension = self.model.get_embedding_dimension()
        if dimension is None:
            raise RuntimeError(f"Model {self.model_name!r} has no sentence embedding dimension")
        return int(dimension)

    def embed_text(self, text: str) -> np.ndarray:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return np.asarray(
            self.model.encode(
                text,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def embed_query(self, query: str) -> np.ndarray:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        return np.asarray(
            self.model.encode_query(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("all documents must be strings")
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return np.asarray(
            self.model.encode_document(
                list(texts),
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
