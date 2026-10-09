import os
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder


BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_ROOT / ".env")

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=2)
def _load_model(model_name: str, device: str) -> CrossEncoder:
    return CrossEncoder(model_name, device=device)


class RerankerService:
    """Locally score query/document pairs with a cached CrossEncoder."""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or os.getenv(
            "RERANKER_MODEL", DEFAULT_RERANKER_MODEL
        )
        self.device = device or os.getenv("RERANKER_DEVICE", "cpu")

    @property
    def model(self) -> CrossEncoder:
        return _load_model(self.model_name, self.device)

    @staticmethod
    def _text(document: str | Mapping) -> str:
        if isinstance(document, str):
            return document
        if isinstance(document, Mapping) and isinstance(document.get("text"), str):
            return document["text"]
        raise ValueError("each document must be a string or contain string 'text'")

    def rerank(self, query: str, documents: Sequence, top_k: int = 5) -> list[dict]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if top_k <= 0 or not documents:
            return []

        pairs = [[query, self._text(document)] for document in documents]
        scores = np.asarray(
            self.model.predict(
                pairs,
                batch_size=16,
                show_progress_bar=False,
                convert_to_numpy=True,
            ),
            dtype=np.float32,
        ).reshape(-1)

        ranked = []
        for document, score in zip(documents, scores):
            result = dict(document) if isinstance(document, Mapping) else {"text": document}
            result["reranker_score"] = float(score)
            result["explanation"] = {
                "vector_score": result.get("vector_score", 0.0),
                "bm25_score": result.get("bm25_score", 0.0),
                "hybrid_score": result.get("hybrid_score", 0.0),
                "reranker_score": float(score),
            }
            ranked.append(result)

        ranked.sort(key=lambda item: item["reranker_score"], reverse=True)
        for final_rank, result in enumerate(ranked[:top_k], start=1):
            result["final_rank"] = final_rank
            result["explanation"]["final_rank"] = final_rank
        return ranked[:top_k]
