import numpy as np

from app.vector.cosine import CosineSimilarity

class SearchEngine:

    def __init__(self, index):
        self.index = index

    def search(self, query_vector, k=5):
        query_vector = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if self.index.embeddings is None:
            return []
        if query_vector.shape[0] != self.index.embeddings.shape[1]:
            raise ValueError(
                "query embedding dimension does not match vector index dimension"
            )

        results = []

        for i in range(len(self.index.embeddings)):

            score = CosineSimilarity.score(
                query_vector,
                self.index.embeddings[i]
            )

            results.append({
                "score": score,
                "chunk": self.index.metadata[i]
            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:k]