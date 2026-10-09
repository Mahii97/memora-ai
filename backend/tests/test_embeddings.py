import unittest

import numpy as np

from app.embeddings.embedding_service import EmbeddingService
from app.vector.search import SearchEngine
from app.vector.vector_index import VectorIndex


class EmbeddingServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.service = EmbeddingService(device="cpu")
        cls.model = cls.service.model

    def test_pretrained_model_loads_once_on_cpu(self):
        self.assertIs(self.model, self.service.model)
        self.assertEqual(str(self.model.device), "cpu")
        self.assertEqual(
            self.service.model_name,
            "sentence-transformers/all-MiniLM-L6-v2",
        )

    def test_embed_text_returns_real_embedding(self):
        vector = self.service.embed_text("A student studies database systems.")

        self.assertEqual(vector.shape, (self.service.dimension,))
        self.assertEqual(vector.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=5)

    def test_embed_query_returns_dimension_consistent_vector(self):
        vector = self.service.embed_query("Explain database normalization.")

        self.assertEqual(vector.shape, (self.service.dimension,))
        self.assertTrue(np.isfinite(vector).all())

    def test_embed_documents_supports_batches(self):
        vectors = self.service.embed_documents(
            [
                "A dog runs through a grassy field.",
                "The stock market closed higher today.",
            ]
        )

        self.assertEqual(vectors.shape, (2, self.service.dimension))
        np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)

    def test_all_embedding_methods_have_consistent_dimensions(self):
        text_vector = self.service.embed_text("A short test sentence.")
        query_vector = self.service.embed_query("A related search question.")
        document_vectors = self.service.embed_documents(["A document passage."])

        self.assertEqual(text_vector.shape[0], self.service.dimension)
        self.assertEqual(query_vector.shape[0], self.service.dimension)
        self.assertEqual(document_vectors.shape[1], self.service.dimension)

    def test_numpy_vector_search_ranks_semantically_related_text_first(self):
        documents = [
            {"chunk_id": "dog", "text": "A puppy runs across a grassy field."},
            {"chunk_id": "finance", "text": "The stock market was volatile today."},
        ]
        document_vectors = self.service.embed_documents(
            [document["text"] for document in documents]
        )
        index = VectorIndex()
        index.build(document_vectors, documents)

        results = SearchEngine(index).search(
            self.service.embed_query("A dog is running in a field."), k=2
        )

        self.assertEqual(results[0]["chunk"]["chunk_id"], "dog")
        self.assertGreater(results[0]["score"], results[1]["score"])


if __name__ == "__main__":
    unittest.main()
