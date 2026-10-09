import unittest

import numpy as np

from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.hybrid_engine import HybridEngine
from app.retrieval.reranker import RerankerService
from app.rag.query_understanding import QueryUnderstanding
from app.vector.search import SearchEngine
from app.vector.vector_index import VectorIndex


class RerankerServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reranker = RerankerService(device="cpu")
        cls.model = cls.reranker.model

    def test_initialization_loads_and_reuses_local_model(self):
        self.assertIs(self.model, self.reranker.model)
        self.assertEqual(str(self.model.device), "cpu")
        self.assertEqual(
            self.reranker.model_name,
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        )

    def test_ranks_multiple_candidates_by_model_score(self):
        candidates = [
            {"chunk_id": "finance", "text": "The stock market closed higher today."},
            {"chunk_id": "dog", "text": "A puppy runs across a grassy field."},
            {"chunk_id": "cooking", "text": "Bake the bread until the crust turns golden."},
        ]

        ranked = self.reranker.rerank(
            "Where is a dog running in a field?", candidates, top_k=3
        )

        self.assertEqual(len(ranked), 3)
        self.assertEqual(ranked[0]["chunk_id"], "dog")
        self.assertEqual(
            [item["reranker_score"] for item in ranked],
            sorted((item["reranker_score"] for item in ranked), reverse=True),
        )

    def test_top_k_limits_results_and_assigns_final_rank(self):
        candidates = [
            {"text": "A dog runs across a field."},
            {"text": "A market report lists stock prices."},
            {"text": "A lecture explains database indexes."},
        ]

        ranked = self.reranker.rerank("Explain database indexes.", candidates, top_k=2)

        self.assertEqual(len(ranked), 2)
        self.assertEqual([item["final_rank"] for item in ranked], [1, 2])

    def test_empty_candidates_return_empty_list(self):
        self.assertEqual(self.reranker.rerank("any query", [], top_k=5), [])

    def test_hybrid_candidates_drop_repeated_text_in_same_source_location(self):
        repeated = [
            {"chunk_id": "a", "source_name": "Notes.pdf", "location": "Page 1", "text": "Same text."},
            {"chunk_id": "b", "source_name": "Notes.pdf", "location": "Page 1", "text": "Same   text."},
            {"chunk_id": "c", "source_name": "Other.pdf", "location": "Page 1", "text": "Same text."},
        ]
        unique = HybridEngine._deduplicate_candidates(repeated)
        self.assertEqual([item["chunk_id"] for item in unique], ["a", "c"])

    def test_inference_is_repeatable(self):
        candidates = [
            {"text": "A dog runs across a grassy field."},
            {"text": "The stock market closed higher today."},
        ]

        first = self.reranker.rerank("A dog running outdoors", candidates, top_k=2)
        second = self.reranker.rerank("A dog running outdoors", candidates, top_k=2)

        self.assertEqual(
            [item["final_rank"] for item in first],
            [item["final_rank"] for item in second],
        )
        np.testing.assert_allclose(
            [item["reranker_score"] for item in first],
            [item["reranker_score"] for item in second],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_hybrid_pipeline_reranks_top_twenty_to_top_five(self):
        vector_index = VectorIndex()
        vector_index.load("data/vector_index.pkl")
        query = "computer science information technology exam at IIT Roorkee"
        query_vector = EmbeddingService().embed_query(query)
        engine = HybridEngine(SearchEngine(vector_index), reranker=self.reranker)

        trace = engine.search_with_trace(query, query_vector)

        self.assertLessEqual(trace["candidate_count"], 20)
        self.assertEqual(len(trace["hybrid_order"]), trace["candidate_count"])
        self.assertEqual(len(trace["results"]), 5)
        self.assertEqual(
            [item["final_rank"] for item in trace["results"]],
            [1, 2, 3, 4, 5],
        )
        for item in trace["results"]:
            self.assertIn("vector_score", item)
            self.assertIn("bm25_score", item)
            self.assertIn("hybrid_score", item)
            self.assertIn("reranker_score", item)
            self.assertIn("final_rank", item)
            self.assertEqual(
                set(item["explanation"]),
                {
                    "vector_score",
                    "bm25_score",
                    "hybrid_score",
                    "reranker_score",
                    "final_rank",
                },
            )

    def test_temporal_summary_uses_only_timeline_records_in_requested_range(self):
        class Timeline:
            def by_range(self, start, end):
                self.requested = (start, end)
                return [
                    {"chunk_id": "in-range", "text": "Project notes", "source_name": "Notes.txt",
                     "source_type": "txt", "location": "Paragraph 1", "created_at": "2026-09-24"},
                ]

        class Reranker:
            def rerank(self, query, documents, top_k):
                return [dict(item, reranker_score=1.0, explanation={}) for item in documents]

        engine = HybridEngine.__new__(HybridEngine)
        engine.timeline = Timeline()
        engine.candidate_k = 20
        engine.top_k = 5
        engine.reranker = Reranker()
        engine.query_understanding = QueryUnderstanding()
        engine.retrieve_candidates = lambda _query, _vector, _plan=None: {
            "vector_candidate_count": 0, "bm25_candidate_count": 0,
            "candidate_count": 0, "candidates": [], "hybrid_order": [],
            "hybrid_latency_ms": 0.0,
        }

        trace = engine.search_with_trace(
            "What did I work on last week?", [],
            query_plan={"intent": "temporal_summary", "time_range": {
                "start": "2026-09-21", "end": "2026-09-27"
            }},
        )

        self.assertEqual(engine.timeline.requested, ("2026-09-21", "2026-09-27"))
        self.assertEqual([item["chunk_id"] for item in trace["results"]], ["in-range"])
        self.assertEqual(trace["temporal_candidate_count"], 1)


if __name__ == "__main__":
    unittest.main()
