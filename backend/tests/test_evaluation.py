import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from app.citations.citation_service import CitationService
from app.embeddings.embedding_service import EmbeddingService
from app.evaluation.latency import LatencyCollector
from app.evaluation.metrics import (
    hit_rate_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    query_metrics,
    recall_at_k,
    reciprocal_rank,
)
from app.evaluation.rag_evaluator import RAGEvaluator, citation_validity
from app.evaluation.results import save_run
from app.evaluation.retrieval_evaluator import METHODS, RetrievalEvaluator
from app.evaluation.schemas import EvaluationQuery, load_evaluation_dataset, resolve_relevant_chunk_ids
from app.models.rag_response import CitationResponse, RAGResponse
from compile_evaluation_labels import compile_labels
from app.retrieval.hybrid_engine import HybridEngine
from app.vector.search import SearchEngine
from app.vector.vector_index import VectorIndex


class RetrievalMetricTests(unittest.TestCase):
    def setUp(self):
        self.relevant = {"A", "C"}
        self.retrieved = ["A", "B", "C", "D"]

    def test_precision_at_k(self):
        self.assertEqual(precision_at_k(self.retrieved, self.relevant, 1), 1.0)
        self.assertAlmostEqual(precision_at_k(self.retrieved, self.relevant, 3), 2 / 3)
        self.assertEqual(precision_at_k(self.retrieved, self.relevant, 5), 2 / 5)
        self.assertEqual(precision_at_k(self.retrieved, self.relevant, 10), 2 / 10)

    def test_recall_at_k(self):
        self.assertEqual(recall_at_k(self.retrieved, self.relevant, 1), 0.5)
        self.assertEqual(recall_at_k(self.retrieved, self.relevant, 3), 1.0)
        self.assertEqual(recall_at_k(self.retrieved, self.relevant, 5), 1.0)
        self.assertEqual(recall_at_k(self.retrieved, self.relevant, 10), 1.0)

    def test_mrr(self):
        self.assertEqual(reciprocal_rank(self.retrieved, self.relevant), 1.0)
        self.assertAlmostEqual(
            mean_reciprocal_rank([["A"], ["B", "C"]], [self.relevant, self.relevant]),
            0.75,
        )

    def test_ndcg_binary_relevance(self):
        expected = 1.5 / (1 + 1 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(self.retrieved, self.relevant, 3), expected)
        self.assertAlmostEqual(ndcg_at_k(self.retrieved, self.relevant, 5), expected)
        self.assertAlmostEqual(ndcg_at_k(self.retrieved, self.relevant, 10), expected)

    def test_hit_rate_at_k(self):
        self.assertEqual(hit_rate_at_k(self.retrieved, self.relevant, 1), 1.0)
        self.assertEqual(hit_rate_at_k(["B", "C"], self.relevant, 1), 0.0)
        self.assertEqual(hit_rate_at_k(["B", "C"], self.relevant, 2), 1.0)

    def test_empty_sets_and_retrieval_return_zero(self):
        self.assertEqual(precision_at_k([], {"A"}, 5), 0.0)
        self.assertEqual(recall_at_k(["A"], set(), 5), 0.0)
        self.assertEqual(reciprocal_rank([], {"A"}), 0.0)
        self.assertEqual(ndcg_at_k([], {"A"}, 5), 0.0)
        self.assertEqual(hit_rate_at_k([], {"A"}, 5), 0.0)
        self.assertEqual(query_metrics([], set())["precision@1"], 0.0)

    def test_duplicate_retrieved_ids_count_only_once(self):
        self.assertAlmostEqual(precision_at_k(["A", "A", "C"], self.relevant, 3), 2 / 3)
        self.assertEqual(recall_at_k(["A", "A", "C"], self.relevant, 3), 1.0)
        self.assertEqual(reciprocal_rank(["B", "A", "A"], self.relevant), 0.5)


class LatencyTests(unittest.TestCase):
    def test_latency_collector_summarizes_real_samples(self):
        collector = LatencyCollector()
        collector.extend("search", [1.0, 2.0, 3.0, 4.0])

        summary = collector.summary()["search"]

        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["mean_ms"], 2.5)
        self.assertEqual(summary["median_ms"], 2.5)
        self.assertEqual(summary["min_ms"], 1.0)
        self.assertEqual(summary["max_ms"], 4.0)
        self.assertAlmostEqual(summary["p95_ms"], 3.85)
        collector.add("generation", None)
        self.assertNotIn("generation", collector.summary())


class DatasetTests(unittest.TestCase):
    def test_dataset_loader_accepts_unlabelled_latency_query(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queries.jsonl"
            path.write_text(
                '{"query_id":"q1","query":"Latency only"}\n',
                encoding="utf-8",
            )
            dataset = load_evaluation_dataset(path)

        self.assertEqual(len(dataset), 1)
        self.assertFalse(dataset[0].has_relevance_labels)

    def test_dataset_loader_reads_human_relevance_and_optional_answer(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queries.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "query_id": "q1",
                        "query": "Find source",
                        "relevant_chunk_ids": ["c1", "c1"],
                        "relevant_document": "notes.pdf",
                        "relevant_locations": ["Page 1"],
                        "reference_answer": "Relevant answer",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            dataset = load_evaluation_dataset(path)

        self.assertEqual(dataset[0].relevant_chunk_ids, ("c1",))
        self.assertEqual(dataset[0].reference_answer, "Relevant answer")
        self.assertTrue(dataset[0].has_relevance_labels)

    def test_resolve_document_and_location_judgments_to_chunk_ids(self):
        query = EvaluationQuery(
            "q1", "find page", relevant_document="notes.pdf", relevant_locations=("Page 1",)
        )
        corpus = [
            {"chunk_id": "c1", "source_name": "notes.pdf", "location": "Page 1"},
            {"chunk_id": "c2", "source_name": "notes.pdf", "location": "Page 2"},
            {"chunk_id": "c3", "source_name": "other.pdf", "location": "Page 1"},
        ]

        self.assertEqual(resolve_relevant_chunk_ids(query, corpus), {"c1"})

    def test_dataset_loader_rejects_duplicate_query_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "queries.jsonl"
            row = '{"query_id":"q1","query":"Question"}\n'
            path.write_text(row + row, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate query_id"):
                load_evaluation_dataset(path)


class CitationValidationTests(unittest.TestCase):
    def test_citation_validity_requires_a_supplied_matching_source(self):
        supplied = CitationResponse(
            citation_id=1,
            source_name="notes.pdf",
            source_type="pdf",
            location="Page 1",
            chunk_id="c1",
        )
        valid = CitationResponse(
            citation_id=1,
            source_name="notes.pdf",
            source_type="pdf",
            location="Page 1",
            chunk_id="c1",
        )
        invalid = CitationResponse(
            citation_id=2,
            source_name="other.pdf",
            source_type="pdf",
            location="Page 4",
            chunk_id="outside",
        )

        result = citation_validity([valid, invalid], [supplied])

        self.assertEqual(result["valid_count"], 1)
        self.assertEqual(result["citation_count"], 2)
        self.assertEqual(result["validity_rate"], 0.5)
        self.assertEqual(result["invalid_citation_ids"], [2])
        self.assertIsNone(citation_validity([], [supplied])["validity_rate"])


class RAGEvaluationTests(unittest.TestCase):
    def test_rag_metrics_check_generation_context_citations_and_abstention(self):
        source = CitationResponse(
            citation_id=1,
            source_name="notes.pdf",
            source_type="pdf",
            location="Page 1",
            chunk_id="c1",
            score=4.2,
        )
        generated = RAGResponse(
            question="Question with evidence",
            answer="Supported fact [1].",
            citations=[source],
            rag_enabled=True,
            llm_provider="ollama",
            llm_model="test-local",
            generation_status="generated",
            context_sources=[source],
            retrieval={"candidate_count": 1},
            results=[{"chunk_id": "c1"}],
        )
        abstained = RAGResponse(
            question="Question without evidence",
            answer="Not enough evidence.",
            citations=[],
            rag_enabled=False,
            llm_provider="ollama",
            llm_model="test-local",
            generation_status="insufficient_evidence",
            context_sources=[],
            retrieval={"candidate_count": 0},
            results=[],
        )

        class FakeRAGService:
            class FakeLLM:
                def is_available(self):
                    return True

                def generate(self, **kwargs):
                    return "ready"

            llm_service = FakeLLM()

            def ask_with_trace(self, query):
                response = generated if "with evidence" in query else abstained
                return {
                    "response": response,
                    "timings_ms": {
                        "retrieval_ms": 1.0,
                        "reranking_ms": 2.0,
                        "context_building_ms": 0.2,
                        "generation_ms": 10.0 if response.rag_enabled else None,
                        "total_rag_ms": 13.2,
                    },
                }

        queries = [
            EvaluationQuery(
                "q1",
                "Question with evidence",
                relevant_chunk_ids=("c1",),
                reference_answer="Supported fact",
                answerable=True,
            ),
            EvaluationQuery(
                "q2",
                "Question without evidence",
                relevant_chunk_ids=(),
                answerable=False,
            ),
        ]
        result = RAGEvaluator(FakeRAGService(), [{"chunk_id": "c1"}]).run(queries)

        self.assertEqual(result["rag_success_rate"], 0.5)
        self.assertEqual(result["generated_with_context_rate"], 0.5)
        self.assertEqual(result["retrieval_success_rate"], 1.0)
        self.assertEqual(result["retrieval_labeled_query_count"], 1)
        self.assertEqual(result["insufficient_evidence_rate"], 0.5)
        self.assertEqual(result["average_citation_count"], 0.5)
        self.assertEqual(result["citation_validity_rate"], 1.0)
        self.assertEqual(result["abstention_accuracy"], 1.0)
        self.assertEqual(result["optional_reference_metrics"]["evaluated_answer_count"], 1)

    def test_rag_generation_not_measured_without_configured_local_model(self):
        class UnavailableRAGService:
            llm_service = SimpleNamespace(is_available=lambda: False)

        result = RAGEvaluator(UnavailableRAGService(), []).run(
            [EvaluationQuery("q1", "unlabelled")]
        )

        self.assertEqual(result["status"], "not_measured")
        self.assertIsNone(result["generation_latency"])
        self.assertEqual(result["generation_availability_rate"], 0.0)


class ResultSerializationTests(unittest.TestCase):
    def test_results_are_serialized_without_overwriting_runs(self):
        retrieval = {
            "status": "not_measured",
            "metrics": None,
            "per_query": [
                {
                    "query_id": "q1",
                    "query": "latency",
                    "relevant_chunk_ids": None,
                    "rankings": {method: ["c1"] for method in METHODS},
                    "metrics": None,
                    "timings_ms": {"bm25": 1.0},
                }
            ],
            "latency": {"bm25": {"mean_ms": 1.0}},
            "query_count": 1,
            "labeled_query_count": 0,
            "relevance_judgment_count": 0,
            "corpus_size": 1,
            "methods": list(METHODS),
            "ks": [1, 3, 5, 10],
            "latency_repeats": 1,
        }
        with tempfile.TemporaryDirectory() as temporary:
            first_id, first_dir = save_run(temporary, {"corpus_size": 1}, retrieval, None)
            second_id, second_dir = save_run(temporary, {"corpus_size": 1}, retrieval, None)

            self.assertNotEqual(first_id, second_id)
            self.assertTrue((first_dir / "retrieval_results.json").exists())
            self.assertTrue((first_dir / "latency_results.json").exists())
            self.assertTrue((first_dir / "rag_results.json").exists())
            self.assertTrue((first_dir / "retrieval_results.csv").exists())
            self.assertTrue((first_dir / "per_query_results.csv").exists())
            self.assertTrue(second_dir.exists())

    def test_human_candidate_labels_compile_to_jsonl(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "candidates.csv"
            output_path = Path(temporary) / "evaluation.jsonl"
            input_path.write_text(
                "query_id,query,chunk_id,relevant\n"
                "q1,Which passage?,c1,yes\n"
                "q1,Which passage?,c2,no\n",
                encoding="utf-8",
            )

            summary = compile_labels(
                input_path, output_path, active_chunk_ids={"c1", "c2"}
            )
            record = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])

        self.assertEqual(summary["queries"], 1)
        self.assertEqual(summary["queries_with_relevant"], 1)
        self.assertEqual(record["relevant_chunk_ids"], ["c1"])

    def test_label_compiler_rejects_chunk_id_outside_active_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            input_path = Path(temporary) / "candidates.csv"
            output_path = Path(temporary) / "evaluation.jsonl"
            input_path.write_text(
                "query_id,query,chunk_id,relevant\nq1,Question,unknown,yes\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "outside the active index"):
                compile_labels(input_path, output_path, active_chunk_ids={"known"})

            self.assertFalse(output_path.exists())


class RetrievalEvaluationIntegrationTests(unittest.TestCase):
    def test_real_retrieval_methods_run_without_labels(self):
        index = VectorIndex()
        index.load("data/vector_index.pkl")
        engine = HybridEngine(SearchEngine(index))
        evaluator = RetrievalEvaluator(EmbeddingService(), engine)
        query = EvaluationQuery(
            "latency-test",
            "computer science information technology exam at IIT Roorkee",
        )

        result = evaluator.run([query], ks=(1, 3, 5, 10), repeats=1)

        self.assertEqual(result["status"], "not_measured")
        self.assertIsNone(result["metrics"])
        self.assertEqual(result["query_count"], 1)
        self.assertEqual(result["labeled_query_count"], 0)
        self.assertEqual(set(result["methods"]), set(METHODS))
        self.assertTrue(all(result["per_query"][0]["rankings"].values()))
        self.assertTrue(result["latency"]["bm25"])
        self.assertTrue(result["latency"]["vector"])
        self.assertTrue(result["latency"]["hybrid"])
        self.assertTrue(result["latency"]["reranker"])
        self.assertTrue(result["latency"]["total_retrieval"])


if __name__ == "__main__":
    unittest.main()
