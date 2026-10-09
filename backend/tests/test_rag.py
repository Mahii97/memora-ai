import asyncio
import importlib
import unittest
from urllib.error import URLError
from unittest.mock import patch

from app.citations.citation_service import CitationService
from app.embeddings.embedding_service import EmbeddingService
from app.llm.llm_service import LLMService, LLMUnavailableError, OllamaRuntime
from app.models.rag_response import RAGResponse
from app.rag.context_builder import ContextBuilder
from app.rag.prompt_builder import SYSTEM_PROMPT, build_prompt
from app.rag.rag_service import (
    DEFAULT_INSUFFICIENT_MESSAGE,
    RAGService,
)
from app.rag.query_understanding import QueryUnderstanding
from app.rag.answer_planner import AnswerPlanner
from app.retrieval.hybrid_engine import HybridEngine
from app.vector.search import SearchEngine
from app.vector.vector_index import VectorIndex


class FakeRuntime:
    def __init__(self, available=True, answer="Supported answer [1]."):
        self.available = available
        self.answer = answer
        self.calls = []

    def is_available(self, model):
        return self.available and model == "test-local-model"

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return self.answer


class FakeEmbeddingService:
    def embed_query(self, question):
        return [0.1, 0.2, 0.3]


class FakeRetrievalEngine:
    def __init__(self, results=None):
        self.results = results or []
        self.query = None
        self.query_vector = None

    def search_with_trace(self, query, query_vector, **kwargs):
        self.query = query
        self.query_vector = query_vector
        self.kwargs = kwargs
        return {
            "vector_candidate_count": 2,
            "bm25_candidate_count": 2,
            "candidate_count": 2,
            "results": self.results,
        }


def make_chunk(chunk_id="chunk-1", text="Normalization organizes database tables.", score=5.0):
    return {
        "chunk_id": chunk_id,
        "text": text,
        "source_name": "DBMS Notes.pdf",
        "source_type": "pdf",
        "location": "Page 12",
        "author": "Unknown",
        "created_at": "2026-09-01",
        "vector_score": 0.82,
        "bm25_score": 3.25,
        "hybrid_score": 0.91,
        "reranker_score": score,
        "final_rank": 1,
        "explanation": {
            "vector_score": 0.82,
            "bm25_score": 3.25,
            "hybrid_score": 0.91,
            "reranker_score": score,
            "final_rank": 1,
        },
    }


class ContextBuilderTests(unittest.TestCase):
    def test_context_contains_source_metadata_and_chunk_text(self):
        package = ContextBuilder().build([make_chunk()])

        self.assertIn("Source [1]: DBMS Notes.pdf", package["context"])
        self.assertIn("Source type: pdf", package["context"])
        self.assertIn("Location: Page 12", package["context"])
        self.assertIn("Date: 2026-09-01", package["context"])
        self.assertIn("Normalization organizes database tables.", package["context"])
        self.assertEqual(package["sources"][0]["chunk_id"], "chunk-1")

    def test_context_obeys_chunk_and_character_limits(self):
        chunks = [
            make_chunk("one", "A" * 300),
            make_chunk("two", "B" * 300),
        ]
        package = ContextBuilder(max_chunks=1, max_characters=220).build(chunks)

        self.assertLessEqual(len(package["context"]), 220)
        self.assertEqual(len(package["sources"]), 1)
        self.assertEqual(package["sources"][0]["chunk_id"], "one")
        self.assertNotIn("B" * 20, package["context"])

    def test_zero_context_limit_returns_no_sources(self):
        package = ContextBuilder(max_chunks=0).build([make_chunk()])

        self.assertEqual(package["context"], "")
        self.assertEqual(package["sources"], [])
        self.assertEqual(package["metadata"]["selected_chunks"], 0)

    def test_duplicate_chunks_are_removed_and_sources_are_diversified(self):
        same_source_a = make_chunk("a", "A distinctive paragraph. " * 10)
        same_source_b = make_chunk("b", "A different paragraph from this same source. " * 8)
        other_source = make_chunk("c", "A separate source contains another topic. " * 8)
        other_source["source_name"] = "Other Notes.pdf"
        package = ContextBuilder(max_chunks=3).build([same_source_a, same_source_b, other_source])

        self.assertEqual(package["metadata"]["selected_chunks"], 3)
        self.assertEqual(package["sources"][0]["source_name"], "DBMS Notes.pdf")
        self.assertEqual(package["sources"][1]["source_name"], "Other Notes.pdf")

    def test_exact_duplicate_text_from_distinct_chunk_ids_is_removed(self):
        original = make_chunk("one", "Same evidence passage. " * 12)
        duplicate = make_chunk("two", "Same evidence passage. " * 12)
        duplicate["location"] = "Page 12 (ocr)"
        original["location"] = "Page 12 (text)"
        package = ContextBuilder().build([original, duplicate])
        self.assertEqual(package["metadata"]["duplicate_chunks_removed"], 1)
        self.assertEqual(package["metadata"]["selected_chunks"], 1)


class QueryUnderstandingTests(unittest.TestCase):
    def setUp(self):
        self.analyzer = QueryUnderstanding()

    def test_greeting_is_conversational_and_does_not_request_retrieval(self):
        plan = self.analyzer.analyze("hi")
        self.assertEqual(plan["intent"], "conversational")

    def test_gate_question_number_adds_aliases_and_exact_question_match(self):
        plan = self.analyzer.analyze("GATE Q43")
        self.assertEqual(plan["question_number"], 43)
        self.assertEqual(plan["intent"], "question_lookup")
        self.assertIn("Q.43", plan["retrieval_query"])
        features = self.analyzer.match_features(plan, {"text": "Q.43: Solve the problem", "source_name": "GATE DA.pdf"})
        self.assertTrue(features["question_number_match"])

    def test_domain_search_and_typo_normalization(self):
        broad = self.analyzer.analyze("gate ds")
        self.assertEqual(broad["intent"], "document_search")
        self.assertIn("Data Science and Artificial Intelligence", broad["retrieval_query"])
        self.assertIn("DA", broad["entity_aliases"])

        paper = self.analyzer.analyze("find gate ds paper data")
        self.assertEqual(paper["intent"], "document_search")

        typo = self.analyzer.analyze("can you please about date ds paper")
        self.assertEqual(typo["intent"], "document_search")
        self.assertTrue(typo["normalizations"])
        self.assertNotIn("date", typo["retrieval_query"].split())
        self.assertIn("science", typo["retrieval_query"])

    def test_question_answering_document_summary_and_memory_summary_are_distinct(self):
        self.assertEqual(self.analyzer.analyze("solve Q43")["intent"], "question_answering")
        self.assertEqual(self.analyzer.analyze("what is this paper about?")["intent"], "document_summary")
        self.assertEqual(self.analyzer.analyze("what do you remember about my projects?")["intent"], "memory_summary")
        self.assertEqual(self.analyzer.analyze("summarize my most relevant memories")["intent"], "memory_summary")

    def test_month_year_temporal_query_is_a_range(self):
        from datetime import date
        plan = self.analyzer.analyze("documents from September 2026", today=date(2026, 10, 1))
        self.assertEqual(plan["intent"], "temporal_summary")
        self.assertEqual(plan["time_range"], {"start": "2026-09-01", "end": "2026-09-30"})

    def test_temporal_query_uses_deterministic_calendar_range(self):
        from datetime import date
        plan = self.analyzer.analyze("What did I work on last week?", today=date(2026, 10, 1))
        self.assertEqual(plan["intent"], "temporal_summary")
        self.assertEqual(plan["time_range"], {"start": "2026-09-21", "end": "2026-09-27"})
        month_range = self.analyzer.analyze(
            "What happened between January and March?", today=date(2026, 10, 1)
        )
        self.assertEqual(
            month_range["time_range"], {"start": "2026-01-01", "end": "2026-03-31"}
        )

    def test_intents_cover_comparison_summary_list_and_explanation(self):
        cases = {
            "Compare my two project reports": "comparison",
            "Summarize my project notes": "memory_summary",
            "List the action items in my notes": "list_extraction",
            "Explain the database normalization notes": "explanation",
        }
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(self.analyzer.analyze(question)["intent"], expected)

    def test_document_candidates_aggregate_and_prefer_exact_gated_entity_evidence(self):
        plan = self.analyzer.analyze("gate ds")
        gate_document = make_chunk(
            "gate-q1", "Data Science & Artificial Intelligence (DA) GATE 2026"
        )
        gate_document["source_name"] = "da gate q1.pdf"
        gate_document["query_match"] = self.analyzer.match_features(plan, gate_document)
        unrelated = make_chunk("unrelated", "The project has a date field.")
        unrelated["source_name"] = "MemoraAI documentation.pdf"
        unrelated["query_match"] = self.analyzer.match_features(plan, unrelated)
        result = RAGService._document_payloads([unrelated, gate_document, gate_document], plan)
        self.assertEqual(result[0]["source_name"], "da gate q1.pdf")
        self.assertEqual(result[0]["matching_chunk_count"], 1)

    def test_answer_planner_gives_distinct_task_instructions(self):
        planner = AnswerPlanner()
        comparison = planner.plan({"intent": "comparison"})
        summary = planner.plan({"intent": "summarization"})
        discovery = planner.plan({"intent": "document_search"})
        document_summary = planner.plan({"intent": "document_summary"})
        question_lookup = planner.plan({"intent": "question_lookup", "question_number": 43})
        question_answer = planner.plan({"intent": "question_answering", "question_number": 43})
        memory_summary = planner.plan({"intent": "memory_summary"})
        self.assertIn("Compare only", comparison)
        self.assertIn("Synthesize", summary)
        self.assertNotEqual(comparison, summary)
        self.assertIn("locate a document", discovery)
        self.assertIn("Summarize the supplied document evidence", document_summary)
        self.assertIn("Do not substitute a different question", question_lookup)
        self.assertIn("Answer the exact numbered question", question_answer)
        self.assertIn("distinct retrieved memories", memory_summary)


class CitationTests(unittest.TestCase):
    def test_citations_are_built_only_from_supplied_context_sources(self):
        sources = ContextBuilder().build([make_chunk()])["sources"]
        citations = CitationService.build(sources)

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0]["citation_id"], 1)
        self.assertEqual(citations[0]["chunk_id"], "chunk-1")

    def test_unknown_generated_citation_ids_are_removed(self):
        service = CitationService()
        sources = ContextBuilder().build([make_chunk()])["sources"]

        answer = service.remove_unknown_citations(
            "Normalization reduces redundancy [1]. Unsupported claim [9].", sources
        )

        self.assertIn("[1]", answer)
        self.assertNotIn("[9]", answer)


class PromptTests(unittest.TestCase):
    def test_prompt_uses_strict_grounding_instruction_and_context(self):
        prompt = build_prompt("What is normalization?", "Source [1]: DBMS Notes")

        self.assertIn("only the retrieved context", prompt["system_prompt"])
        self.assertIn("Do not use outside knowledge", prompt["system_prompt"])
        self.assertIn("Do not reveal hidden reasoning", prompt["system_prompt"])
        self.assertEqual(prompt["prompt"], "What is normalization?")
        self.assertEqual(prompt["context"], "Source [1]: DBMS Notes")
        self.assertEqual(prompt["system_prompt"], SYSTEM_PROMPT)


class LLMServiceTests(unittest.TestCase):
    def test_local_provider_model_and_base_url_configuration(self):
        runtime = FakeRuntime()
        service = LLMService(
            provider="ollama",
            model_name="test-local-model",
            base_url="http://127.0.0.1:11434/",
            runtime=runtime,
        )

        self.assertEqual(service.provider, "ollama")
        self.assertEqual(service.model_name, "test-local-model")
        self.assertEqual(service.base_url, "http://127.0.0.1:11434")
        self.assertTrue(service.is_available())
        self.assertEqual(
            service.generate("Question?", "Source [1]", "Use context."),
            "Supported answer [1].",
        )
        self.assertEqual(len(runtime.calls), 1)

    def test_unconfigured_llm_is_unavailable_without_cloud_fallback(self):
        service = LLMService(provider="ollama", model_name="", runtime=None)

        self.assertFalse(service.is_available())

    def test_ollama_runtime_rejects_nonlocal_base_urls(self):
        with self.assertRaises(ValueError):
            LLMService(
                provider="ollama",
                model_name="test-local-model",
                base_url="https://example.com",
            )

    def test_missing_model_health_check_gives_exact_pull_command(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                return b'{"models":[{"name":"other-model"}]}'

        runtime = OllamaRuntime("http://127.0.0.1:11434")
        with patch("app.llm.llm_service.urlopen", return_value=Response()):
            result = runtime.check_model("qwen2.5:3b")

        self.assertFalse(result["available"])
        self.assertIn("ollama pull qwen2.5:3b", result["detail"])

    def test_connection_health_check_explains_how_to_recover(self):
        runtime = OllamaRuntime("http://127.0.0.1:11434")
        with patch("app.llm.llm_service.urlopen", side_effect=URLError("connection refused")):
            result = runtime.check_model("qwen2.5:3b")

        self.assertFalse(result["available"])
        self.assertIn("Cannot connect to Ollama", result["detail"])
        self.assertIn("Start the local Ollama service", result["detail"])

    def test_generation_timeout_is_reported_as_local_llm_unavailable(self):
        runtime = OllamaRuntime("http://127.0.0.1:11434")
        with patch("app.llm.llm_service.urlopen", side_effect=TimeoutError()):
            with self.assertRaisesRegex(LLMUnavailableError, "timed out after 8 seconds"):
                runtime.generate("qwen2.5:3b", "Question", "Context", "System", 8)


class RAGServiceTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.llm = LLMService(
            provider="ollama",
            model_name="test-local-model",
            runtime=self.runtime,
        )
        self.engine = FakeRetrievalEngine([make_chunk()])
        self.service = RAGService(
            self.engine,
            embedding_service=FakeEmbeddingService(),
            llm_service=self.llm,
        )

    def test_insufficient_evidence_skips_generation(self):
        service = RAGService(
            self.engine,
            embedding_service=FakeEmbeddingService(),
            llm_service=self.llm,
            min_reranker_score=100.0,
        )

        response = service.ask("Unrelated question")

        self.assertEqual(response.generation_status, "insufficient_evidence")
        self.assertEqual(response.answer, DEFAULT_INSUFFICIENT_MESSAGE)
        self.assertFalse(response.rag_enabled)
        self.assertEqual(self.runtime.calls, [])

    def test_negative_reranker_score_is_not_a_default_rejection_threshold(self):
        engine = FakeRetrievalEngine([make_chunk(score=-5.03)])
        service = RAGService(
            engine,
            embedding_service=FakeEmbeddingService(),
            llm_service=self.llm,
        )

        response = service.ask("Summarize the relevant memory")

        self.assertEqual(response.generation_status, "generated")
        self.assertTrue(response.rag_enabled)
        self.assertEqual(response.retrieval["context_chunk_count"], 1)
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.citations[0].chunk_id, "chunk-1")
        self.assertEqual(len(self.runtime.calls), 1)
        self.assertIn("Normalization organizes database tables.", self.runtime.calls[0]["context"])

    def test_llm_unavailable_returns_retrieval_results(self):
        unavailable = LLMService(
            provider="ollama",
            model_name="test-local-model",
            runtime=FakeRuntime(available=False),
        )
        service = RAGService(
            self.engine,
            embedding_service=FakeEmbeddingService(),
            llm_service=unavailable,
        )

        response = service.ask("What is normalization?")

        self.assertEqual(response.generation_status, "unavailable")
        self.assertFalse(response.rag_enabled)
        self.assertIsNone(response.answer)
        self.assertEqual(len(response.results), 1)
        self.assertEqual(response.results[0]["source_name"], "DBMS Notes.pdf")
        self.assertEqual(response.results[0]["reranker_score"], 5.0)
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.citations[0].chunk_id, "chunk-1")
        self.assertEqual(len(response.context_sources), 1)

    def test_generated_response_schema_citations_and_retrieval_metadata(self):
        response = self.service.ask("What is normalization?")
        serialized = response.model_dump()
        validated = RAGResponse.model_validate(serialized)

        self.assertEqual(validated.generation_status, "generated")
        self.assertTrue(validated.rag_enabled)
        self.assertIn("[1]", validated.answer)
        self.assertEqual(len(validated.citations), 1)
        self.assertEqual(validated.citations[0].chunk_id, "chunk-1")
        self.assertEqual(validated.retrieval["context_chunk_count"], 1)
        self.assertEqual(validated.retrieval["context_chunks"], 1)
        self.assertGreaterEqual(validated.timings_ms["query_embedding_ms"], 0)
        self.assertGreaterEqual(validated.timings_ms["total_rag_ms"], 0)
        self.assertGreaterEqual(validated.timings_ms["generation_ms"], 0)
        self.assertIn("DBMS Notes.pdf", self.runtime.calls[0]["context"])
        self.assertEqual(validated.intent, "direct_question")
        self.assertEqual(validated.retrieval["selected_chunks"], 1)
        self.assertTrue(validated.unique_results)

    def test_full_retrieval_to_rag_integration_uses_mock_llm(self):
        vector_index = VectorIndex()
        vector_index.load("data/vector_index.pkl")
        query = "computer science information technology exam at IIT Roorkee"
        embedding_service = EmbeddingService()
        retrieval_engine = HybridEngine(SearchEngine(vector_index))
        runtime = FakeRuntime(answer="The material is an exam document [1].")
        llm_service = LLMService(
            provider="ollama",
            model_name="test-local-model",
            runtime=runtime,
        )
        rag_service = RAGService(
            retrieval_engine,
            embedding_service=embedding_service,
            llm_service=llm_service,
        )

        response = rag_service.ask(query)

        self.assertTrue(response.rag_enabled)
        self.assertEqual(response.generation_status, "generated")
        self.assertTrue(response.results)
        self.assertTrue(response.citations)
        self.assertLessEqual(response.retrieval["candidate_count"], 20)
        self.assertLessEqual(response.retrieval["context_chunk_count"], 5)
        self.assertTrue(runtime.calls)
        self.assertIn("Source [1]:", runtime.calls[0]["context"])
        self.assertEqual(
            response.citations[0].source_name,
            response.context_sources[0].source_name,
        )

    def test_conversation_skips_embedding_and_retrieval_but_uses_local_llm(self):
        response = self.service.ask("hi")

        self.assertEqual(response.intent, "conversational")
        self.assertEqual(response.generation_status, "generated")
        self.assertEqual(self.engine.query, None)
        self.assertEqual(response.results, [])
        self.assertEqual(response.citations, [])
        self.assertEqual(len(self.runtime.calls), 1)
        self.assertEqual(self.runtime.calls[0]["context"], "")

    def test_document_search_response_groups_retrieved_document_details(self):
        response = self.service.ask("Find my DBMS notes document")
        self.assertEqual(response.intent, "document_search")
        self.assertEqual(response.documents[0]["source_name"], "DBMS Notes.pdf")
        self.assertEqual(response.documents[0]["locations"], ["Page 12"])
        self.assertTrue(response.documents[0]["excerpts"])

    def test_unique_results_deduplicate_repeated_chunk_ids(self):
        duplicate = make_chunk("chunk-1", "Normalization organizes database tables.")
        engine = FakeRetrievalEngine([make_chunk(), duplicate])
        service = RAGService(engine, embedding_service=FakeEmbeddingService(), llm_service=self.llm)
        response = service.ask("What is normalization?")
        self.assertEqual(len(response.results), 2)
        self.assertEqual(len(response.unique_results), 1)

    def test_context_chunk_metadata_matches_the_context_sent_to_local_llm(self):
        chunks = [make_chunk(f"chunk-{index}", f"Unique evidence passage number {index}.") for index in range(5)]
        engine = FakeRetrievalEngine(chunks)
        service = RAGService(engine, embedding_service=FakeEmbeddingService(), llm_service=self.llm)
        response = service.ask("Summarize my most relevant memories")
        sent_chunks = self.runtime.calls[-1]["context"].count("Text: ")
        self.assertEqual(sent_chunks, 5)
        self.assertEqual(response.retrieval["context_chunks"], sent_chunks)
        self.assertEqual(response.retrieval["context_chunk_count"], sent_chunks)
        self.assertEqual(response.retrieved_results, response.results)

    def test_empty_evidence_abstains_without_fabricating_an_answer(self):
        engine = FakeRetrievalEngine([])
        service = RAGService(engine, embedding_service=FakeEmbeddingService(), llm_service=self.llm)
        response = service.ask("What color is the invisible dragon in my garage?")
        self.assertEqual(response.generation_status, "insufficient_evidence")
        self.assertEqual(response.answer, DEFAULT_INSUFFICIENT_MESSAGE)
        self.assertEqual(self.runtime.calls, [])

    def test_ask_api_keeps_retrieval_fields_and_adds_rag_fields(self):
        ask_api = importlib.import_module("app.api.ask")

        class FakeApiRAGService:
            def ask(self, question):
                return RAGResponse(
                    question=question,
                    answer="Supported answer [1].",
                    citations=[],
                    rag_enabled=True,
                    llm_provider="ollama",
                    llm_model="test-local-model",
                    generation_status="generated",
                    context_sources=[],
                    retrieval={"candidate_count": 1},
                    results=[
                        {
                            "score": 0.9,
                            "vector_score": 0.8,
                            "bm25_score": 3.0,
                            "hybrid_score": 0.9,
                            "reranker_score": 5.0,
                            "final_rank": 1,
                            "explanation": {"reranker_score": 5.0},
                            "source_type": "pdf",
                            "source_name": "Notes.pdf",
                            "author": "Unknown",
                            "location": "Page 2",
                            "date": "2026-09-01",
                            "preview": "Evidence",
                            "chunk_id": "chunk-1",
                        }
                    ],
                )

        class FakeResources:
            def get_rag_service(self):
                return FakeApiRAGService()

        original_resources = ask_api.search_resources
        ask_api.search_resources = FakeResources()
        try:
            response = asyncio.run(
                ask_api.ask(ask_api.QuestionRequest(question="Question?"))
            )
        finally:
            ask_api.search_resources = original_resources

        self.assertTrue(response["rag_enabled"])
        self.assertEqual(response["answer"], "Supported answer [1].")
        self.assertEqual(response["llm_model"], "test-local-model")
        self.assertEqual(response["results"][0]["vector_score"], 0.8)
        self.assertEqual(response["results"][0]["bm25_score"], 3.0)
        self.assertEqual(response["results"][0]["final_rank"], 1)


if __name__ == "__main__":
    unittest.main()
