import os
import re
from time import perf_counter
from pathlib import Path

from dotenv import load_dotenv

from app.citations.citation_service import CitationService
from app.embeddings.embedding_service import EmbeddingService
from app.llm.llm_service import LLMService, LLMUnavailableError
from app.models.rag_response import RAGResponse
from app.rag.context_builder import ContextBuilder
from app.rag.prompt_builder import build_prompt
from app.rag.query_understanding import QueryUnderstanding
from app.rag.answer_planner import AnswerPlanner, CONVERSATIONAL_SYSTEM_PROMPT


BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_ROOT / ".env")

DEFAULT_INSUFFICIENT_MESSAGE = (
    "I could not find enough supporting information in your knowledge base "
    "to answer this confidently."
)
DEFAULT_UNAVAILABLE_MESSAGE = (
    "Local generation is unavailable; showing retrieved sources only."
)


class RAGService:
    def __init__(
        self,
        retrieval_engine,
        embedding_service: EmbeddingService | None = None,
        llm_service: LLMService | None = None,
        context_builder: ContextBuilder | None = None,
        citation_service: CitationService | None = None,
        min_reranker_score: float | None = None,
    ):
        self.retrieval_engine = retrieval_engine
        self.embedding_service = embedding_service or EmbeddingService()
        self.llm_service = llm_service or LLMService()
        self.context_builder = context_builder or ContextBuilder()
        self.citation_service = citation_service or CitationService()
        self.query_understanding = QueryUnderstanding()
        self.answer_planner = AnswerPlanner()
        configured_min_score = os.getenv("RAG_MIN_RERANKER_SCORE", "").strip()
        if min_reranker_score is not None:
            self.min_reranker_score = float(min_reranker_score)
        elif configured_min_score:
            self.min_reranker_score = float(configured_min_score)
        else:
            # CrossEncoder scores are raw logits, not calibrated probabilities.
            # Negative scores can still rank the best available evidence first.
            self.min_reranker_score = None
        self.insufficient_message = os.getenv(
            "RAG_INSUFFICIENT_EVIDENCE_MESSAGE", DEFAULT_INSUFFICIENT_MESSAGE
        )
        self.unavailable_message = os.getenv(
            "RAG_GENERATION_UNAVAILABLE_MESSAGE", DEFAULT_UNAVAILABLE_MESSAGE
        )

    @staticmethod
    def _result_payload(chunk: dict) -> dict:
        return {
            "score": round(chunk["hybrid_score"], 3),
            "vector_score": chunk["vector_score"],
            "bm25_score": chunk["bm25_score"],
            "hybrid_score": chunk["hybrid_score"],
            "reranker_score": chunk["reranker_score"],
            "final_rank": chunk["final_rank"],
            "explanation": chunk["explanation"],
            "source_type": chunk["source_type"],
            "source_name": chunk["source_name"],
            "author": chunk["author"],
            "location": chunk["location"],
            "date": chunk["created_at"],
            "preview": chunk["text"][:220],
            "text": chunk["text"],
            "chunk_id": chunk["chunk_id"],
            "query_match": chunk.get("query_match", {}),
        }

    def _response(
        self,
        question: str,
        retrieval: dict,
        answer: str | None,
        citations: list[dict],
        context_sources: list[dict],
        rag_enabled: bool,
        generation_status: str,
        generation_message: str | None = None,
        understanding: dict | None = None,
        answer_task: str | None = None,
        context_metadata: dict | None = None,
        unique_results: list[dict] | None = None,
        documents: list[dict] | None = None,
    ) -> RAGResponse:
        understanding = understanding or {}
        context_metadata = context_metadata or {}
        result_payloads = [self._result_payload(chunk) for chunk in retrieval.get("results", [])]
        return RAGResponse(
            question=question,
            answer=answer,
            citations=citations,
            rag_enabled=rag_enabled,
            llm_provider=self.llm_service.provider,
            llm_model=self.llm_service.model_name or None,
            generation_status=generation_status,
            generation_message=generation_message,
            context_sources=context_sources,
            retrieval={
                "vector_candidate_count": retrieval.get("vector_candidate_count", 0),
                "bm25_candidate_count": retrieval.get("bm25_candidate_count", 0),
                "candidate_count": retrieval.get("candidate_count", 0),
                "context_chunk_count": len(context_sources),
                "context_chunks": len(context_sources),
                "retrieved_result_count": len(retrieval.get("results", [])),
                **({"temporal_candidate_count": retrieval["temporal_candidate_count"]}
                   if "temporal_candidate_count" in retrieval else {}),
                **({"time_range": understanding.get("time_range")}
                   if understanding.get("time_range") else {}),
                **context_metadata,
            },
            results=result_payloads,
            retrieved_results=result_payloads,
            unique_results=unique_results if unique_results is not None else result_payloads,
            documents=documents or [],
            intent=understanding.get("intent"),
            query_understanding=understanding,
            answer_task=answer_task,
        )

    def ask(self, question: str) -> RAGResponse:
        return self.ask_with_trace(question)["response"]

    def ask_with_trace(self, question: str) -> dict:
        total_start = perf_counter()
        analysis_start = perf_counter()
        understanding = self.query_understanding.analyze(question)
        answer_task = self.answer_planner.plan(understanding)
        query_understanding_ms = (perf_counter() - analysis_start) * 1000
        if understanding["intent"] == "conversational":
            response, generation_ms = self._answer_conversational(question, understanding)
            response.timings_ms = {
                "query_understanding_ms": query_understanding_ms,
                "query_embedding_ms": 0.0,
                "retrieval_ms": 0.0,
                "hybrid_retrieval_ms": 0.0,
                "reranking_ms": 0.0,
                "context_building_ms": 0.0,
                "generation_ms": generation_ms,
                "total_rag_ms": (perf_counter() - total_start) * 1000,
            }
            return {"response": response, "retrieval": {}, "context": "", "timings_ms": response.timings_ms}

        embedding_start = perf_counter()
        query_vector = self.embedding_service.embed_query(understanding["retrieval_query"])
        query_embedding_ms = (perf_counter() - embedding_start) * 1000
        retrieval = self.retrieval_engine.search_with_trace(
            question, query_vector,
            retrieval_query=understanding["retrieval_query"],
            query_plan=understanding,
        )
        context_start = perf_counter()
        context_package = self.context_builder.build(retrieval["results"], understanding)
        context_building_ms = (perf_counter() - context_start) * 1000
        response, generation_ms = self._answer(question, retrieval, context_package, understanding, answer_task)
        response.timings_ms = {
            "query_understanding_ms": query_understanding_ms,
            "query_embedding_ms": query_embedding_ms,
            "retrieval_ms": retrieval.get("total_retrieval_latency_ms")
            or retrieval.get("hybrid_latency_ms"),
            "hybrid_retrieval_ms": retrieval.get("hybrid_latency_ms"),
            "reranking_ms": retrieval.get("reranker_latency_ms"),
            "context_building_ms": context_building_ms,
            "generation_ms": generation_ms,
            "total_rag_ms": (perf_counter() - total_start) * 1000,
        }
        return {
            "response": response,
            "retrieval": retrieval,
            "context": context_package["context"],
            "timings_ms": response.timings_ms,
        }

    def _availability(self) -> tuple[bool, str | None]:
        try:
            health = self.llm_service.health_check()
            return bool(health.get("available")), health.get("detail")
        except AttributeError:
            try:
                return bool(self.llm_service.is_available()), None
            except Exception as error:
                return False, str(error)
        except Exception as error:
            return False, str(error)

    def _answer_conversational(self, question: str, understanding: dict):
        retrieval = {"results": []}
        available, detail = self._availability()
        if not available:
            return self._response(
                question, retrieval, None, [], [], False, "unavailable",
                detail or self.unavailable_message, understanding=understanding,
                answer_task="Respond briefly without searching personal memories.",
            ), None
        start = perf_counter()
        try:
            answer = self.llm_service.generate(
                prompt=question, context="", system_prompt=CONVERSATIONAL_SYSTEM_PROMPT
            )
        except LLMUnavailableError as error:
            return self._response(
                question, retrieval, None, [], [], False, "unavailable",
                str(error) or self.unavailable_message, understanding=understanding,
            ), None
        return self._response(
            question, retrieval, answer, [], [], False, "generated",
            understanding=understanding,
            answer_task="Respond briefly without searching personal memories.",
        ), (perf_counter() - start) * 1000

    @staticmethod
    def _unique_result_payloads(results: list[dict], sources: list[dict]) -> list[dict]:
        source_ids = [str(item.get("chunk_id")) for item in sources if item.get("chunk_id")]
        if not source_ids:
            return []
        by_id = {}
        for item in results:
            chunk_id = str(item.get("chunk_id") or "")
            if chunk_id and chunk_id not in by_id:
                by_id[chunk_id] = item
        selected = [by_id[chunk_id] for chunk_id in source_ids if chunk_id in by_id]
        unique = []
        seen_ids = set()
        seen_text_locations = set()
        for item in selected:
            chunk_id = str(item.get("chunk_id") or "")
            text_key = (
                str(item.get("source_name") or "").casefold(),
                ContextBuilder._normalized_location(str(item.get("location") or "")),
                ContextBuilder._normalized_text(str(item.get("text") or "")),
            )
            if (chunk_id and chunk_id in seen_ids) or (text_key[2] and text_key in seen_text_locations):
                continue
            if chunk_id:
                seen_ids.add(chunk_id)
            if text_key[2]:
                seen_text_locations.add(text_key)
            unique.append(item)
        return [RAGService._result_payload(item) for item in unique]

    @staticmethod
    def _document_payloads(
        results: list[dict], plan: dict, context_sources: list[dict] | None = None
    ) -> list[dict]:
        selected_sources = {
            str(item.get("source_name") or "").casefold()
            for item in (context_sources or [])
            if item.get("source_name")
        }
        documents = {}
        seen_evidence = set()
        for item in results:
            name = str(item.get("source_name") or "Unknown source")
            if selected_sources and name.casefold() not in selected_sources:
                continue
            chunk_id = str(item.get("chunk_id") or "")
            evidence_key = (
                (name.casefold(), chunk_id)
                if chunk_id
                else (
                    name.casefold(),
                    ContextBuilder._normalized_location(str(item.get("location") or "")),
                    ContextBuilder._normalized_text(str(item.get("text") or "")),
                )
            )
            if evidence_key in seen_evidence:
                continue
            seen_evidence.add(evidence_key)
            key = name.casefold()
            document = documents.setdefault(key, {
                "source_name": name,
                "source_type": item.get("source_type"),
                "author": item.get("author"),
                "date": item.get("date"),
                "locations": [],
                "excerpts": [],
                "matched_terms": [],
                "matched_entities": [],
                "chunk_ids": [],
                "matching_chunk_count": 0,
                "_best_reranker_score": float("-inf"),
                "_best_hybrid_score": float("-inf"),
                "_best_vector_score": float("-inf"),
                "_best_bm25_score": float("-inf"),
            })
            document["matching_chunk_count"] += 1
            if item.get("location") and item["location"] not in document["locations"]:
                document["locations"].append(item["location"])
            if item.get("chunk_id"):
                document["chunk_ids"].append(item["chunk_id"])
            body = str(item.get("text") or "")
            if body:
                document["excerpts"].append({
                    "text": body[:600],
                    "location": item.get("location"),
                    "reranker_score": item.get("reranker_score"),
                })
            source_and_text = f"{name} {body}".casefold()
            for entity in plan.get("entity_aliases", []):
                if re.search(rf"\b{re.escape(entity.casefold())}\b", source_and_text):
                    if entity not in document["matched_entities"]:
                        document["matched_entities"].append(entity)
            document["matched_terms"].extend(
                term for term in plan.get("keywords", [])
                if re.search(rf"\b{re.escape(term.casefold())}\b", source_and_text)
            )
            features = item.get("query_match", {})
            document["_best_reranker_score"] = max(document["_best_reranker_score"], float(item["reranker_score"]) if item.get("reranker_score") is not None else float("-inf"))
            document["_best_hybrid_score"] = max(document["_best_hybrid_score"], float(item["hybrid_score"]) if item.get("hybrid_score") is not None else float("-inf"))
            document["_best_vector_score"] = max(document["_best_vector_score"], float(item["vector_score"]) if item.get("vector_score") is not None else float("-inf"))
            document["_best_bm25_score"] = max(document["_best_bm25_score"], float(item["bm25_score"]) if item.get("bm25_score") is not None else float("-inf"))
            document["_source_term_matches"] = max(document.get("_source_term_matches", 0), int(features.get("source_term_matches", 0)))
            document["_topic_term_matches"] = max(document.get("_topic_term_matches", 0), int(features.get("topic_term_matches", 0)))
        for document in documents.values():
            document["matched_terms"] = list(dict.fromkeys(document["matched_terms"]))
            document["chunk_ids"] = list(dict.fromkeys(document["chunk_ids"]))
            document["excerpts"].sort(
                key=lambda item: float(item.get("reranker_score") or float("-inf")),
                reverse=True,
            )
            document["excerpt_details"] = document["excerpts"][:2]
            document["excerpts"] = [item["text"] for item in document["excerpt_details"]]
            document["best_scores"] = {
                "reranker": document.pop("_best_reranker_score"),
                "hybrid": document.pop("_best_hybrid_score"),
                "vector": document.pop("_best_vector_score"),
                "bm25": document.pop("_best_bm25_score"),
            }
        ranked = list(documents.values())
        ranked.sort(
            key=lambda item: (
                len(item["matched_entities"]),
                item.get("_source_term_matches", 0),
                item.get("_topic_term_matches", 0),
                item["best_scores"]["reranker"],
                item["best_scores"]["hybrid"],
            ),
            reverse=True,
        )
        for index, item in enumerate(ranked, start=1):
            item.pop("_source_term_matches", None)
            item.pop("_topic_term_matches", None)
            item["document_rank"] = index
        return ranked[:5]

    @staticmethod
    def _document_payloads_for_plan(
        results: list[dict], plan: dict, context_sources: list[dict] | None = None
    ) -> list[dict]:
        if plan.get("intent") not in {"document_search", "document_summary"}:
            return []
        return RAGService._document_payloads(results, plan, context_sources)

    def _answer(
        self, question: str, retrieval: dict, context_package: dict,
        understanding: dict, answer_task: str,
    ) -> tuple[RAGResponse, float | None]:
        results = retrieval["results"]
        context = context_package["context"]
        sources = context_package["sources"]

        strongest_score = max(
            (float(item.get("reranker_score", float("-inf"))) for item in results),
            default=float("-inf"),
        )
        below_score_cutoff = (
            self.min_reranker_score is not None
            and strongest_score < self.min_reranker_score
        )
        if not sources or below_score_cutoff:
            return self._response(
                question=question,
                retrieval=retrieval,
                answer=self.insufficient_message,
                citations=[],
                context_sources=[],
                rag_enabled=False,
                generation_status="insufficient_evidence",
                understanding=understanding, answer_task=answer_task,
                context_metadata=context_package.get("metadata"),
                documents=self._document_payloads_for_plan(
                    retrieval.get("candidate_results", results), understanding, sources
                ),
            ), None

        llm_available, availability_error = self._availability()

        if not llm_available:
            citations = self.citation_service.build(sources)
            return self._response(
                question=question,
                retrieval=retrieval,
                answer=None,
                citations=citations,
                context_sources=citations,
                rag_enabled=False,
                generation_status="unavailable",
                generation_message=availability_error or self.unavailable_message,
                understanding=understanding, answer_task=answer_task,
                context_metadata=context_package.get("metadata"),
                documents=self._document_payloads_for_plan(
                    retrieval.get("candidate_results", results), understanding, sources
                ),
            ), None

        prompt = build_prompt(question, context, answer_task)
        generation_start = perf_counter()
        try:
            answer = self.llm_service.generate(
                prompt=prompt["prompt"],
                context=prompt["context"],
                system_prompt=prompt["system_prompt"],
            )
        except LLMUnavailableError as error:
            citations = self.citation_service.build(sources)
            return self._response(
                question=question,
                retrieval=retrieval,
                answer=None,
                citations=citations,
                context_sources=citations,
                rag_enabled=False,
                generation_status="unavailable",
                generation_message=str(error) or self.unavailable_message,
                understanding=understanding, answer_task=answer_task,
                context_metadata=context_package.get("metadata"),
                documents=self._document_payloads_for_plan(
                    retrieval.get("candidate_results", results), understanding, sources
                ),
            ), None
        generation_ms = (perf_counter() - generation_start) * 1000

        answer = self.citation_service.remove_unknown_citations(answer, sources)
        citations = self.citation_service.build(sources)
        return self._response(
            question=question,
            retrieval=retrieval,
            answer=answer,
            citations=citations,
            context_sources=citations,
            rag_enabled=True,
            generation_status="generated",
            understanding=understanding, answer_task=answer_task,
            context_metadata=context_package.get("metadata"),
            unique_results=self._unique_result_payloads(results, sources),
            documents=self._document_payloads_for_plan(
                retrieval.get("candidate_results", results), understanding, sources
            ),
        ), generation_ms
