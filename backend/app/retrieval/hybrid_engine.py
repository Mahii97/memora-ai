import os
from time import perf_counter
import re

from dotenv import load_dotenv
from pathlib import Path

from app.timeline.timeline_search import TimelineSearch
from app.retrieval.query_analyzer import QueryAnalyzer
from app.retrieval.hybrid_ranker import HybridRanker
from app.retrieval.reranker import RerankerService
from app.retrieval.bm25 import BM25Index
from app.tokenizer.tokenizer import Tokenizer
from app.rag.query_understanding import QueryUnderstanding, STOP_WORDS, FILLER_WORDS, GENERIC_WORDS


BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_ROOT / ".env")


class HybridEngine:

    def __init__(
        self,
        search_engine,
        reranker=None,
        top_k=None,
        candidate_k=None,
        timeline=None,
        bm25_index_path="data/bm25_index.json",
    ):

        self.vector = search_engine
        self.timeline = timeline or TimelineSearch()
        self.analyzer = QueryAnalyzer()
        self.query_understanding = QueryUnderstanding()
        self.ranker = HybridRanker()
        self.reranker = reranker or RerankerService()
        self.tokenizer = Tokenizer()
        self.top_k = top_k or int(os.getenv("RERANKER_TOP_K", "5"))
        self.candidate_k = candidate_k or int(
            os.getenv("RERANKER_CANDIDATE_K", "20")
        )
        if self.top_k <= 0 or self.candidate_k <= 0:
            raise ValueError("reranker top-k settings must be positive")

        self.bm25 = BM25Index()
        self.bm25.load(bm25_index_path)

    def search(self, query, query_vector):
        return self.search_with_trace(query, query_vector)["results"]

    def search_with_trace(
        self, query, query_vector, retrieval_query=None, query_plan=None
    ):
        query_plan = query_plan or self.query_understanding.analyze(query)
        search_query = retrieval_query or query_plan.get("retrieval_query") or query
        retrieval = self.retrieve_candidates(search_query, query_vector, query_plan)
        if query_plan.get("intent") == "temporal_summary" and query_plan.get("time_range"):
            temporal = self._temporal_candidates(query_plan["time_range"])
            retrieval["temporal_candidate_count"] = len(temporal)
            # Temporal questions must use records from the requested time range.
            retrieval["candidates"] = temporal[:self.candidate_k]
            retrieval["candidate_count"] = len(retrieval["candidates"])
            retrieval["hybrid_candidates_before_temporal_filter"] = len(
                retrieval.get("hybrid_order", [])
            )
            retrieval["hybrid_order"] = [
                {"chunk_id": item["chunk_id"], "source_name": item["source_name"],
                 "location": item["location"], "hybrid_score": item["hybrid_score"]}
                for item in retrieval["candidates"]
            ]
        reranker_start = perf_counter()
        # Rerank every candidate before applying question-specific priorities so
        # an exact question-number/source match cannot be cut off prematurely.
        ranked_results = self.reranker.rerank(
            search_query, retrieval["candidates"], top_k=len(retrieval["candidates"])
        )
        ranked_results = self._apply_query_priorities(ranked_results, query_plan)
        for rank, item in enumerate(ranked_results, start=1):
            item["final_rank"] = rank
            if isinstance(item.get("explanation"), dict):
                item["explanation"]["final_rank"] = rank
        results = ranked_results[:self.top_k]
        reranker_latency_ms = (perf_counter() - reranker_start) * 1000
        return {
            **retrieval,
            "results": results,
            "candidate_results": ranked_results,
            "reranker_latency_ms": reranker_latency_ms,
            "total_retrieval_latency_ms": (
                retrieval["hybrid_latency_ms"] + reranker_latency_ms
            ),
        }

    @staticmethod
    def _deduplicate_candidates(candidates):
        unique = []
        seen_ids = set()
        seen_text_locations = set()
        for chunk in candidates:
            chunk_id = str(chunk.get("chunk_id") or "")
            text = re.sub(r"\s+", " ", str(chunk.get("text") or "").casefold()).strip()
            text_location = (
                str(chunk.get("source_name") or "").casefold(),
                re.sub(r"\s*\((?:ocr|text)\)$", "", str(chunk.get("location") or "").casefold().strip()),
                text,
            )
            if (chunk_id and chunk_id in seen_ids) or (text and text_location in seen_text_locations):
                continue
            if chunk_id:
                seen_ids.add(chunk_id)
            if text:
                seen_text_locations.add(text_location)
            unique.append(chunk)
        return unique

    def _temporal_candidates(self, time_range):
        date_from = time_range["start"]
        date_to = time_range["end"]
        rows = self.timeline.by_range(date_from, date_to)
        candidates = []
        for item in rows:
            candidate = dict(item)
            candidate.update({
                "vector_score": candidate.get("vector_score", 0.0),
                "bm25_score": candidate.get("bm25_score", 0.0),
                "hybrid_score": candidate.get("hybrid_score", 0.0),
                "reranker_score": candidate.get("reranker_score", 0.0),
                "explanation": candidate.get("explanation", {}),
                "final_rank": candidate.get("final_rank", 0),
            })
            candidates.append(candidate)
        return candidates[:self.candidate_k]

    def _apply_query_priorities(self, results, query_plan):
        decorated = []
        for ordinal, item in enumerate(results):
            features = self.query_understanding.match_features(query_plan, item)
            item["query_match"] = features
            question_match = int(features["question_number_match"])
            source_match = (
                features["source_term_matches"]
                if query_plan.get("intent") in {"document_search", "document_summary"}
                or any(entity in query_plan.get("entities", []) for entity in ("GATE", "Data Science"))
                else 0
            )
            decorated.append((
                -question_match,
                -source_match,
                -float(item.get("reranker_score", float("-inf"))),
                ordinal,
                item,
            ))
        decorated.sort(key=lambda row: row[:-1])
        return [row[-1] for row in decorated]

    def retrieve_candidates(self, query, query_vector, query_plan=None):
        hybrid_start = perf_counter()

        # -----------------------------
        # 1. Semantic Retrieval
        # -----------------------------
        vector_start = perf_counter()
        vector_results = self.vector.search(query_vector, k=self.candidate_k)
        vector_latency_ms = (perf_counter() - vector_start) * 1000

        # -----------------------------
        # 2. BM25 Retrieval
        # -----------------------------
        ignored_terms = STOP_WORDS | FILLER_WORDS | GENERIC_WORDS
        query_tokens = list(dict.fromkeys(
            token for token in self.tokenizer.tokenize(query) if token not in ignored_terms
        ))
        bm25_start = perf_counter()
        bm25_results = self.bm25.search(query_tokens, k=self.candidate_k)
        bm25_latency_ms = (perf_counter() - bm25_start) * 1000

        # -----------------------------
        # 3. Hybrid Score Fusion
        # -----------------------------
        fused_results = self.ranker.fuse(
            vector_results,
            bm25_results
        )
        # Vector and lexical retrieval can return the same indexed chunk. Keep
        # its best fused entry once so reranking and UI results are not padded
        # with duplicate copies of one memory.
        fused_results = self._deduplicate_candidates(fused_results)

        # -----------------------------
        # 4. Metadata Extraction
        # -----------------------------
        metadata = self.analyzer.analyze(query)

        filtered = []

        for chunk in fused_results:

            # Source type filter
            if metadata["source_type"]:
                if chunk["source_type"] != metadata["source_type"]:
                    continue

            # Date filter
            if metadata["date"]:
                if not str(chunk["created_at"]).startswith(metadata["date"]):
                    continue

            # Author filter
            if metadata["author"]:
                if metadata["author"].lower() not in chunk["author"].lower():
                    continue

            filtered.append(chunk)

        # If metadata matched, return filtered results
        if filtered:
            fused_results = filtered

        candidates = fused_results[:self.candidate_k]
        hybrid_order = [
            {
                "chunk_id": chunk["chunk_id"],
                "source_name": chunk["source_name"],
                "location": chunk["location"],
                "hybrid_score": chunk["hybrid_score"],
            }
            for chunk in candidates
        ]

        return {
            "vector_candidate_count": len(vector_results),
            "bm25_candidate_count": len(bm25_results),
            "candidate_count": len(candidates),
            "hybrid_order": hybrid_order,
            "candidates": candidates,
            "vector_latency_ms": vector_latency_ms,
            "bm25_latency_ms": bm25_latency_ms,
            "hybrid_latency_ms": (perf_counter() - hybrid_start) * 1000,
        }
