from collections import defaultdict
from time import perf_counter
from collections.abc import Sequence

from app.evaluation.latency import LatencyCollector
from app.evaluation.metrics import aggregate_query_metrics, query_metrics
from app.evaluation.schemas import EvaluationQuery, resolve_relevant_chunk_ids


METHODS = ("bm25", "vector", "hybrid", "hybrid_reranker")


class RetrievalEvaluator:
    def __init__(self, embedding_service, hybrid_engine):
        self.embedding_service = embedding_service
        self.hybrid_engine = hybrid_engine
        self.corpus_metadata = list(hybrid_engine.vector.index.metadata)

    def warm_up(self) -> None:
        vector = self.embedding_service.embed_query("evaluation warm-up query")
        self.hybrid_engine.vector.search(vector, k=1)
        self.hybrid_engine.bm25.search(
            self.hybrid_engine.tokenizer.tokenize("evaluation warm-up query"), k=1
        )
        self.hybrid_engine.reranker.model
        self.hybrid_engine.reranker.rerank(
            "evaluation warm-up query", ["evaluation warm-up passage"], top_k=1
        )

    def run(
        self,
        queries: Sequence[EvaluationQuery],
        ks: Sequence[int] = (1, 3, 5, 10),
        repeats: int = 1,
        warm_up: bool = True,
    ) -> dict:
        if repeats <= 0:
            raise ValueError("repeats must be positive")
        if not queries:
            raise ValueError("at least one evaluation query is required")
        max_k = max(ks)
        if max_k <= 0:
            raise ValueError("k values must be positive")
        if warm_up:
            self.warm_up()

        latency = LatencyCollector()
        per_query = []
        per_method_metrics = defaultdict(list)
        relevance_judgment_count = 0
        labeled_query_count = 0
        corpus_ids = {str(item["chunk_id"]) for item in self.corpus_metadata}

        for query_record in queries:
            query_embedding_start = perf_counter()
            query_vector = self.embedding_service.embed_query(query_record.query)
            query_embedding_ms = (perf_counter() - query_embedding_start) * 1000
            latency.add("query_embedding", query_embedding_ms)

            bm25_results = []
            vector_results = []
            hybrid_results = []
            reranked_results = []
            query_timings = defaultdict(list)

            for _ in range(repeats):
                bm25_start = perf_counter()
                query_tokens = self.hybrid_engine.tokenizer.tokenize(query_record.query)
                bm25_results = self.hybrid_engine.bm25.search(query_tokens, k=max_k)
                query_timings["bm25"].append((perf_counter() - bm25_start) * 1000)

                vector_start = perf_counter()
                vector_results = self.hybrid_engine.vector.search(query_vector, k=max_k)
                query_timings["vector"].append((perf_counter() - vector_start) * 1000)

                total_retrieval_start = perf_counter()
                hybrid_start = perf_counter()
                candidate_trace = self.hybrid_engine.retrieve_candidates(
                    query_record.query, query_vector
                )
                query_timings["hybrid"].append((perf_counter() - hybrid_start) * 1000)

                rerank_start = perf_counter()
                reranked_results = self.hybrid_engine.reranker.rerank(
                    query_record.query,
                    candidate_trace["candidates"],
                    top_k=max_k,
                )
                query_timings["reranker"].append((perf_counter() - rerank_start) * 1000)
                query_timings["total_retrieval"].append(
                    (perf_counter() - total_retrieval_start) * 1000
                )

            for method, values in query_timings.items():
                latency.extend(method, values)

            rankings = {
                "bm25": [str(item["chunk"]["chunk_id"]) for item in bm25_results],
                "vector": [str(item["chunk"]["chunk_id"]) for item in vector_results],
                "hybrid": [str(item["chunk_id"]) for item in candidate_trace["candidates"]],
                "hybrid_reranker": [str(item["chunk_id"]) for item in reranked_results],
            }
            resolved_relevant = resolve_relevant_chunk_ids(query_record, self.corpus_metadata)
            query_result = {
                "query_id": query_record.query_id,
                "query": query_record.query,
                "rankings": rankings,
                "timings_ms": {
                    name: sum(values) / len(values) for name, values in query_timings.items()
                },
                "relevant_chunk_ids": (
                    sorted(resolved_relevant) if resolved_relevant is not None else None
                ),
                "metrics": None,
            }
            if resolved_relevant is not None:
                unknown = resolved_relevant - corpus_ids
                if unknown:
                    raise ValueError(
                        f"Unknown relevance judgments for {query_record.query_id}: {sorted(unknown)}"
                    )
                labeled_query_count += 1
                relevance_judgment_count += len(resolved_relevant)
                for method, ranking in rankings.items():
                    method_result = query_metrics(ranking, resolved_relevant, ks)
                    per_method_metrics[method].append(method_result)
                    query_result["metrics"] = query_result["metrics"] or {}
                    query_result["metrics"][method] = method_result
            per_query.append(query_result)

        metrics = (
            {
                method: aggregate_query_metrics(per_method_metrics[method])
                for method in METHODS
            }
            if labeled_query_count
            else None
        )
        return {
            "status": "measured" if labeled_query_count else "not_measured",
            "metrics": metrics,
            "per_query": per_query,
            "latency": latency.summary(),
            "query_count": len(queries),
            "labeled_query_count": labeled_query_count,
            "relevance_judgment_count": relevance_judgment_count,
            "corpus_size": len(self.corpus_metadata),
            "methods": list(METHODS),
            "ks": list(ks),
            "latency_repeats": repeats,
            "label_note": (
                None
                if labeled_query_count == len(queries)
                else "Quality metrics include only explicitly labeled queries; unlabeled rows are excluded."
            ),
        }
