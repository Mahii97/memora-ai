from app.evaluation.latency import LatencyCollector
from app.evaluation.metrics import exact_match, token_f1
from app.evaluation.schemas import EvaluationQuery, resolve_relevant_chunk_ids
from app.llm.llm_service import LLMUnavailableError


def citation_validity(citations: list, context_sources: list) -> dict:
    context_by_id = {item.citation_id: item for item in context_sources}
    valid_count = 0
    invalid = []
    for citation in citations:
        source = context_by_id.get(citation.citation_id)
        is_valid = (
            source is not None
            and citation.chunk_id == source.chunk_id
            and citation.source_name == source.source_name
            and citation.location == source.location
        )
        if is_valid:
            valid_count += 1
        else:
            invalid.append(citation.citation_id)
    return {
        "valid_count": valid_count,
        "citation_count": len(citations),
        "validity_rate": valid_count / len(citations) if citations else None,
        "invalid_citation_ids": invalid,
    }


class RAGEvaluator:
    def __init__(self, rag_service, corpus_metadata: list[dict]):
        self.rag_service = rag_service
        self.corpus_metadata = corpus_metadata

    def run(self, queries: list[EvaluationQuery]) -> dict:
        if not queries:
            raise ValueError("at least one RAG evaluation query is required")
        llm_available = self.rag_service.llm_service.is_available()
        if not llm_available:
            return {
                "status": "not_measured",
                "reason": "No configured local LLM model is available.",
                "query_count": len(queries),
                "generation_available": False,
                "generation_latency": None,
                "retrieval_success_rate": None,
                "rag_success_rate": None,
                "generation_availability_rate": 0.0,
                "insufficient_evidence_rate": None,
                "average_citation_count": None,
                "citation_validity_rate": None,
                "citation_answer_count": 0,
                "per_query": [],
            }

        try:
            self.rag_service.llm_service.generate(
                prompt="Reply with the single word ready.",
                context="Evaluation warm-up only; this is not benchmark evidence.",
                system_prompt="Reply with the requested word only.",
            )
        except LLMUnavailableError:
            return {
                "status": "not_measured",
                "reason": "The configured local model failed its warm-up request.",
                "query_count": len(queries),
                "generation_available": False,
                "generation_latency": None,
                "retrieval_success_rate": None,
                "rag_success_rate": None,
                "generation_availability_rate": 0.0,
                "insufficient_evidence_rate": None,
                "average_citation_count": None,
                "citation_validity_rate": None,
                "citation_answer_count": 0,
                "per_query": [],
            }

        latency = LatencyCollector()
        per_query = []
        successful = 0
        insufficient = 0
        retrieval_successes = 0
        retrieval_labeled_count = 0
        generated_with_context = 0
        answerability_labeled_count = 0
        correct_abstentions = 0
        total_citations = 0
        valid_citations = 0
        citation_answer_count = 0
        exact_matches = []
        token_f1_scores = []

        for query in queries:
            trace = self.rag_service.ask_with_trace(query.query)
            response = trace["response"]
            timings = trace["timings_ms"]
            for name, value in timings.items():
                latency.add(name, value)

            validity = citation_validity(response.citations, response.context_sources)
            total_citations += validity["citation_count"]
            valid_citations += validity["valid_count"]
            if response.generation_status == "generated":
                successful += 1
                citation_answer_count += 1
                if response.rag_enabled and response.context_sources:
                    generated_with_context += 1
            elif response.generation_status == "insufficient_evidence":
                insufficient += 1

            if query.answerable is not None:
                answerability_labeled_count += 1
                expected_abstention = not query.answerable
                actual_abstention = response.generation_status == "insufficient_evidence"
                correct_abstentions += int(expected_abstention == actual_abstention)

            relevant = resolve_relevant_chunk_ids(query, self.corpus_metadata)
            if relevant:
                retrieval_labeled_count += 1
                result_ids = {str(item.get("chunk_id")) for item in response.results}
                retrieval_successes += int(bool(result_ids & relevant))

            reference_metrics = None
            if query.reference_answer is not None and response.answer is not None:
                reference_metrics = {
                    "exact_match": exact_match(response.answer, query.reference_answer),
                    "token_f1": token_f1(response.answer, query.reference_answer),
                }
                exact_matches.append(reference_metrics["exact_match"])
                token_f1_scores.append(reference_metrics["token_f1"])

            per_query.append(
                {
                    "query_id": query.query_id,
                    "generation_status": response.generation_status,
                    "rag_enabled": response.rag_enabled,
                    "retrieval_count": len(response.results),
                    "context_source_count": len(response.context_sources),
                    "citation_count": validity["citation_count"],
                    "valid_citation_count": validity["valid_count"],
                    "citation_validity_rate": validity["validity_rate"],
                    "invalid_citation_ids": validity["invalid_citation_ids"],
                    "retrieval_success": (
                        bool(
                            {str(item.get("chunk_id")) for item in response.results}
                            & relevant
                        )
                        if relevant is not None
                        else None
                    ),
                    "answer_metrics": reference_metrics,
                    "timings_ms": timings,
                }
            )

        count = len(queries)
        return {
            "status": "measured",
            "query_count": count,
            "generation_available": True,
            "generation_availability_rate": 1.0,
            "rag_success_rate": successful / count,
            "generated_with_context_rate": generated_with_context / count,
            "retrieval_success_rate": (
                retrieval_successes / retrieval_labeled_count
                if retrieval_labeled_count
                else None
            ),
            "retrieval_labeled_query_count": retrieval_labeled_count,
            "insufficient_evidence_rate": insufficient / count,
            "average_citation_count": total_citations / count,
            "citation_validity_rate": (
                valid_citations / total_citations if total_citations else None
            ),
            "citation_answer_count": citation_answer_count,
            "answerability_labeled_query_count": answerability_labeled_count,
            "abstention_accuracy": (
                correct_abstentions / answerability_labeled_count
                if answerability_labeled_count
                else None
            ),
            "latency": latency.summary(),
            "optional_reference_metrics": (
                {
                    "evaluated_answer_count": len(token_f1_scores),
                    "mean_exact_match": sum(exact_matches) / len(exact_matches),
                    "mean_token_f1": sum(token_f1_scores) / len(token_f1_scores),
                    "warning": "Text overlap metrics are not factual correctness measures.",
                }
                if token_f1_scores
                else None
            ),
            "per_query": per_query,
        }
