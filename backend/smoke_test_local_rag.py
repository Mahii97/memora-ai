import argparse
import json

from app.embeddings.embedding_service import EmbeddingService
from app.llm.llm_service import LLMService
from app.rag.rag_service import RAGService
from app.retrieval.hybrid_engine import HybridEngine
from app.vector.search import SearchEngine
from app.vector.vector_index import VectorIndex


def main():
    parser = argparse.ArgumentParser(description="Run one local MemoraAI RAG query.")
    parser.add_argument(
        "--query",
        default="computer science information technology exam at IIT Roorkee",
    )
    parser.add_argument(
        "--model",
        help="Optional installed local Ollama model name; defaults to LLM_MODEL.",
    )
    args = parser.parse_args()

    llm_service = LLMService(model_name=args.model) if args.model else LLMService()
    if not llm_service.model_name:
        parser.error("Set LLM_MODEL in backend/.env or pass --model with an installed local model.")
    if not llm_service.is_available():
        raise SystemExit(
            f"Local model {llm_service.model_name!r} is unavailable. Start Ollama and verify `ollama list`."
        )

    index = VectorIndex()
    index.load("data/vector_index.pkl")
    retrieval_engine = HybridEngine(SearchEngine(index))
    service = RAGService(
        retrieval_engine,
        embedding_service=EmbeddingService(),
        llm_service=llm_service,
    )
    trace = service.ask_with_trace(args.query)
    response = trace["response"]

    report = {
        "question": args.query,
        "llm_provider": response.llm_provider,
        "llm_model": response.llm_model,
        "generation_status": response.generation_status,
        "hybrid_top_five": trace["retrieval"]["hybrid_order"][:5],
        "reranked_results": [
            {
                "final_rank": result["final_rank"],
                "source_name": result["source_name"],
                "location": result["location"],
                "hybrid_score": result["hybrid_score"],
                "reranker_score": result["reranker_score"],
            }
            for result in response.results
        ],
        "context_supplied_to_llm": trace["context"],
        "answer": response.answer,
        "citations": [citation.model_dump() for citation in response.citations],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
