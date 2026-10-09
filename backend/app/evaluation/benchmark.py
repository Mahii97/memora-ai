import argparse
import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from app.embeddings.embedding_service import EmbeddingService
from app.evaluation.rag_evaluator import RAGEvaluator
from app.evaluation.results import save_run
from app.evaluation.retrieval_evaluator import RetrievalEvaluator
from app.evaluation.schemas import load_evaluation_dataset
from app.llm.llm_service import LLMService
from app.rag.rag_service import RAGService
from app.retrieval.hybrid_engine import HybridEngine
from app.vector.search import SearchEngine
from app.vector.vector_index import VectorIndex


BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_ROOT / ".env")
DEFAULT_KS = (1, 3, 5, 10)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def database_chunk_count() -> int | None:
    database_path = BACKEND_ROOT / "data" / "memora.db"
    if not database_path.exists():
        return None
    connection = sqlite3.connect(f"file:{database_path.as_posix()}?mode=ro", uri=True)
    try:
        return connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    except sqlite3.Error:
        return None
    finally:
        connection.close()


def build_runtime():
    index = VectorIndex()
    index.load(BACKEND_ROOT / "data" / "vector_index.pkl")
    embedding_service = EmbeddingService()
    hybrid_engine = HybridEngine(SearchEngine(index))
    llm_service = LLMService()
    rag_service = RAGService(
        hybrid_engine,
        embedding_service=embedding_service,
        llm_service=llm_service,
    )
    return index, embedding_service, hybrid_engine, llm_service, rag_service


def print_latency(label: str, summary: dict | None) -> None:
    if not summary:
        print(f"{label}: not measured")
        return
    print(
        f"{label}: mean={summary['mean_ms']:.2f} ms, "
        f"median={summary['median_ms']:.2f} ms, "
        f"min={summary['min_ms']:.2f} ms, max={summary['max_ms']:.2f} ms, "
        f"p95={summary['p95_ms']:.2f} ms"
    )


def run_benchmark(
    dataset_path: str | Path,
    output_root: str | Path,
    ks=DEFAULT_KS,
    repeats: int = 3,
    run_rag: bool = False,
) -> Path:
    dataset_file = Path(dataset_path)
    if not dataset_file.is_absolute():
        dataset_file = BACKEND_ROOT / dataset_file
    records = load_evaluation_dataset(dataset_file)
    if not records:
        raise ValueError("Evaluation dataset contains no queries")

    print("Loading evaluation dataset...")
    labeled_count = sum(record.has_relevance_labels for record in records)
    print(f"Queries: {len(records)} (labeled: {labeled_count})")

    index, embedding_service, hybrid_engine, llm_service, rag_service = build_runtime()
    print(f"Indexed chunks: {len(index.metadata)}")
    sqlite_chunk_rows = database_chunk_count()
    print(f"SQLite chunk rows: {sqlite_chunk_rows if sqlite_chunk_rows is not None else 'unavailable'}")
    print("Evaluating BM25, Vector, Hybrid, and Hybrid + CrossEncoder...")
    retrieval = RetrievalEvaluator(embedding_service, hybrid_engine).run(
        records, ks=ks, repeats=repeats
    )

    rag = None
    if run_rag:
        print("Evaluating local RAG...")
        rag = RAGEvaluator(rag_service, index.metadata).run(records)
        if rag["status"] == "not_measured":
            print(f"RAG generation: not measured ({rag['reason']})")

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_dataset_path": str(dataset_file),
        "evaluation_dataset_sha256": sha256_file(dataset_file),
        "evaluation_dataset_version": dataset_file.name,
        "number_of_queries": len(records),
        "labeled_query_count": labeled_count,
        "relevance_judgment_count": retrieval["relevance_judgment_count"],
        "corpus_size": len(index.metadata),
        "sqlite_chunk_rows": sqlite_chunk_rows,
        "indexed_chunk_ids_sha256": hashlib.sha256(
            "\n".join(str(item["chunk_id"]) for item in index.metadata).encode("utf-8")
        ).hexdigest(),
        "embedding_model": embedding_service.model_name,
        "reranker_model": hybrid_engine.reranker.model_name,
        "llm_provider": llm_service.provider,
        "llm_model": llm_service.model_name or None,
        "retrieval_configuration": {
            "vector_weight": hybrid_engine.ranker.VECTOR_WEIGHT,
            "bm25_weight": hybrid_engine.ranker.BM25_WEIGHT,
            "candidate_k": hybrid_engine.candidate_k,
            "production_final_k": hybrid_engine.top_k,
            "evaluation_k": list(ks),
            "latency_repeats": repeats,
        },
    }
    run_id, run_dir = save_run(output_root, metadata, retrieval, rag)

    if retrieval["metrics"] is None:
        print(
            "Retrieval quality metrics: not measured. "
            "Human-verified relevance labels are required."
        )
    else:
        for method, metrics in retrieval["metrics"].items():
            print(
                f"{method}: P@5={metrics['precision@5']:.4f}, "
                f"Recall@5={metrics['recall@5']:.4f}, "
                f"MRR={metrics['mrr']:.4f}, "
                f"NDCG@5={metrics['ndcg@5']:.4f}, "
                f"Hit@5={metrics['hit_rate@5']:.4f}"
            )

    print("Latency measurements:")
    for name, summary in retrieval["latency"].items():
        print_latency(name, summary)
    print(f"Results saved to: {run_dir}")
    print(f"Run ID: {run_id}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Benchmark MemoraAI retrieval and RAG.")
    parser.add_argument("--dataset", required=True, help="JSONL query/relevance dataset")
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_KS))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--run-rag", action="store_true")
    parser.add_argument("--output", default="evaluation_results")
    args = parser.parse_args()

    run_benchmark(
        args.dataset,
        args.output,
        ks=args.k,
        repeats=args.repeats,
        run_rag=args.run_rag,
    )


if __name__ == "__main__":
    main()
