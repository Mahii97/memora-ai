import argparse
import csv
import json
from pathlib import Path

from app.embeddings.embedding_service import EmbeddingService
from app.retrieval.hybrid_engine import HybridEngine
from app.vector.search import SearchEngine
from app.vector.vector_index import VectorIndex


BACKEND = Path(__file__).resolve().parent


def load_query_drafts(path: str | Path) -> list[dict]:
    queries = []
    seen_ids = set()
    with Path(path).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            query_id = record.get("query_id")
            query = record.get("query")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(f"query_id is required at line {line_number}")
            if query_id in seen_ids:
                raise ValueError(f"duplicate query_id: {query_id}")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"query must be nonempty at line {line_number}")
            if "relevant_chunk_ids" in record or "relevant" in record:
                raise ValueError("query drafts must not contain relevance labels")
            seen_ids.add(query_id)
            queries.append(record)
    if not queries:
        raise ValueError("query draft file has no queries")
    return queries


def export_candidates(
    queries: list[dict],
    output_path: str | Path,
    candidate_k: int = 50,
    queries_output: str | Path | None = None,
    force: bool = False,
):
    index = VectorIndex()
    index.load(BACKEND / "data" / "vector_index.pkl")
    if candidate_k <= 0:
        raise ValueError("candidate_k must be positive")
    active_ids = {str(chunk["chunk_id"]) for chunk in index.metadata}
    engine = HybridEngine(SearchEngine(index), candidate_k=candidate_k)
    embeddings = EmbeddingService()
    rows = []
    query_exports = []

    for query_record in queries:
        query_id = query_record["query_id"]
        query = query_record["query"]
        query_vector = embeddings.embed_query(query)
        trace = engine.retrieve_candidates(query, query_vector)
        ranked_candidates = trace["candidates"]
        candidate_ids = []
        candidate_details = []

        for candidate_rank, chunk in enumerate(ranked_candidates, start=1):
            chunk_id = str(chunk["chunk_id"])
            if chunk_id not in active_ids:
                raise ValueError(f"Exporter returned a chunk outside the active index: {chunk_id}")
            candidate_ids.append(chunk_id)
            candidate_details.append(
                {
                    "candidate_rank": candidate_rank,
                    "chunk_id": chunk_id,
                    "source_name": chunk.get("source_name", ""),
                    "source_type": chunk.get("source_type", ""),
                    "location": chunk.get("location", ""),
                    "author": chunk.get("author", ""),
                    "date": chunk.get("created_at", ""),
                    "text_preview": chunk.get("text", "")[:500],
                }
            )
            rows.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "candidate_rank": candidate_rank,
                    "chunk_id": chunk_id,
                    "source_name": chunk.get("source_name", ""),
                    "source_type": chunk.get("source_type", ""),
                    "location": chunk.get("location", ""),
                    "author": chunk.get("author", ""),
                    "date": chunk.get("created_at", ""),
                    "chunk_text": chunk.get("text", ""),
                    "relevant": "",
                }
            )
        query_exports.append(
            {
                "query_id": query_id,
                "query": query,
                "query_type": query_record.get("query_type", ""),
                "source_ref": query_record.get("source_ref", ""),
                "candidate_chunk_ids": candidate_ids,
                "candidates": candidate_details,
            }
        )

    review_path = Path(output_path)
    if not review_path.is_absolute():
        review_path = BACKEND / review_path
    query_path = Path(queries_output) if queries_output is not None else None
    if query_path is not None and not query_path.is_absolute():
        query_path = BACKEND / query_path
    destinations = [review_path] + ([query_path] if query_path is not None else [])
    existing = [str(path) for path in destinations if path.exists()]
    if existing and not force:
        raise FileExistsError(
            "Refusing to overwrite existing review files: " + ", ".join(existing)
        )

    review_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id", "query", "query_type", "candidate_rank", "chunk_id",
        "source_name", "source_type", "location", "author", "date",
        "chunk_text", "relevant",
    ]
    with review_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if query_path is not None:
        query_path.parent.mkdir(parents=True, exist_ok=True)
        with query_path.open("w", encoding="utf-8", newline="\n") as output:
            for record in query_exports:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return review_path, len(rows), len(active_ids), query_exports


def main():
    parser = argparse.ArgumentParser(
        description="Export active-index candidates for human relevance labeling."
    )
    parser.add_argument("--queries-file", help="JSONL query drafts with query_id/query fields")
    parser.add_argument("--query", action="append", help="Repeat for user-supplied query text")
    parser.add_argument(
        "--output",
        default="data/evaluation/human_review.csv",
    )
    parser.add_argument(
        "--queries-output",
        default="data/evaluation/queries_to_label.jsonl",
    )
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--force", action="store_true", help="Explicitly overwrite existing review files")
    args = parser.parse_args()
    if args.queries_file:
        query_records = load_query_drafts(args.queries_file)
    elif args.query:
        query_records = [
            {"query_id": f"q{number:03d}", "query": query}
            for number, query in enumerate(args.query, start=1)
        ]
    else:
        parser.error("provide --queries-file or one or more --query values")
    output, row_count, active_count, query_exports = export_candidates(
        query_records,
        args.output,
        args.candidate_k,
        queries_output=args.queries_output,
        force=args.force,
    )
    print(f"Queries: {len(query_exports)}")
    print(f"Candidate rows: {row_count}")
    print(f"Active indexed chunks: {active_count}")
    print(f"Human review CSV: {output}")
    print(f"Candidate query JSONL: {BACKEND / args.queries_output}")
    print("Every relevance cell is blank; no relevance labels were generated.")


if __name__ == "__main__":
    main()
