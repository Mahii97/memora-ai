import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

from app.vector.vector_index import VectorIndex


POSITIVE = {"1", "true", "yes", "y", "relevant"}
NEGATIVE = {"0", "false", "no", "n", "not relevant"}
BACKEND = Path(__file__).resolve().parent


def load_active_chunk_ids(index_path: str | Path = "data/vector_index.pkl") -> set[str]:
    path = Path(index_path)
    if not path.is_absolute():
        path = BACKEND / path
    index = VectorIndex()
    index.load(path)
    return {str(item["chunk_id"]) for item in index.metadata}


def compile_labels(
    input_path: str | Path,
    output_path: str | Path,
    active_chunk_ids: set[str] | None = None,
    force: bool = False,
) -> dict:
    if active_chunk_ids is None:
        active_chunk_ids = load_active_chunk_ids()
    grouped = OrderedDict()
    invalid_chunk_ids = set()
    with Path(input_path).open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            query_id = row.get("query_id", "").strip()
            query = row.get("query", "").strip()
            chunk_id = row.get("chunk_id", "").strip()
            label = row.get("relevant", "").strip().casefold()
            if not query_id or not query or not chunk_id:
                raise ValueError("Each candidate row must include query_id, query, and chunk_id")
            if chunk_id not in active_chunk_ids:
                invalid_chunk_ids.add(chunk_id)
                continue
            if label not in POSITIVE | NEGATIVE:
                raise ValueError(
                    f"Every candidate row needs a yes/no relevance label: {query_id}/{chunk_id}"
                )
            group = grouped.setdefault(
                query_id,
                {"query": query, "relevant_chunk_ids": [], "notes": []},
            )
            if group["query"] != query:
                raise ValueError(f"Inconsistent query text for {query_id}")
            if label in POSITIVE:
                group["relevant_chunk_ids"].append(chunk_id)
            if row.get("notes", "").strip():
                group["notes"].append(row["notes"].strip())

    if invalid_chunk_ids:
        raise ValueError(
            "Invalid chunk IDs outside the active index: "
            + ", ".join(sorted(invalid_chunk_ids))
        )
    if not grouped:
        raise ValueError("No labeled candidate rows were found")
    output_path = Path(output_path)
    if not output_path.is_absolute():
        output_path = BACKEND / output_path
    if output_path.exists() and not force:
        raise FileExistsError(
            f"Refusing to overwrite {output_path}; pass --force to replace it."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    unique_relevant = set()
    labeled_queries = 0
    zero_relevant_queries = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as output:
        for query_id, group in grouped.items():
            relevant_chunk_ids = list(dict.fromkeys(group["relevant_chunk_ids"]))
            if relevant_chunk_ids:
                labeled_queries += 1
                unique_relevant.update(relevant_chunk_ids)
            else:
                zero_relevant_queries += 1
            record = {
                "query_id": query_id,
                "query": group["query"],
                "relevant_chunk_ids": relevant_chunk_ids,
            }
            if group["notes"]:
                record["notes"] = "; ".join(dict.fromkeys(group["notes"]))
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
    return {
        "queries": len(grouped),
        "queries_with_relevant": labeled_queries,
        "queries_with_zero_relevant": zero_relevant_queries,
        "unique_relevant_chunks": len(unique_relevant),
        "invalid_chunk_ids": 0,
        "output_path": output_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compile fully human-labeled candidate CSV into evaluation JSONL."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/evaluation/evaluation.jsonl")
    parser.add_argument("--active-index", default="data/vector_index.pkl")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    summary = compile_labels(
        args.input,
        args.output,
        active_chunk_ids=load_active_chunk_ids(args.active_index),
        force=args.force,
    )
    print(f"Queries: {summary['queries']}")
    print(f"Queries with at least one relevant chunk: {summary['queries_with_relevant']}")
    print(f"Queries with zero relevant candidates: {summary['queries_with_zero_relevant']}")
    print(f"Unique relevant chunks: {summary['unique_relevant_chunks']}")
    print(f"Invalid chunk IDs: {summary['invalid_chunk_ids']}")
    print(f"Compiled manually labeled queries to {summary['output_path']}")
    print("Only the judged candidate pool is covered; inspect pool coverage before research claims.")


if __name__ == "__main__":
    main()
