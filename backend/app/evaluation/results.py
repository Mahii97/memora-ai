import csv
import json
from datetime import datetime, timezone
from pathlib import Path


def create_run_directory(output_root: str | Path) -> tuple[str, Path]:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = timestamp
    run_dir = root / f"run_{run_id}"
    suffix = 1
    while run_dir.exists():
        run_id = f"{timestamp}_{suffix}"
        run_dir = root / f"run_{run_id}"
        suffix += 1
    run_dir.mkdir()
    return run_id, run_dir


def write_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_per_query_csv(path: Path, retrieval: dict, rag: dict | None) -> None:
    rag_rows = {row["query_id"]: row for row in (rag or {}).get("per_query", [])}
    fieldnames = [
        "query_id",
        "query",
        "relevant_chunk_ids",
        "rankings",
        "metrics",
        "timings_ms",
        "rag_status",
        "citation_count",
        "citation_validity_rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in retrieval["per_query"]:
            rag_row = rag_rows.get(row["query_id"], {})
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "query": row["query"],
                    "relevant_chunk_ids": json.dumps(row["relevant_chunk_ids"]),
                    "rankings": json.dumps(row["rankings"], ensure_ascii=False),
                    "metrics": json.dumps(row["metrics"], ensure_ascii=False),
                    "timings_ms": json.dumps(row["timings_ms"]),
                    "rag_status": rag_row.get("generation_status", "not_run"),
                    "citation_count": rag_row.get("citation_count"),
                    "citation_validity_rate": rag_row.get("citation_validity_rate"),
                }
            )


def save_run(output_root: str | Path, metadata: dict, retrieval: dict, rag: dict | None):
    run_id, run_dir = create_run_directory(output_root)
    shared_metadata = {"run_id": run_id, **metadata}
    retrieval_output = {"metadata": shared_metadata, **retrieval}
    latency_output = {
        "metadata": shared_metadata,
        "retrieval_latency": retrieval["latency"],
        "rag_latency": (rag or {}).get("latency"),
        "rag_latency_status": (rag or {}).get("status", "not_run"),
    }
    rag_output = {
        "metadata": shared_metadata,
        **(rag or {"status": "not_run", "reason": "RAG evaluation not requested."}),
    }
    write_json(run_dir / "retrieval_results.json", retrieval_output)
    write_json(run_dir / "latency_results.json", latency_output)
    write_json(run_dir / "rag_results.json", rag_output)
    write_per_query_csv(run_dir / "per_query_results.csv", retrieval, rag)

    with (run_dir / "retrieval_results.csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["method", "metric", "value"])
        if retrieval["metrics"] is not None:
            for method, metrics in retrieval["metrics"].items():
                for metric, value in metrics.items():
                    writer.writerow([method, metric, value])
    return run_id, run_dir
