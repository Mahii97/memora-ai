import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from app.retrieval.reranker import RerankerService


BACKEND = Path(__file__).resolve().parent
REQUIRED_COLUMNS = {
    "query_id",
    "query",
    "candidate_rank",
    "chunk_id",
    "source_name",
    "source_type",
    "location",
    "author",
    "date",
    "chunk_text",
    "relevant",
}


def suggest_labels(input_path: str | Path, output_path: str | Path, summary_path: str | Path):
    input_path = Path(input_path)
    output_path = Path(output_path)
    summary_path = Path(summary_path)
    if not input_path.is_absolute():
        input_path = BACKEND / input_path
    if not output_path.is_absolute():
        output_path = BACKEND / output_path
    if not summary_path.is_absolute():
        summary_path = BACKEND / summary_path

    with input_path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames or []
        missing = REQUIRED_COLUMNS - set(columns)
        if missing:
            raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("Input review CSV contains no candidate rows")
    if any(row["relevant"].strip() for row in rows):
        raise ValueError("Expected the original relevant column to be blank; refusing to overwrite labels")

    model = RerankerService().model
    scores = model.predict(
        [[row["query"], row["chunk_text"]] for row in rows],
        batch_size=16,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    counts = Counter()
    per_query = defaultdict(Counter)
    for row, score in zip(rows, scores):
        score = float(score)
        if score >= 2.0:
            label = "1"
            reason = (
                f"Local CrossEncoder score {score:.2f} is in the strong-match suggestion band; "
                "check the query and chunk text before accepting."
            )
        elif score <= -2.0:
            label = "0"
            reason = (
                f"Local CrossEncoder score {score:.2f} is in the low-match suggestion band; "
                "check the query and chunk text before accepting."
            )
        else:
            label = "REVIEW"
            reason = (
                f"Local CrossEncoder score {score:.2f} is in the uncertain band; "
                "manual relevance judgment required."
            )
        row["suggested_relevant"] = label
        row["reason"] = reason
        counts[label] += 1
        per_query[row["query_id"]][label] += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(columns) + ["suggested_relevant", "reason"]
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Human Review Suggestion Summary",
        "",
        "These are unverified machine suggestions from the already-installed local CrossEncoder, using only each CSV query and candidate chunk text. They are not human-verified relevance labels and must be reviewed before use.",
        "",
        f"- Total queries: {len(per_query)}",
        f"- Total candidate rows: {len(rows)}",
        f"- Suggested relevant (`1`): {counts['1']}",
        f"- Suggested irrelevant (`0`): {counts['0']}",
        f"- Ambiguous/manual review (`REVIEW`): {counts['REVIEW']}",
        "- Labels are suggestions, not calibrated probabilities; thresholds are broad review heuristics.",
        "- The original `relevant` column is preserved blank for every row.",
        "",
        "## Suggestions Per Query",
        "",
        "| Query ID | Rows | Suggested 1 | Suggested 0 | REVIEW |",
        "|---|---:|---:|---:|---:|",
    ]
    for query_id in sorted(per_query):
        query_counts = per_query[query_id]
        lines.append(
            f"| {query_id} | {sum(query_counts.values())} | {query_counts['1']} | "
            f"{query_counts['0']} | {query_counts['REVIEW']} |"
        )
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "queries": len(per_query),
        "candidate_rows": len(rows),
        "suggested_relevant": counts["1"],
        "suggested_irrelevant": counts["0"],
        "review": counts["REVIEW"],
        "output_path": output_path,
        "summary_path": summary_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create unverified, local CrossEncoder relevance suggestions for human review."
    )
    parser.add_argument("--input", default="data/evaluation/human_review.csv")
    parser.add_argument(
        "--output", default="data/evaluation/human_review_suggestions.csv"
    )
    parser.add_argument(
        "--summary", default="data/evaluation/HUMAN_REVIEW_SUMMARY.md"
    )
    args = parser.parse_args()
    summary = suggest_labels(args.input, args.output, args.summary)
    print(f"Queries: {summary['queries']}")
    print(f"Candidate rows: {summary['candidate_rows']}")
    print(f"Suggested relevant: {summary['suggested_relevant']}")
    print(f"Suggested irrelevant: {summary['suggested_irrelevant']}")
    print(f"Review: {summary['review']}")
    print(f"Suggestion CSV: {summary['output_path']}")
    print(f"Summary: {summary['summary_path']}")
    print("Suggestions are not human-verified; original labels remain blank.")


if __name__ == "__main__":
    main()
