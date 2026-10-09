import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _bar_chart(values: dict[str, float], title: str, ylabel: str, destination: Path):
    if not values:
        return False
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(list(values), list(values.values()), color="#356859")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.set_xlabel("Retrieval method")
    axis.tick_params(axis="x", labelrotation=15)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    return True


def generate_charts(run_dir: str | Path, charts_root: str | Path | None = None) -> list[Path]:
    run_dir = Path(run_dir)
    retrieval = json.loads((run_dir / "retrieval_results.json").read_text(encoding="utf-8"))
    latency = json.loads((run_dir / "latency_results.json").read_text(encoding="utf-8"))
    destination = (
        Path(charts_root) / run_dir.name
        if charts_root is not None
        else run_dir.parent / "charts" / run_dir.name
    )
    destination.mkdir(parents=True, exist_ok=True)
    generated = []
    metrics = retrieval.get("metrics")
    chart_specs = [
        ("precision@5", "Precision@5 by Retrieval Method", "Precision@5", "precision_at_5.png"),
        ("recall@5", "Recall@5 by Retrieval Method", "Recall@5", "recall_at_5.png"),
        ("mrr", "Mean Reciprocal Rank by Retrieval Method", "MRR", "mrr_comparison.png"),
        ("ndcg@5", "NDCG@5 by Retrieval Method", "NDCG@5", "ndcg_at_5.png"),
    ]
    if metrics:
        for metric, title, ylabel, filename in chart_specs:
            values = {method: scores[metric] for method, scores in metrics.items() if metric in scores}
            if _bar_chart(values, title, ylabel, destination / filename):
                generated.append(destination / filename)

    retrieval_latency = latency.get("retrieval_latency") or {}
    latency_values = {
        method: values["mean_ms"]
        for method, values in retrieval_latency.items()
        if method in {"bm25", "vector", "hybrid", "reranker", "total_retrieval"}
    }
    if _bar_chart(
        latency_values,
        "Measured Retrieval Latency",
        "Mean latency (ms)",
        destination / "retrieval_latency.png",
    ):
        generated.append(destination / "retrieval_latency.png")

    rag_latency = latency.get("rag_latency") or {}
    rag_stages = {
        name: values["mean_ms"]
        for name, values in rag_latency.items()
        if name in {
            "retrieval_ms",
            "reranking_ms",
            "context_building_ms",
            "generation_ms",
            "total_rag_ms",
        }
    }
    if _bar_chart(
        rag_stages,
        "Measured RAG Latency Breakdown",
        "Mean latency (ms)",
        destination / "rag_latency_breakdown.png",
    ):
        generated.append(destination / "rag_latency_breakdown.png")
    return generated
