import re
import math
from collections import Counter
from collections.abc import Iterable, Sequence


def _validate_k(k: int) -> None:
    if k <= 0:
        raise ValueError("k must be positive")


def _unique_ranked(retrieved: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(retrieved))


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    _validate_k(k)
    relevant_set = set(relevant)
    ranked = _unique_ranked(retrieved)[:k]
    return sum(item in relevant_set for item in ranked) / k


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    _validate_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    ranked = _unique_ranked(retrieved)[:k]
    return sum(item in relevant_set for item in ranked) / len(relevant_set)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    for rank, item in enumerate(_unique_ranked(retrieved), start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    retrieved_lists: Sequence[Sequence[str]], relevant_sets: Sequence[Iterable[str]]
) -> float:
    if len(retrieved_lists) != len(relevant_sets):
        raise ValueError("retrieved_lists and relevant_sets must have equal length")
    if not retrieved_lists:
        return 0.0
    return sum(
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(retrieved_lists, relevant_sets)
    ) / len(retrieved_lists)


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    _validate_k(k)
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0

    ranked = _unique_ranked(retrieved)[:k]
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(ranked, start=1)
        if item in relevant_set
    )
    ideal_relevant_count = min(len(relevant_set), k)
    ideal_dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def hit_rate_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    _validate_k(k)
    relevant_set = set(relevant)
    return float(any(item in relevant_set for item in _unique_ranked(retrieved)[:k]))


def query_metrics(
    retrieved: Sequence[str], relevant: Iterable[str], ks: Sequence[int] = (1, 3, 5, 10)
) -> dict[str, float]:
    relevant_set = set(relevant)
    metrics = {}
    for k in ks:
        metrics[f"precision@{k}"] = precision_at_k(retrieved, relevant_set, k)
        metrics[f"recall@{k}"] = recall_at_k(retrieved, relevant_set, k)
        metrics[f"hit_rate@{k}"] = hit_rate_at_k(retrieved, relevant_set, k)
    metrics["mrr"] = reciprocal_rank(retrieved, relevant_set)
    metrics["ndcg@5"] = ndcg_at_k(retrieved, relevant_set, 5)
    metrics["ndcg@10"] = ndcg_at_k(retrieved, relevant_set, 10)
    return metrics


def aggregate_query_metrics(per_query: Sequence[dict[str, float]]) -> dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0]
    return {
        key: sum(result[key] for result in per_query) / len(per_query)
        for key in keys
    }


def normalize_answer(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted_tokens = normalize_answer(prediction)
    reference_tokens = normalize_answer(reference)
    if not predicted_tokens or not reference_tokens:
        return float(predicted_tokens == reference_tokens)
    overlap = sum((Counter(predicted_tokens) & Counter(reference_tokens)).values())
    if not overlap:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(reference_tokens)
    return 2 * precision * recall / (precision + recall)
