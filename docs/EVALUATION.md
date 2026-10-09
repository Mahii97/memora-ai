# MemoraAI Evaluation

## Purpose and current status

This evaluation layer compares the existing BM25, semantic vector, hybrid, and hybrid-plus-CrossEncoder retrieval methods without changing their ranking logic. It also records component latency and optionally measures operational RAG/citation behavior.

**Current status: no human-verified query-to-chunk relevance judgments were present in the repository.** `data/evaluation/latency_queries.jsonl` is an unlabelled latency probe only. It is not a retrieval-quality dataset. Therefore the current run can measure latency but must report retrieval-quality metrics as not measured. Any final academic benchmark requires a larger manually reviewed test set with explicit judgments.

Phase 5.1 has prepared 50 content-grounded query drafts and a 2,500-row candidate pool from the 99-chunk active index. All labels in `data/evaluation/human_review.csv` are blank. `data/evaluation/queries_to_label.jsonl` lists candidate IDs and source/location previews but deliberately contains no `relevant_chunk_ids`. The final `evaluation.jsonl` has not been created; do not run a final benchmark until human labeling and compilation are complete.

The inspected SQLite database contains 1,273 chunk rows, while the active tokenized/vector/BM25 indexes contain 99 chunks. This benchmark evaluates the 99 chunks in the active retrieval indexes and records the SQLite row count plus an indexed-chunk-ID fingerprint. Rebuild/synchronize indexes before a final benchmark if the SQLite/index mismatch is unintended.

## Retrieval interfaces reused

- BM25 only: `HybridEngine.bm25.search(tokenized_query, k)`.
- Vector only: `HybridEngine.vector.search(query_embedding, k)`.
- Hybrid before reranking: `HybridEngine.retrieve_candidates(query, query_embedding)`; this calls the existing vector search, BM25 search, fusion, and filters.
- Hybrid plus CrossEncoder: `HybridEngine.reranker.rerank(query, hybrid_candidates, top_k)`.
- RAG: `RAGService.ask_with_trace(query)`; stage timings, response, retrieval trace, and exact context are measured from the existing RAG service.

All methods run on the same query rows and same corpus/index. The evaluation code does not replace ranking behavior. The CrossEncoder evaluation ranking takes the top `max(k)` from the same hybrid candidate pool, allowing requested metrics through @10; production API output remains controlled by its configured final K.

## Evaluation dataset and human labeling

One JSON object per line is accepted:

```json
{"query_id":"q001","query":"What is computer networking?","relevant_chunk_ids":["chunk-id-a","chunk-id-b"],"notes":"Definition and explanation passages."}
{"query_id":"q002","query":"Explain the transport layer.","relevant_document":"lecture.pdf","relevant_locations":["Page 12"]}
{"query_id":"q003","query":"What is TCP for?","relevant_chunk_ids":["chunk-id-c"],"reference_answer":"TCP provides reliable, ordered delivery.","answerable":true}
```

Supported label fields are `relevant_chunk_ids`, `relevant_document`, and/or `relevant_locations`. A present empty `relevant_chunk_ids: []` is an explicit judgment of no relevant chunk; omitting all relevance fields means unjudged. Optional `reference_answer` enables token F1/exact match, and optional `answerable` supports abstention checks. Text overlap metrics are not evidence of factual correctness.

No sample relevance judgments have been created. Export candidate chunks for human inspection:

```powershell
..\.venv\Scripts\python.exe create_evaluation_dataset.py --query "What is computer networking?" --query "Explain the transport layer." --output data/evaluation/candidates_for_labeling.csv
```

The CSV includes query, chunk ID, source, location, BM25/vector ranks, and a text preview. The relevance and notes fields are blank by design. Review every candidate row in the pool as relevant/not relevant, then compile it:

```powershell
..\.venv\Scripts\python.exe compile_evaluation_labels.py --input data/evaluation/candidates_for_labeling.csv --output data/evaluation/evaluation.jsonl
```

The exported pool is the top candidate union from BM25 and vector search. Pool-based judgments cannot establish that no relevant passage exists outside the pool; report the candidate-pooling protocol and manually inspect misses. Keep training/development queries separate from a held-out test set for final academic results.

## Retrieval metrics

Metrics use binary chunk relevance and deduplicate repeated chunk IDs while preserving the first rank. Precision@K divides relevant retrieved chunks by K (even if fewer than K results exist). Recall@K divides by the number of judged relevant chunks. MRR averages the reciprocal rank of the first relevant result. NDCG@K uses binary gain and logarithmic rank discount. Hit Rate@K is one if any relevant chunk appears in the first K results, otherwise zero.

For an explicitly empty relevant set, precision, recall, MRR, NDCG, and hit rate are reported as zero; recall is mathematically undefined there, so this convention is documented rather than hidden. Unlabelled queries are excluded from quality metrics and separately counted. The output reports judged-query count and relevance-judgment count.

If `reference_answer` is supplied, optional normalized token exact match and token F1 are reported separately. They are lexical overlap indicators only, not semantic or factual correctness tests. BLEU/ROUGE are not used.

## Latency

Times are captured with `time.perf_counter()` after one warm-up pass. Model load/warm-up is excluded; query embedding, BM25, vector, hybrid retrieval, CrossEncoder reranking, total retrieval, context building, generation, and total RAG are recorded separately in milliseconds. The CLI reports mean, median, min, max, and p95 from actual samples. A missing/unavailable generation stage is stored as not measured, never as zero.

Retrieval-only latency can be run with the unlabelled probe:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe evaluate.py --dataset data/evaluation/latency_queries.jsonl --k 1 3 5 10 --repeats 5 --run-rag
```

The checked-in query is an unlabeled latency-only example. `--run-rag` records a not-measured reason if no local LLM model is configured and available. It does not install models or override `.env`.

## RAG and citation evaluation

RAG's operational success rate means the service returned a generated answer; it is not factual answer correctness. Retrieval success rate is measured only for labeled queries and means at least one relevant chunk appears in final results. Generation availability is reported separately. The evaluator reports insufficient-evidence rate, generation-with-context rate, average structured citations, citation validity, and stage latency.

Citation validity checks every returned citation's ID, chunk ID, source name, and location against `context_sources` actually admitted to the LLM context. Empty citation sets have validity rate `null`, not 100%. Abstention correctness is only calculated for queries with explicit `answerable` labels. No metric uses answer fluency as proof of correctness.

## Results and charts

Each run creates a new timestamped directory under `backend/evaluation_results/` with:

- `retrieval_results.json` and `.csv`
- `latency_results.json`
- `per_query_results.csv`
- `rag_results.json`

The metadata includes timestamp, dataset path/SHA-256, query and corpus counts, label counts, model names, provider, retrieval weights, candidate/final K, and evaluation K. Previous runs are not overwritten.

Generate charts from one saved run:

```powershell
..\.venv\Scripts\python.exe generate_evaluation_charts.py --run-dir evaluation_results/run_<timestamp>
```

Images are saved under `evaluation_results/charts/<run_id>/`. Precision@5, Recall@5, MRR, and NDCG@5 charts are generated only when labeled quality metrics exist. Latency charts are generated only from saved measured timings. Missing measurements are skipped, not fabricated.

## Interpreting results and limitations

- Small demo datasets are for pipeline verification, not statistically meaningful academic claims.
- Report label authors, instructions, disagreements/adjudication, query split, corpus snapshot, and candidate pool coverage in the final project report.
- Compare actual values; do not presume hybrid/reranking is superior.
- Retrieval metrics measure labeled relevance, not answer truth.
- CrossEncoder relevance logits are not confidence probabilities.
- RAG reference token scores do not establish factual correctness.
- Latency depends on CPU load, model cache, local hardware, and Ollama/model configuration; report hardware and repeat count with results.

## Reproduction

From `backend/`:

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
..\.venv\Scripts\python.exe evaluate.py --dataset data/evaluation/latency_queries.jsonl --repeats 5 --run-rag
..\.venv\Scripts\python.exe generate_evaluation_charts.py --run-dir evaluation_results/run_<timestamp>
```

For final retrieval quality, first create and review `data/evaluation/evaluation.jsonl`; then run the same evaluator with that labeled dataset. No metric values in this document are experimental results.
