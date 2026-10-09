# Local Neural Reranking

## Selected model

MemoraAI uses `cross-encoder/ms-marco-MiniLM-L-6-v2` through the `CrossEncoder` API already included in `sentence-transformers`. It is a compact pretrained cross-encoder trained for query/passage relevance, suitable for local CPU inference and English text retrieval. It scores each query-candidate pair jointly, so it is used only on the small hybrid candidate set, not on the entire corpus. It is not an answer-generating LLM.

The model ID is configurable. No model-specific vector dimension is used by the reranker; it consumes query/text pairs and produces one relevance score per pair.

## Local CPU and installation

The verified project environment is Python 3.11.7 with PyTorch `2.14.0+cpu`, Transformers `5.17.0`, and Sentence Transformers `6.1.0`. CrossEncoder is provided by the existing Sentence Transformers dependency, so no FAISS or cloud service is needed.

From `backend/`:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Inference defaults to CPU. The first model load requires internet access unless the model is already cached or `RERANKER_MODEL` points to a local model directory. Hugging Face Hub caches downloaded files; set `HF_HOME` to relocate the cache. Once downloaded, query scoring is performed locally.

## Configuration

Copy `.env.example` to `.env` in `backend/` and configure:

```dotenv
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_DEVICE=cpu
RERANKER_CANDIDATE_K=20
RERANKER_TOP_K=5
```

`RERANKER_CANDIDATE_K` limits the hybrid candidate pool; `RERANKER_TOP_K` sets the final returned result count. The current defaults implement top 20 to top 5.

## Candidate generation and scoring

1. BM25 retrieves up to `RERANKER_CANDIDATE_K` chunks.
2. NumPy cosine vector search retrieves up to the same number.
3. Existing hybrid fusion combines the union, preserving its current 0.60 normalized-vector / 0.40 normalized-BM25 formula.
4. The highest-ranked hybrid candidates are truncated to `RERANKER_CANDIDATE_K`.
5. `RerankerService` scores each `(query, chunk text)` pair locally and sorts by the cross-encoder relevance score.
6. The first `RERANKER_TOP_K` chunks are returned.

The reranker does not replace BM25, vector search, or hybrid fusion. Candidate records carry raw `vector_score` and `bm25_score`, normalized component scores, `hybrid_score`, `reranker_score`, and `final_rank`. The `explanation` object exposes those retrieval signals for a future â€œWhy this result?â€ UI. Scores are not chain-of-thought. CrossEncoder output is a model relevance score, not a calibrated probability.

## Model reuse and tests

`RerankerService` lazily loads and caches one CrossEncoder per model/device pair for reuse in the process. Empty candidate lists return immediately without model inference.

Run backend tests from `backend/`:

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The reranker tests load the real model, verify repeated inference and top-k behavior, test empty input, and exercise the full BM25 + vector + hybrid top-20 + reranker top-5 flow on the current indexed corpus. No quality improvement is assumed; compare the recorded hybrid order to the final reranked order for each query.
