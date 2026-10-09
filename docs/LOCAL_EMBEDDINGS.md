# Local Semantic Embeddings

## Selected model

MemoraAI uses `sentence-transformers/all-MiniLM-L6-v2` through Sentence Transformers. It produces 384-dimensional sentence embeddings, is compact enough for CPU inference, and is distributed under the Apache-2.0 license. The pretrained encoder provides semantic text representations; MemoraAI does not train this model.

The model is configurable with `EMBEDDING_MODEL`, which accepts a Hugging Face model ID or a local model directory. `EMBEDDING_DEVICE` defaults to `cpu`.

## Installation

Use the backend's active Python environment. With the currently verified interpreter:

```powershell
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The verified project environment is Python 3.11.7 with PyTorch `2.14.0+cpu`, Transformers `5.17.0`, and Sentence Transformers `6.1.0`. The model runtime does not require FAISS; vector search remains a NumPy cosine scan.

Copy `.env.example` to `.env` in `backend/` and adjust the model/device if needed:

```dotenv
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu
```

## Local loading and cache behavior

`EmbeddingService` loads the model lazily and caches the loaded `SentenceTransformer` instance by model name and device for reuse within the process. Model inference runs on the configured device; the default is CPU.

On first use, Hugging Face Hub downloads the model files, so internet access is required unless the model is already cached or `EMBEDDING_MODEL` points to a local model directory. Subsequent loads use the Hugging Face cache. Set `HF_HOME` to relocate that cache. After the first download, inference is local and does not call a hosted embedding API. On Windows without symlink support, the Hub cache may use extra disk space; this does not prevent model use.

The selected model weights were downloaded and actually loaded on CPU during implementation. The verified output dimension is 384.

## Embedding API

`app.embeddings.embedding_service.EmbeddingService` provides:

- `embed_text(text)`: one general-purpose text vector.
- `embed_query(query)`: one query vector.
- `embed_documents(texts)`: a batch matrix with one row per document.
- `dimension`: the model-reported embedding dimension; no dimension is hard-coded in indexing or query encoding.

Embeddings are returned as normalized `float32` NumPy arrays. The same service/configuration must be used for corpus and query embeddings. If the model changes, regenerate `data/embeddings.npy` and `data/vector_index.pkl` before searching; vectors from different models cannot be mixed.

## Building and verifying indexes

From `backend/`, after `data/tokenized_chunks.json` has been created:

```powershell
..\.venv\Scripts\python.exe build_embeddings.py
..\.venv\Scripts\python.exe build_vector_index.py
..\.venv\Scripts\python.exe build_bm25_index.py
..\.venv\Scripts\python.exe build_timeline.py
```

Or invoke the existing `IndexPipeline.rebuild()` to execute those build scripts in order. The upload flow still rebuilds BM25 and timeline indexes as before, but the vector rebuild now uses the configured pretrained model. No random embedding weights are produced or loaded.

Run the complete backend test suite from `backend/`:

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`tests/test_embeddings.py` loads the actual model and checks CPU loading, query/document/batch embeddings, consistent dimensions, and NumPy cosine vector search. The first test run downloads model files if they are not already cached.
