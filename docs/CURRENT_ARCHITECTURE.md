# MemoraAI Current Architecture

## Scope and audit date

This document records the repository state inspected on 2026-09-28, before implementation of the advanced local-first multimodal brief. It is an inventory, not a claim that every existing path is production-ready.

## Repository layout

- `main.py`: root FastAPI entry point.
- `backend/main.py`: second FastAPI entry point; additionally registers timeline routes.
- `backend/app/`: application modules grouped into API, ingestion, models, chunking, database, embeddings, tokenizer, vector, retrieval, timeline, citations, and pipeline.
- `backend/*.py`: index-building, training, diagnostic, and manual test scripts.
- `backend/requirements.txt`: backend dependencies.
- `backend/data/`: SQLite database, uploaded files, and generated JSON/NumPy/pickle indexes and datasets.
- `backend/Dataset/`: two DOCX sample files.

No frontend directory, `package.json`, README, existing `docs/` directory, `.gitignore`, `.env` example, Python project manifest, Alembic configuration, or formal test configuration was present in the inspected workspace.

## Existing architecture

### Runtime flow

1. FastAPI registers upload and ask routes; `backend/main.py` also registers timeline routes.
2. Upload writes the incoming file under `backend/data/uploads/`, chooses a parser by extension, parses it to a `SourceDocument` containing page-like text records, chunks the text, and writes the document, records, and chunks to SQLite.
3. After storing each upload, the upload route reads all persisted chunks through `DocumentRepository.get_all_chunks()`. `IndexPipeline.rebuild()` creates one canonical tokenized dataset and stages embeddings, vector, BM25, and timeline artifacts together. It validates chunk-ID agreement, promotes the completed bundle, and rolls back prior artifacts if promotion fails.
4. `/ask` obtains lazy-loaded retrieval/RAG services from `SearchResourceManager`. A successful upload invalidates that manager after index promotion, so the next ask reloads the new bundle without restarting the server.
5. Timeline routes obtain their search service from the same invalidatable resource manager, so the next timeline request reloads the newly promoted timeline index.

### Existing API surface

| Method | Path | Behavior |
| --- | --- | --- |
| `POST` | `/upload` | Accept one uploaded file, ingest it, rebuild indexes, and return basic ingestion counts. |
| `POST` | `/ask` | Return ranked text excerpts and metadata; this is retrieval, not generated question answering. |
| `POST` | `/timeline/date` | Return timeline entries whose `created_at` starts with the requested date. |
| `POST` | `/timeline/author` | Return exact case-insensitive author matches. |
| `POST` | `/timeline/source` | Return exact source-type matches. |

There are no authentication, document-list/detail/delete, video, conversation, graph, analytics, search-history, evaluation, or health endpoints. Route availability depends on which entry point is launched: the root `main.py` does not register timeline routes, while `backend/main.py` does.

### Ingestion and content models

- `ParserFactory` dispatches PDF, DOCX, TXT, and WhatsApp-like text exports. The WhatsApp parser accepts text input, although the factory also maps `.zip` to that parser; it does not extract ZIP contents.
- PDF parsing uses PyMuPDF text extraction. There is no scanned-page OCR fallback.
- DOCX parsing extracts paragraphs and table rows. It labels records as paragraphs/table rows rather than true page numbers.
- TXT parsing stores each nonempty line as a record.
- WhatsApp parsing recognizes one specific date/time/message pattern and skips nonmatching lines.
- `TextCleaner` collapses whitespace.
- `SourceDocument`, `SourcePage`, and chunk dataclasses are in `backend/app/models/`. A second, unused-looking `Page`/`Document` model and a second chunk dataclass exist in the chunking package.
- `SmartChunker` splits each record into fixed 500-character windows with 100-character overlap; it does not preserve semantic boundaries beyond not crossing a record boundary.
- Metadata is sparse: parsers currently assign `author="Unknown"` except WhatsApp, which uses `Multiple Participants`; extracted document title/author/date/headings are not populated.

### Database

The active database layer is direct `sqlite3` access at the relative path `data/memora.db` (normally expected to be launched with `backend/` as the working directory). `initialize_database()` creates:

- `documents(id, source_type, source_name, author, created_at, total_pages)`
- `pages(id, document_id, location, text)`
- `chunks(chunk_id, document_id, source_type, source_name, author, created_at, location, chunk_index, text, start_char, end_char)`

The schema has no declared foreign keys, user ownership, uniqueness constraints beyond primary keys, timestamps for updates, or migration system. `DocumentRepository` provides inserts and `get_all_chunks()` for full-corpus indexing, but no deletion or transactional ingestion across document/pages/chunks. `save_*` opens independent connections.

### Tokenization, embeddings, and indexes

- Tokenization lowercases text and retains ASCII alphanumeric sequences.
- `DatasetBuilder` builds a vocabulary and stores tokens, token IDs, source metadata, document IDs, locations, and text in `data/tokenized_chunks.json`. The canonical dataset is rebuilt from all persisted SQLite chunks after each upload.
- `EmbeddingBuilder` uses the configured local `sentence-transformers/all-MiniLM-L6-v2` model to produce 384-dimensional embeddings.
- Vector index uses a pickled dense NumPy matrix and parallel metadata. Search performs a full cosine scan. Pickle loading must only be used with trusted local artifacts.
- BM25 is implemented locally and stores its documents, document frequencies, and average length as JSON. BM25 parameters are fixed in code.
- Hybrid fusion independently min-max normalizes each returned top-k result set and applies fixed weights of 0.60 vector / 0.40 BM25. It returns a single hybrid score, not component scores.
- Persisted artifacts include `bm25_index.json`, `vector_index.pkl`, `embeddings.npy`, `tokenized_chunks.json`, `vocabulary.json`, and `timeline_index.json`. `embedding_weights.npy` may remain as an unused legacy artifact. Each successful upload rebuilds all six active artifacts from the full stored chunk corpus using a staging directory and rollback-safe promotion.

### Query analysis, citations, and timeline

- `QueryAnalyzer` detects only an ISO date, a few literal source words, and an author following `from`.
- Hybrid filtering silently falls back to unfiltered results if metadata filters match nothing, which can return results that violate the requested filter.
- `Citation` is a small dataclass and is not used by the ask route; there is no page-accurate citation assembly or citation validation.
- Timeline is a persisted JSON index of chunks sorted by document creation timestamp. It preserves `document_id`; its date is the ingestion/document timestamp, not necessarily a date mentioned in the content. Missing or invalid dates receive timestamp zero.

### Dependencies and development checks

`backend/requirements.txt` includes FastAPI, Uvicorn, python-multipart, PyMuPDF, NumPy, python-docx, Sentence Transformers, Transformers, and matplotlib. SQLAlchemy/Alembic/PostgreSQL, video processing/transcription, OCR, authentication, and React are not implemented.

The `test_*.py` scripts are manual probes, not pytest tests: one prompts interactively and the DOCX tests depend on a specific local upload path. Other scripts build indexes, inspect data, or generate pairs. No automated regression suite or evaluation dataset/metrics implementation was found.

## Existing limitations and risks

- Ingestion and full-corpus index rebuild run synchronously in the upload request. Database writes commit before indexing. If index generation fails, the new document remains stored while the prior valid index bundle is retained; the new content requires a later successful rebuild to become searchable.
- Database/repository/index paths are relative to the process working directory.
- Startup initializes SQLite but does not rebuild indexes. `SearchResourceManager` lazily loads persisted artifacts and returns no search service if required artifacts are missing.
- No user accounts, access control, password handling, conversation storage, encryption policy, deletion workflow, or privacy/security controls are implemented.
- No video/audio ingestion, timestamped transcript model, OCR, entity/topic extraction, graph store/API, frontend, observability, or measured retrieval evaluation exists.
- Two entry points can drift: `main.py` and `backend/main.py` register different routes.

## Upgrade plan: files expected to change

Changes should be staged, preserving compatibility and adding tests before replacing existing behavior.

- `backend/main.py` and root `main.py`: converge on one documented application entry point and register routers consistently.
- `backend/requirements.txt`: add only dependencies used by an implemented, tested feature.
- `backend/app/api/upload.py`, `ask.py`, and `timeline.py`: secure upload handling, typed request/response schemas, filters, citations, and backward-compatible endpoints (upload-all-chunks indexing and cache refresh are implemented).
- `backend/app/database/database.py` and `repository.py`: introduce a database abstraction, environment-driven configuration, SQLAlchemy models/repositories, and migrations while retaining SQLite development support (`get_all_chunks()` is implemented).
- `backend/app/models/`: define document/chunk/video-segment and citation metadata needed by the shared retrieval pipeline.
- `backend/app/ingestion/parser_factory.py` and relevant parser modules: preserve current supported formats, add OCR/video processing as separate services, and validate inputs.
- `backend/app/chunking/smart_chunker.py`: keep fixed-window chunking as a fallback while adding configurable boundary-aware chunking.
- `backend/app/embeddings/`: replace random vectors with a configurable local `EmbeddingService` and cache keyed by content/model.
- `backend/app/vector/`: introduce a `VectorStore` interface and local FAISS implementation with persistent metadata.
- `backend/app/retrieval/`: centralize configurable score weights, preserve BM25, expose component scores, add reranking and filter-correct behavior.
- `backend/app/citations/`: expand the currently unused citation helper into source-location assembly/validation.
- `backend/app/pipeline/index_pipeline.py`: staged full-corpus rebuild with validated chunk-ID consistency and rollback-safe artifact promotion is implemented; incremental indexing remains future work.
- Add focused tests under `backend/tests/`; keep manual scripts only where useful as developer utilities.
- Add `frontend/` as a separate React/Vite application only after stable API contracts are in place.
- Add project configuration/security files such as `.env.example`, `.gitignore`, and Alembic configuration when their corresponding features are implemented.
- Add the requested architecture, API, database, local-AI, video, and evaluation documents as the implementation establishes their actual behavior.

## Existing functionality to preserve initially

- PDF/DOCX/TXT parsing and source metadata flow, with compatibility tests before parser refactors.
- Existing WhatsApp text parsing behavior while improving supported export formats separately.
- `SmartChunker` as a fallback option.
- BM25 retrieval and existing timeline search semantics where they are correct; do not replace them before baseline tests and comparisons exist.
- Existing API behavior for upload and retrieval results where practical; introduce new typed response fields additively before considering a versioned API change.
- Current user data and generated artifacts: do not delete or overwrite existing uploads/database/indexes during migration without an explicit backup/migration strategy.

## Recommended implementation order

1. Establish tests, configuration, error handling, safe upload storage, and one canonical app entry point.
2. Add real local embeddings and a vector-store abstraction; retain BM25 and measure a baseline.
3. Add configurable hybrid fusion, correct filters, reranking fallback, and detailed retrieval scores.
4. Add local RAG with citations and a retrieval-only fallback.
5. Add timestamped video transcription and searchable video citations; add OCR for scanned PDFs/slides as an optional stage.
6. Add SQLAlchemy/migrations and user isolation before exposing multi-user access; authentication and authorization must be implemented together.
7. Add entity/topic extraction and knowledge graph only after provenance and core retrieval are reliable.
8. Build the frontend against stable API schemas, then add evaluation and operational documentation.

This order is intentionally incremental: the brief describes a multi-phase product, not a safe single-pass rewrite. Each phase should preserve a working local retrieval path and include focused regression tests.

## First implementation slice

After recording the audit, the following initial changes were made:

- Added `backend/tests/test_retrieval_basics.py` with six `unittest` checks for tokenization, fixed-window chunking, BM25 ranking, current hybrid fusion behavior, safe upload basename handling, and unsupported upload extensions.
- Updated `backend/app/api/upload.py` to allowlist `.pdf`, `.docx`, `.txt`, and `.txtwa`, enforce a 100 MB limit while reading in bounded chunks, generate a UUID-based storage filename, retain the sanitized display filename as document metadata, return HTTP errors for invalid/failed processing, and retain stored files after document persistence.
- At that point in the implementation, the random-vector embedding code was still present; it was replaced in the Phase 2 update below.

Focused validation command: `cd backend; python -m unittest discover -s tests -v` (6 tests passed). Existing warning: PyMuPDF's `fitz` import is deprecated.

## Phase 2: real local embeddings

- Added `backend/app/embeddings/embedding_service.py`, using the cached CPU Sentence Transformers model configured by `EMBEDDING_MODEL` and `EMBEDDING_DEVICE`.
- Selected `sentence-transformers/all-MiniLM-L6-v2`; the pretrained model was loaded locally and returned 384-dimensional normalized vectors.
- Replaced random embedding generation with `EmbeddingBuilder` and `build_embeddings.py`; the existing 99 tokenized chunks were embedded and the vector index was rebuilt to shape `(99, 384)`.
- Updated `/ask` query encoding to use the same embedding service. The NumPy cosine search interface and BM25 implementation remain in place.
- Removed the random `EmbeddingLayer`, old chunk `encoder.py`, and obsolete `train_embeddings.py` entry point. The old generated `embedding_weights.npy` file is left on disk as an unused legacy artifact; runtime code no longer reads or writes it.
- Added model-backed tests, backend `.env.example`, `requirements.txt` dependencies, and `docs/LOCAL_EMBEDDINGS.md`.
- Verified direct vector search, BM25, hybrid search, and timeline search against the rebuilt indexes. The current sample query returns `CS22025.pdf` as the top vector and BM25 source and as the top hybrid source.
- Focused/full backend suite command: `..\\.venv\\Scripts\\python.exe -m unittest discover -s tests -v` (run from `backend/`).

## Phase 4: local RAG

- Added `backend/app/llm/llm_service.py`, a provider/model-configured service with an Ollama local HTTP adapter. The adapter only permits loopback URLs; no cloud provider or remote base URL is used.
- Added `backend/app/rag/` for bounded context construction, a strict grounded prompt, and `RAGService`, which reuses the existing hybrid/reranker engine.
- Added `backend/app/citations/citation_service.py` and `backend/app/models/rag_response.py` for structured citations and an explicit generation/retrieval response contract.
- `/ask` retains the existing retrieval scores/results and adds answer, citations, generation status, model/provider, context source, and retrieval metadata. With no configured or available local model it returns retrieval-only results and indicates generation is unavailable.
- Added context limits via `.env`; citation IDs are allocated only for context chunks, and unknown IDs emitted by the model are removed. `RAG_MIN_RERANKER_SCORE` is an opt-in raw CrossEncoder score cutoff; it is unset by default because raw scores are not calibrated confidence probabilities and may be negative.
- Ollama was detected running locally with `qwen2.5:3b` and `qwen2.5:7b-instruct` available. The configured model remains empty by default; the actual local smoke test uses `qwen2.5:3b` explicitly and does not modify `.env`.
- The actual local smoke query completed with `qwen2.5:3b`; it returned an evidence-limited answer and structured citations for the supplied `CS22025.pdf` passages. This verifies integration, not answer-quality improvement.
- Added `backend/smoke_test_local_rag.py`, `backend/tests/test_rag.py`, `docs/LOCAL_RAG.md`, and a root `README.md`.

## Phase 5: retrieval and RAG evaluation

- Added `backend/app/evaluation/` with JSONL schema/loading, binary Precision@K/Recall@K/MRR/NDCG/Hit Rate metrics, latency summaries, retrieval benchmarking, citation validation, and RAG operational metrics.
- Exposed `HybridEngine.retrieve_candidates()` as the same production hybrid pre-rerank stage for independent evaluation; stage timers are observational and do not alter ranking.
- Added human-label candidate export/compile tools and `backend/data/evaluation/`. No human relevance judgments are present; the only staged query is explicitly latency-only.
- Added timestamped, non-overwriting JSON/CSV run output and chart generation. Quality charts are skipped when labels are absent.
- At Phase 5 evaluation time, SQLite contained 1,273 rows while the active indexes contained only the latest 99. The cause and correction are documented in Phase 6 below.
- Actual latency-only benchmark ran on one query over 99 indexed chunks, with five measured repeats per retrieval stage. Quality metrics and RAG generation were correctly marked not measured because there are no labels and `LLM_MODEL` is unset.
- Phase 5.1 created 50 query drafts grounded in the indexed `CS22025.pdf` contents, a 2,500-row full-text review CSV, and a companion JSONL with candidate IDs/source previews. Validation confirmed all candidate IDs are in the active index, all relevance cells are blank, and no final `evaluation.jsonl` was created.

Evaluation layer:

```text
Human-labeled Evaluation Dataset
		   |
	+---------+----------+------------------+
	|         |          |                  |
      BM25     Vector     Hybrid       Hybrid + Reranker
	+---------+----------+------------------+
		   |
	   Evaluation Engine
		   |
 Precision / Recall / MRR / NDCG / Hit Rate / Latency
		   |
	   JSON / CSV / Charts
```

	## Phase 6: multi-document indexing consistency

	- Added `DocumentRepository.get_all_chunks()` to read every persisted chunk with its existing IDs, document metadata, page/location, text, offsets, and creation metadata in deterministic order.
	- `upload_file()` now persists the upload, fetches the complete SQLite chunk corpus, then passes that corpus to `IndexPipeline.rebuild()` rather than indexing only the current request's local chunks.
	- `IndexPipeline.rebuild()` creates the canonical vocabulary/tokenized dataset and all embedding/vector/BM25/timeline artifacts in a same-filesystem staging directory. It validates IDs across artifacts before promotion and restores the prior bundle if promotion fails.
	- Added `SearchResourceManager` for lazy persisted-index loading. Successful upload invalidates its ask/RAG/vector/BM25 and timeline caches; the next request reloads the new bundle without a server restart.
	- Timeline index records now retain `document_id`.
	- The existing SQLite corpus was rebuilt without deduplication, as requested. Verified totals: 1,273 tokenized chunks, 1,273 embeddings at 384 dimensions, 1,273 vector entries, 1,273 BM25 documents, and 1,273 timeline entries across 18 document IDs. All artifact chunk-ID sets match SQLite exactly.
	- Added temporary-database integration tests for one upload and two sequential uploads, including a warmed `/ask`, retrieval of both unique documents after the second upload, timeline refresh, full index-ID consistency, and injected artifact-promotion failure rollback.
	- Complete backend test suite: 54 tests passed. The production index bundle now covers the full 1,273-row SQLite corpus; duplicate upload records remain indexed until a separate deduplication policy is approved.
