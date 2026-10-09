# MemoraAI

MemoraAI is a local-first knowledge application for PDF, DOCX, TXT, and WhatsApp text exports. It includes a React frontend, FastAPI APIs, BM25 and semantic-vector retrieval, hybrid fusion, local CrossEncoder reranking, timeline lookup, and optional local RAG with source citations. Video ingestion is not implemented.

## Requirements

- Python 3.11 (use the project-local `.venv`)
- Backend dependencies from `backend/requirements.txt`
- Local indexes and document data in `backend/data/` (not included in the public repository; build or add your own locally)
- Optional: Ollama and a locally downloaded chat model to enable answer generation

## Setup

From the repository root, create the Python 3.11 environment, activate it, and install backend dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
Copy-Item backend/.env.example backend/.env
```

Edit `backend/.env`. `LLM_MODEL` is intentionally empty by default, which keeps `/ask` in retrieval-only mode. To enable generation, install and start Ollama separately, pull a model that fits your machine, and set its exact installed name, for example:

```powershell
ollama pull qwen2.5:3b
```

```dotenv
LLM_PROVIDER=ollama
LLM_MODEL=qwen2.5:3b


LLM_BASE_URL=http://127.0.0.1:11434
```

The app only accepts loopback Ollama URLs; it does not send documents or questions to a cloud model API.

## Run

Start the backend with `backend/` as the working directory:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

The ask endpoint is `POST /ask`. With no configured/available local model it returns retrieval results and marks generation unavailable; with a local model it adds a grounded answer and citations.

Run the real local smoke test after configuring `LLM_MODEL`:

```powershell
..\.venv\Scripts\python.exe smoke_test_local_rag.py --query "computer science information technology exam at IIT Roorkee"
```

The smoke test prints retrieved and reranked sources, exact context, answer, and citations. It requires a running local Ollama model.

## Frontend

In a second terminal, install and start the Vite app:

```powershell
Set-Location frontend
npm install
npm run dev
```

The frontend uses `http://localhost:8000` by default. To change it, copy `.env.example` to `.env.local` inside `frontend/` and set `VITE_API_BASE_URL`.

## Tests

From `backend/`:

```powershell
..\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Automated RAG tests mock the LLM runtime and do not require Ollama. Embedding and reranker tests use their real pretrained local models; first use may require downloading those model weights.

## Documentation

- [Current architecture](docs/CURRENT_ARCHITECTURE.md)
- [Local embeddings](docs/LOCAL_EMBEDDINGS.md)
- [Neural reranking](docs/RERANKING.md)
- [Local RAG](docs/LOCAL_RAG.md)
- [Evaluation and benchmarking](docs/EVALUATION.md)

## Evaluation

No human-verified relevance labels are currently included. Run the unlabelled latency probe (quality metrics will be reported as not measured):

```powershell
Set-Location backend
..\.venv\Scripts\python.exe evaluate.py --dataset data/evaluation/latency_queries.jsonl --repeats 5 --run-rag
```

Create a candidate CSV for human labeling with `create_evaluation_dataset.py`; compile fully reviewed relevance judgments with `compile_evaluation_labels.py`. Evaluation runs are saved separately and are not overwritten.
