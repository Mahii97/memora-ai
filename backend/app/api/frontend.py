import json
import re
import sqlite3
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from app.database.repository import DocumentRepository
from app.embeddings.embedding_service import EmbeddingService
from app.llm.llm_service import LLMService
from app.pipeline.index_pipeline import IndexPipeline
from app.search_resources import search_resources
from app.retrieval.reranker import RerankerService

router = APIRouter(tags=["Frontend"])
BACKEND_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_ROOT / "data"
EVALUATION_DIR = BACKEND_ROOT / "evaluation_results"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    source_type: str | None = None
    author: str | None = None
    date_from: str | None = None
    date_to: str | None = None


def latest_evaluation():
    if not EVALUATION_DIR.exists():
        return None
    for directory in sorted(EVALUATION_DIR.glob("run_*"), reverse=True):
        path = directory / "retrieval_results.json"
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                return {"run_id": directory.name.removeprefix("run_"), **data}
            except (OSError, json.JSONDecodeError):
                continue
    return None


@router.get("/api/dashboard")
def dashboard():
    try:
        counts = DocumentRepository().counts()
        recent = DocumentRepository().list_documents(limit=6)
        db_status = "ok"
    except sqlite3.Error as error:
        raise HTTPException(status_code=503, detail="Database unavailable") from error
    artifact_names = ["vector_index.pkl", "bm25_index.json", "timeline_index.json", "embeddings.npy"]
    artifacts = {name: (DATA_DIR / name).is_file() for name in artifact_names}
    vector_exists = artifacts["vector_index.pkl"]
    indexed_chunks = 0
    if vector_exists:
        try:
            indexed_chunks = len(search_resources.get_engine().vector.index.metadata)
        except Exception:
            indexed_chunks = 0
    evaluation = latest_evaluation()
    return {
        "counts": counts,
        "indexed_chunks": indexed_chunks,
        "indexed": vector_exists and indexed_chunks == counts["chunks"],
        "recent_documents": recent,
        "recent_queries": {"available": False, "reason": "Query history is not persisted."},
        "evaluation": ({"run_id": evaluation["run_id"], "status": evaluation.get("status"),
                        "metrics": evaluation.get("metrics"), "latency": evaluation.get("latency")}
                       if evaluation else None),
        "database_status": db_status,
        "artifacts": artifacts,
    }


@router.get("/api/documents")
def documents(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
    repo = DocumentRepository()
    return {"total": repo.count_documents(), "limit": limit, "offset": offset,
            "results": repo.list_documents(limit=limit, offset=offset)}


@router.get("/api/documents/{document_id}")
def document_detail(document_id: str):
    item = DocumentRepository().get_document(document_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return item


@router.delete("/api/documents/{document_id}")
def delete_document(document_id: str):
    repo = DocumentRepository()
    if repo.get_document(document_id) is None:
        raise HTTPException(status_code=404, detail="Document not found")
    repo.delete_document(document_id)
    summary = IndexPipeline.rebuild(repo.get_all_chunks())
    search_resources.invalidate()
    return {"status": "success", "document_id": document_id, **summary}


@router.get("/api/memories")
def memories(
    q: str | None = None,
    source_type: str | None = None,
    document_id: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
):
    total, results = DocumentRepository().browse_chunks(q, source_type, document_id, limit, offset)
    return {"total": total, "limit": limit, "offset": offset, "results": results}


def filtered(item, request):
    created = str(item.get("created_at", ""))
    if request.source_type and item.get("source_type", "").lower() != request.source_type.lower():
        return False
    if request.author and request.author.casefold() not in str(item.get("author", "")).casefold():
        return False
    if request.date_from and created[:10] < request.date_from:
        return False
    if request.date_to and created[:10] > request.date_to:
        return False
    return True


@router.post("/api/search")
def advanced_search(request: SearchRequest):
    engine = search_resources.get_engine()
    if engine is None:
        raise HTTPException(status_code=503, detail="Search indexes are unavailable")
    vector = search_resources.embedding_service.embed_query(request.query)
    trace = engine.retrieve_candidates(request.query, vector)
    candidates = [item for item in trace["candidates"] if filtered(item, request)]
    reranked = engine.reranker.rerank(request.query, candidates, top_k=request.top_k)
    tokenizer = engine.tokenizer
    bm25 = engine.bm25.search(tokenizer.tokenize(request.query), k=request.top_k)
    semantic = engine.vector.search(vector, k=request.top_k)
    return {
        "query": request.query,
        "count": len(reranked),
        "results": reranked,
        "vector_results": [{"score": item["score"], **item["chunk"]} for item in semantic if filtered(item["chunk"], request)],
        "bm25_results": [{"score": item["score"], **item["chunk"]} for item in bm25 if filtered(item["chunk"], request)],
        "hybrid_results": candidates[:request.top_k],
        "reranked_results": reranked,
        "retrieval": {key: trace[key] for key in ("vector_candidate_count", "bm25_candidate_count", "candidate_count", "vector_latency_ms", "bm25_latency_ms", "hybrid_latency_ms")},
    }


@router.get("/timeline/search")
def timeline_search(
    q: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    author: str | None = None,
    source: str | None = None,
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    timeline = search_resources.get_timeline_search()
    if timeline is None:
        raise HTTPException(status_code=503, detail="Timeline index is unavailable")
    rows = []
    for item in timeline.timeline:
        created = str(item.get("created_at", ""))[:10]
        if date_from and created < date_from or date_to and created > date_to:
            continue
        if author and author.casefold() not in str(item.get("author", "")).casefold():
            continue
        if source and source.casefold() != str(item.get("source_type", "")).casefold():
            continue
        if q and q.casefold() not in (str(item.get("text", "")) + " " + str(item.get("source_name", ""))).casefold():
            continue
        rows.append(item)
    rows.sort(key=lambda item: item.get("timestamp", 0), reverse=True)
    return {"total": len(rows), "limit": limit, "offset": offset, "results": rows[offset:offset + limit]}


@router.get("/api/evaluations")
def evaluations():
    runs = []
    if EVALUATION_DIR.exists():
        for directory in sorted(EVALUATION_DIR.glob("run_*"), reverse=True):
            path = directory / "retrieval_results.json"
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            latency_path = directory / "latency_results.json"
            rag_path = directory / "rag_results.json"
            latency = json.loads(latency_path.read_text(encoding="utf-8")) if latency_path.exists() else None
            rag = json.loads(rag_path.read_text(encoding="utf-8")) if rag_path.exists() else None
            runs.append({"run_id": directory.name.removeprefix("run_"), "retrieval": data,
                         "latency": latency, "rag": rag})
    return {"count": len(runs), "results": runs}


@router.get("/api/evaluations/{run_id}")
def evaluation_detail(run_id: str):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", run_id):
        raise HTTPException(status_code=400, detail="Invalid run ID")
    directory = EVALUATION_DIR / f"run_{run_id}"
    if not directory.is_dir():
        raise HTTPException(status_code=404, detail="Evaluation run not found")
    result = {"run_id": run_id}
    for name in ("retrieval_results.json", "latency_results.json", "rag_results.json"):
        path = directory / name
        if path.is_file():
            try:
                result[name.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                result[name.removesuffix(".json")] = None
    return result


@router.get("/api/status")
def system_status():
    checks = {}
    try:
        conn = sqlite3.connect(f"file:{(DATA_DIR / 'memora.db').as_posix()}?mode=ro", uri=True)
        conn.execute("SELECT 1 FROM documents LIMIT 1").fetchone()
        conn.close()
        checks["database"] = {"status": "ok"}
    except sqlite3.Error as error:
        checks["database"] = {"status": "error", "detail": str(error)}
    for artifact in ("vector_index.pkl", "bm25_index.json", "timeline_index.json", "embeddings.npy"):
        path = DATA_DIR / artifact
        checks[artifact.removesuffix(".json").removesuffix(".pkl").removesuffix(".npy")] = {
            "status": "ok" if path.is_file() else "missing", "available": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
        }
    try:
        checks["embedding_model"] = {"status": "ok", "model": EmbeddingService().model_name,
                                     "dimension": EmbeddingService().dimension}
    except Exception as error:
        checks["embedding_model"] = {"status": "error", "model": EmbeddingService().model_name,
                                     "detail": str(error)}
    try:
        reranker = RerankerService()
        _ = reranker.model
        checks["reranker"] = {"status": "ok", "model": reranker.model_name}
    except Exception as error:
        checks["reranker"] = {"status": "error", "model": RerankerService().model_name,
                               "detail": str(error)}
    llm = LLMService()
    try:
        checks["llm"] = llm.health_check()
    except Exception as error:
        checks["llm"] = {"status": "unavailable", "provider": llm.provider,
                         "model": llm.model_name or None, "available": False,
                         "detail": f"Local LLM health check failed: {error}"}
    overall = "ok" if all(item["status"] == "ok" for key, item in checks.items()
                           if key not in {"reranker", "embedding_model", "llm"}) else "degraded"
    return {"status": overall, "api": "ok", "checks": checks}
