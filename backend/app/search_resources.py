from pathlib import Path
from threading import RLock

from app.embeddings.embedding_service import EmbeddingService
from app.rag.rag_service import RAGService
from app.retrieval.hybrid_engine import HybridEngine
from app.timeline.timeline_search import TimelineSearch
from app.vector.search import SearchEngine
from app.vector.vector_index import VectorIndex


BACKEND_ROOT = Path(__file__).resolve().parents[1]


class SearchResourceManager:
    """Lazily load persisted search artifacts and refresh them after index swaps."""

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir is not None else BACKEND_ROOT / "data"
        self.embedding_service = EmbeddingService()
        self._lock = RLock()
        self._engine = None
        self._rag_service = None
        self._timeline_search = None

    def get_engine(self):
        with self._lock:
            if self._engine is not None:
                return self._engine

            vector_path = self.data_dir / "vector_index.pkl"
            if not vector_path.exists():
                return None

            index = VectorIndex()
            index.load(vector_path)
            if not (self.data_dir / "bm25_index.json").exists():
                return None
            timeline = self.get_timeline_search()
            if timeline is None:
                return None

            self._engine = HybridEngine(
                SearchEngine(index),
                timeline=timeline,
                bm25_index_path=self.data_dir / "bm25_index.json",
            )
            return self._engine

    def get_rag_service(self):
        with self._lock:
            if self._rag_service is not None:
                return self._rag_service
            engine = self.get_engine()
            if engine is None:
                return None
            self._rag_service = RAGService(
                engine,
                embedding_service=self.embedding_service,
            )
            return self._rag_service

    def get_timeline_search(self):
        with self._lock:
            if self._timeline_search is not None:
                return self._timeline_search
            timeline_path = self.data_dir / "timeline_index.json"
            if not timeline_path.exists():
                return None
            self._timeline_search = TimelineSearch(timeline_path)
            return self._timeline_search

    def invalidate(self) -> None:
        with self._lock:
            self._engine = None
            self._rag_service = None
            self._timeline_search = None


search_resources = SearchResourceManager()
