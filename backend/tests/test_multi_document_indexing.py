import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starlette.datastructures import UploadFile

from app.api import ask as ask_api
from app.api import timeline as timeline_api
from app.api import upload as upload_api
from app.database import database
from app.database.repository import DocumentRepository
from app.models.chunk_model import Chunk
from app.pipeline.index_pipeline import IndexPipeline
from app.retrieval.bm25 import BM25Index
from app.search_resources import SearchResourceManager
from app.vector.vector_index import VectorIndex


class OfflineLLM:
    def __init__(self):
        self.provider = "ollama"
        self.model_name = ""

    def is_available(self):
        return False


class MultiDocumentIndexingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_dir = self.root / "data"
        self.upload_dir = self.data_dir / "uploads"
        self.upload_dir.mkdir(parents=True)
        self.previous_cwd = Path.cwd()
        os.chdir(self.root)

        self.patchers = [
            patch("app.database.database.DB_PATH", self.data_dir / "memora.db"),
            patch("app.pipeline.index_pipeline.IndexPipeline.DATA_DIR", self.data_dir),
            patch("app.api.upload.UPLOAD_DIR", str(self.upload_dir)),
        ]
        for patcher in self.patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        database.initialize_database()
        self.resources = SearchResourceManager(self.data_dir)
        self.resource_patchers = [
            patch("app.api.upload.search_resources", self.resources),
            patch("app.api.ask.search_resources", self.resources),
            patch("app.api.timeline.search_resources", self.resources),
            patch("app.rag.rag_service.LLMService", OfflineLLM),
        ]
        for patcher in self.resource_patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.temporary.cleanup)
        self.addCleanup(self._restore_cwd)

    def _restore_cwd(self):
        os.chdir(self.previous_cwd)

    @staticmethod
    def _document_a():
        return (
            "AURORAQUARTZ_PROTOCOL is a fictional local transfer protocol. "
            "It sends quartz packets between the aurora station and a receiver."
        )

    @staticmethod
    def _document_b():
        return (
            "BASILISKCIRCUIT_STORAGE is a fictional storage design. "
            "It stores basilisk records in a ceramic memory bank."
        )

    async def _upload_text(self, filename, text):
        uploaded = UploadFile(filename=filename, file=__import__("io").BytesIO(text.encode("utf-8")))
        return await upload_api.upload_file(uploaded)

    def _upload(self, filename, text):
        return asyncio.run(self._upload_text(filename, text))

    def _ask(self, query):
        return asyncio.run(ask_api.ask(ask_api.QuestionRequest(question=query)))

    def _assert_index_consistency(self):
        tokenized = json.loads((self.data_dir / "tokenized_chunks.json").read_text(encoding="utf-8"))
        vector = VectorIndex()
        vector.load(self.data_dir / "vector_index.pkl")
        bm25 = BM25Index()
        bm25.load(self.data_dir / "bm25_index.json")
        timeline = json.loads((self.data_dir / "timeline_index.json").read_text(encoding="utf-8"))
        repository_ids = {chunk.chunk_id for chunk in DocumentRepository().get_all_chunks()}

        token_ids = {str(chunk["chunk_id"]) for chunk in tokenized}
        vector_ids = {str(chunk["chunk_id"]) for chunk in vector.metadata}
        bm25_ids = {str(chunk["chunk_id"]) for chunk in bm25.documents}
        timeline_ids = {str(chunk["chunk_id"]) for chunk in timeline}

        self.assertEqual(token_ids, vector_ids)
        self.assertEqual(token_ids, bm25_ids)
        self.assertEqual(token_ids, timeline_ids)
        self.assertTrue(token_ids <= repository_ids)
        return tokenized, vector, bm25, timeline

    def test_single_document_chunks_are_in_sqlite_and_all_indexes(self):
        response = self._upload("alpha.txt", self._document_a())

        self.assertEqual(response["source_name"], "alpha.txt")
        self.assertEqual(response["indexed_chunks"], response["chunks_created"])
        tokenized, vector, bm25, timeline = self._assert_index_consistency()
        self.assertEqual(len(tokenized), response["chunks_created"])
        self.assertEqual({chunk["source_name"] for chunk in tokenized}, {"alpha.txt"})
        self.assertEqual(len(vector.metadata), len(bm25.documents))
        self.assertEqual(len(timeline), len(tokenized))

    def test_two_uploads_preserve_both_and_refresh_warm_ask_and_timeline(self):
        first_upload = self._upload("alpha.txt", self._document_a())
        first_id = first_upload["document_id"]

        first_ask = self._ask("Find AURORAQUARTZ_PROTOCOL quartz packets")
        self.assertEqual(first_ask["generation_status"], "unavailable")
        self.assertTrue(
            any(result["source_name"] == "alpha.txt" for result in first_ask["results"])
        )

        timeline_before = asyncio.run(
            timeline_api.search_by_source(timeline_api.SourceRequest(source="txt"))
        )
        self.assertEqual(timeline_before["count"], first_upload["indexed_chunks"])
        self.assertIsNotNone(self.resources._engine)
        self.assertIsNotNone(self.resources._timeline_search)

        second_upload = self._upload("beta.txt", self._document_b())
        second_id = second_upload["document_id"]
        self.assertNotEqual(first_id, second_id)
        self.assertIsNone(self.resources._engine)
        self.assertIsNone(self.resources._rag_service)
        self.assertIsNone(self.resources._timeline_search)

        tokenized, vector, bm25, timeline = self._assert_index_consistency()
        expected_document_ids = {first_id, second_id}
        self.assertEqual(
            {str(chunk["document_id"]) for chunk in tokenized}, expected_document_ids
        )
        self.assertEqual(
            {str(chunk["document_id"]) for chunk in vector.metadata}, expected_document_ids
        )
        self.assertEqual(
            {str(chunk["document_id"]) for chunk in bm25.documents}, expected_document_ids
        )
        self.assertEqual(
            {str(chunk["document_id"]) for chunk in timeline}, expected_document_ids
        )

        second_ask = self._ask("Find BASILISKCIRCUIT_STORAGE basilisk records")
        self.assertEqual(second_ask["generation_status"], "unavailable")
        self.assertTrue(
            any(result["source_name"] == "beta.txt" for result in second_ask["results"])
        )
        first_ask_again = self._ask("Find AURORAQUARTZ_PROTOCOL quartz packets")
        self.assertTrue(
            any(result["source_name"] == "alpha.txt" for result in first_ask_again["results"])
        )

        timeline_after = asyncio.run(
            timeline_api.search_by_source(timeline_api.SourceRequest(source="txt"))
        )
        timeline_document_ids = {item["document_id"] for item in timeline_after["results"]}
        self.assertEqual(timeline_document_ids, expected_document_ids)
        self.assertEqual(timeline_after["count"], len(tokenized))


class IndexPromotionSafetyTests(unittest.TestCase):
    def test_failed_artifact_promotion_restores_previous_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            data_dir = Path(temporary) / "data"
            data_dir.mkdir()
            old_contents = {}
            for artifact in IndexPipeline.ARTIFACTS:
                content = f"previous:{artifact}".encode("utf-8")
                (data_dir / artifact).write_bytes(content)
                old_contents[artifact] = content

            chunk = Chunk(
                "failure-test-chunk",
                "failure-test-document",
                "txt",
                "failure.txt",
                "Unknown",
                "2026-09-28",
                "Line 1",
                1,
                "A staged bundle used only to test rollback.",
                0,
                46,
            )
            real_replace = os.replace
            injected = {"done": False}

            def fail_during_install(source, destination):
                source_path = Path(source)
                if (
                    not injected["done"]
                    and source_path.name == "bm25_index.json"
                    and ".index-build-" in str(source_path)
                ):
                    injected["done"] = True
                    raise OSError("injected promotion failure")
                return real_replace(source, destination)

            with patch("app.pipeline.index_pipeline.os.replace", side_effect=fail_during_install):
                with self.assertRaisesRegex(OSError, "injected promotion failure"):
                    IndexPipeline.rebuild([chunk], data_dir=data_dir)

            self.assertTrue(injected["done"])
            self.assertEqual(
                {artifact: (data_dir / artifact).read_bytes() for artifact in IndexPipeline.ARTIFACTS},
                old_contents,
            )
            self.assertFalse(any(path.name.startswith(".index-build-") for path in data_dir.iterdir()))


if __name__ == "__main__":
    unittest.main()
