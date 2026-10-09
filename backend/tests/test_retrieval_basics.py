import unittest
from datetime import datetime

from fastapi import HTTPException

from app.chunking.smart_chunker import SmartChunker
from app.models.source_document import SourceDocument, SourcePage
from app.api.upload import validate_upload
from app.retrieval.bm25 import BM25Index
from app.retrieval.hybrid_ranker import HybridRanker
from app.tokenizer.tokenizer import Tokenizer


class TokenizerTests(unittest.TestCase):
    def test_tokenize_lowercases_and_extracts_ascii_words(self):
        self.assertEqual(
            Tokenizer().tokenize("MemoraAI, 2026: PDF."),
            ["memoraai", "2026", "pdf"],
        )


class UploadValidationTests(unittest.TestCase):
    def test_filename_is_reduced_to_safe_basename(self):
        self.assertEqual(
            validate_upload("../../private\\notes.PDF"),
            ("notes.PDF", ".pdf"),
        )

    def test_unsupported_file_type_is_rejected(self):
        with self.assertRaises(HTTPException) as error:
            validate_upload("lecture.mp4")

        self.assertEqual(error.exception.status_code, 415)


class ChunkerTests(unittest.TestCase):
    def test_chunks_keep_configured_overlap_and_record_location(self):
        document = SourceDocument(
            id="doc-1",
            source_type="txt",
            source_name="notes.txt",
            author="Unknown",
            created_at=datetime(2026, 1, 1),
            pages=[SourcePage(location="Line 1", text="abcdefghijkl")],
        )

        chunks = SmartChunker(chunk_size=8, overlap=2).create_chunks(document)

        self.assertEqual([chunk.text for chunk in chunks], ["abcdefgh", "ghijkl"])
        self.assertEqual([chunk.start_char for chunk in chunks], [0, 6])
        self.assertTrue(all(chunk.location == "Line 1" for chunk in chunks))


class BM25Tests(unittest.TestCase):
    def test_exact_match_ranks_above_nonmatching_chunk(self):
        index = BM25Index()
        index.build(
            [
                {"chunk_id": "match", "tokens": ["database", "normalization"]},
                {"chunk_id": "other", "tokens": ["video", "transcript"]},
            ]
        )

        results = index.search(["database"], k=2)

        self.assertEqual(results[0]["chunk"]["chunk_id"], "match")
        self.assertGreater(results[0]["score"], results[1]["score"])


class HybridRankerTests(unittest.TestCase):
    def test_fuses_vector_and_bm25_scores_and_returns_ranked_chunks(self):
        vector_results = [
            {"score": 0.9, "chunk": {"chunk_id": "both"}},
        ]
        bm25_results = [
            {"score": 3.0, "chunk": {"chunk_id": "both"}},
            {"score": 1.0, "chunk": {"chunk_id": "lexical"}},
        ]

        results = HybridRanker().fuse(vector_results, bm25_results)

        self.assertEqual([chunk["chunk_id"] for chunk in results], ["both", "lexical"])
        self.assertEqual(results[0]["hybrid_score"], 1.0)
        self.assertEqual(results[1]["hybrid_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
