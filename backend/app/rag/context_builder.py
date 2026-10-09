import os
import re
from difflib import SequenceMatcher
from pathlib import Path

from dotenv import load_dotenv


BACKEND_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(BACKEND_ROOT / ".env")


class ContextBuilder:
    def __init__(self, max_chunks: int | None = None, max_characters: int | None = None):
        self.max_chunks = (
            max_chunks
            if max_chunks is not None
            else int(os.getenv("RAG_MAX_CONTEXT_CHUNKS", "5"))
        )
        self.max_characters = (
            max_characters
            if max_characters is not None
            else int(os.getenv("RAG_MAX_CONTEXT_CHARACTERS", "12000"))
        )
        if self.max_chunks < 0 or self.max_characters < 0:
            raise ValueError("RAG context limits cannot be negative")

    @staticmethod
    def _normalized_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.casefold()).strip()

    @staticmethod
    def _normalized_location(location: str) -> str:
        return re.sub(r"\s*\((?:ocr|text)\)$", "", location.casefold().strip())

    @classmethod
    def _deduplicate(cls, chunks: list[dict]) -> list[dict]:
        selected = []
        seen_ids = set()
        seen_texts = set()
        for chunk in chunks:
            text = chunk.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            chunk_id = str(chunk.get("chunk_id") or "")
            normalized = cls._normalized_text(text)
            source = str(chunk.get("source_name") or "").casefold()
            location = cls._normalized_location(str(chunk.get("location") or ""))
            text_location = (source, location, normalized)
            if (chunk_id and chunk_id in seen_ids) or text_location in seen_texts:
                continue
            duplicate = False
            if len(normalized) >= 120:
                for prior in selected:
                    prior_text = cls._normalized_text(prior["text"])
                    same_source_location = (
                        source == str(prior.get("source_name") or "").casefold()
                        and location == cls._normalized_location(str(prior.get("location") or ""))
                    )
                    if same_source_location and min(len(normalized), len(prior_text)) >= 120:
                        if SequenceMatcher(None, normalized, prior_text, autojunk=False).ratio() >= 0.94:
                            duplicate = True
                            break
            if duplicate:
                continue
            selected.append(chunk)
            seen_texts.add(text_location)
            if chunk_id:
                seen_ids.add(chunk_id)
        return selected

    @staticmethod
    def _prefer_source_diversity(chunks: list[dict]) -> list[dict]:
        first_per_source = []
        repeated_source = []
        seen_sources = set()
        for chunk in chunks:
            source = str(chunk.get("source_name") or "Unknown source").casefold()
            if source in seen_sources:
                repeated_source.append(chunk)
            else:
                seen_sources.add(source)
                first_per_source.append(chunk)
        return first_per_source + repeated_source

    def build(self, chunks: list[dict], query_plan: dict | None = None) -> dict:
        retrieved_count = len(chunks)
        unique_chunks = self._deduplicate(chunks)
        if query_plan and query_plan.get("intent") == "temporal_summary":
            unique_chunks.sort(
                key=lambda item: (
                    str(item.get("created_at") or item.get("date") or ""),
                    int(item.get("final_rank") or 0),
                )
            )
        ordered_chunks = (
            unique_chunks
            if query_plan and query_plan.get("intent") == "temporal_summary"
            else self._prefer_source_diversity(unique_chunks)
        )
        entries = []
        sources = []
        used_characters = 0

        for chunk in ordered_chunks[:self.max_chunks]:
            text = chunk.get("text")
            if not isinstance(text, str) or not text:
                continue

            citation_id = len(sources) + 1
            source_name = str(chunk.get("source_name") or "Unknown source")
            source_type = str(chunk.get("source_type") or "unknown")
            location = str(chunk.get("location") or "Unknown location")
            author = chunk.get("author")
            created_at = chunk.get("created_at") or chunk.get("date")
            score = chunk.get("reranker_score", chunk.get("hybrid_score"))

            metadata = [
                f"Source [{citation_id}]: {source_name}",
                f"Source type: {source_type}",
                f"Location: {location}",
            ]
            if author and str(author).lower() != "unknown":
                metadata.append(f"Author: {author}")
            if created_at:
                metadata.append(f"Date: {created_at}")
            if chunk.get("chunk_id"):
                metadata.append(f"Chunk ID: {chunk['chunk_id']}")
            if score is not None:
                metadata.append(f"Reranker relevance score: {score}")
            prefix = "\n".join(metadata) + "\nText: "

            separator = "\n\n" if entries else ""
            remaining = self.max_characters - used_characters - len(separator) - len(prefix)
            if remaining <= 0:
                break

            included_text = text[:remaining]
            entry = prefix + included_text
            if not included_text:
                break
            entries.append(entry)
            used_characters += len(separator) + len(entry)
            sources.append(
                {
                    "citation_id": citation_id,
                    "source_name": source_name,
                    "source_type": source_type,
                    "location": location,
                    "author": author if author and str(author).lower() != "unknown" else None,
                    "date": created_at,
                    "chunk_id": chunk.get("chunk_id"),
                    "score": float(score) if score is not None else None,
                }
            )

        return {
            "context": "\n\n".join(entries),
            "sources": sources,
            "metadata": {
                "retrieved_chunks": retrieved_count,
                "unique_chunks": len(unique_chunks),
                "unique_sources": len({str(item.get("source_name") or "Unknown source") for item in unique_chunks}),
                "selected_chunks": len(sources),
                "selected_sources": len({item["source_name"] for item in sources}),
                "context_character_count": used_characters,
                "duplicate_chunks_removed": retrieved_count - len(unique_chunks),
            },
        }
