import re


class CitationService:
    CITATION_PATTERN = re.compile(r"\[(\d+)\]")

    @staticmethod
    def build(context_sources: list[dict]) -> list[dict]:
        return [
            {
                "citation_id": source["citation_id"],
                "source_name": source["source_name"],
                "source_type": source["source_type"],
                "location": source["location"],
                "author": source.get("author"),
                "date": source.get("date"),
                "chunk_id": source.get("chunk_id"),
                "score": source.get("score"),
            }
            for source in context_sources
        ]

    def remove_unknown_citations(self, answer: str, context_sources: list[dict]) -> str:
        allowed_ids = {str(source["citation_id"]) for source in context_sources}
        cleaned = self.CITATION_PATTERN.sub(
            lambda match: match.group(0) if match.group(1) in allowed_ids else "",
            answer,
        )
        return re.sub(r"[ \t]{2,}", " ", cleaned).strip()
