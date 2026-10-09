from typing import Any

from pydantic import BaseModel, Field


class CitationResponse(BaseModel):
    citation_id: int
    source_name: str
    source_type: str
    location: str
    author: str | None = None
    date: str | None = None
    chunk_id: str | None = None
    score: float | None = None


class RAGResponse(BaseModel):
    status: str = "success"
    question: str
    answer: str | None = None
    citations: list[CitationResponse] = Field(default_factory=list)
    rag_enabled: bool = False
    llm_provider: str
    llm_model: str | None = None
    generation_status: str
    generation_message: str | None = None
    context_sources: list[CitationResponse] = Field(default_factory=list)
    retrieval: dict[str, Any]
    results: list[dict[str, Any]]
    retrieved_results: list[dict[str, Any]] = Field(default_factory=list)
    unique_results: list[dict[str, Any]] = Field(default_factory=list)
    documents: list[dict[str, Any]] = Field(default_factory=list)
    intent: str | None = None
    query_understanding: dict[str, Any] = Field(default_factory=dict)
    answer_task: str | None = None
    timings_ms: dict[str, float | None] = Field(default_factory=dict)
