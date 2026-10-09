import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationQuery:
    query_id: str
    query: str
    relevant_chunk_ids: tuple[str, ...] | None = None
    relevant_document: str | None = None
    relevant_locations: tuple[str, ...] | None = None
    reference_answer: str | None = None
    notes: str | None = None
    answerable: bool | None = None

    @property
    def has_relevance_labels(self) -> bool:
        return (
            self.relevant_chunk_ids is not None
            or self.relevant_document is not None
            or self.relevant_locations is not None
        )


def load_evaluation_dataset(path: str | Path) -> list[EvaluationQuery]:
    dataset_path = Path(path)
    records = []
    seen_ids = set()
    with dataset_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSONL at {dataset_path}:{line_number}: {error.msg}"
                ) from error
            if not isinstance(raw, dict):
                raise ValueError(f"Expected a JSON object at line {line_number}")
            query_id = raw.get("query_id")
            query = raw.get("query")
            if not isinstance(query_id, str) or not query_id.strip():
                raise ValueError(f"query_id must be a nonempty string at line {line_number}")
            if query_id in seen_ids:
                raise ValueError(f"duplicate query_id {query_id!r} at line {line_number}")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"query must be a nonempty string at line {line_number}")

            relevant_ids = raw.get("relevant_chunk_ids")
            if relevant_ids is not None and (
                not isinstance(relevant_ids, list)
                or any(not isinstance(item, str) for item in relevant_ids)
            ):
                raise ValueError(f"relevant_chunk_ids must be a string array at line {line_number}")
            locations = raw.get("relevant_locations")
            if locations is not None and (
                not isinstance(locations, list)
                or any(not isinstance(item, str) for item in locations)
            ):
                raise ValueError(f"relevant_locations must be a string array at line {line_number}")
            answerable = raw.get("answerable")
            if answerable is not None and not isinstance(answerable, bool):
                raise ValueError(f"answerable must be boolean at line {line_number}")
            reference_answer = raw.get("reference_answer")
            if reference_answer is not None and not isinstance(reference_answer, str):
                raise ValueError(f"reference_answer must be a string at line {line_number}")
            relevant_document = raw.get("relevant_document")
            if relevant_document is not None and not isinstance(relevant_document, str):
                raise ValueError(f"relevant_document must be a string at line {line_number}")
            notes = raw.get("notes")
            if notes is not None and not isinstance(notes, str):
                raise ValueError(f"notes must be a string at line {line_number}")

            seen_ids.add(query_id)
            records.append(
                EvaluationQuery(
                    query_id=query_id,
                    query=query,
                    relevant_chunk_ids=(tuple(dict.fromkeys(relevant_ids)) if relevant_ids is not None else None),
                    relevant_document=relevant_document,
                    relevant_locations=(tuple(dict.fromkeys(locations)) if locations is not None else None),
                    reference_answer=reference_answer,
                    notes=notes,
                    answerable=answerable,
                )
            )
    return records


def resolve_relevant_chunk_ids(
    query: EvaluationQuery, corpus_metadata: list[dict]
) -> set[str] | None:
    if not query.has_relevance_labels:
        return None

    known_ids = {str(item["chunk_id"]) for item in corpus_metadata}
    relevant = set(query.relevant_chunk_ids or ())
    unknown_ids = relevant - known_ids
    if unknown_ids:
        raise ValueError(
            f"Unknown relevant chunk IDs for {query.query_id}: {sorted(unknown_ids)}"
        )

    locations = set(query.relevant_locations or ())
    target_document = query.relevant_document.casefold() if query.relevant_document else None
    for item in corpus_metadata:
        document_matches = (
            target_document is None
            or str(item.get("source_name", "")).casefold() == target_document
        )
        location_matches = not locations or item.get("location") in locations
        if document_matches and location_matches and (
            target_document is not None or locations
        ):
            relevant.add(str(item["chunk_id"]))
    return relevant
