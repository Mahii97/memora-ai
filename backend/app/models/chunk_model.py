from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    document_id: str

    source_type: str
    source_name: str

    author: str
    created_at: str

    location: str

    chunk_index: int

    text: str

    start_char: int
    end_char: int