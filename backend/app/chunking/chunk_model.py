from dataclasses import dataclass

@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int
    text: str
    start_char: int
    end_char: int