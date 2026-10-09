from dataclasses import dataclass
from datetime import datetime

@dataclass
class SourcePage:
    location: str
    text: str

@dataclass
class SourceDocument:

    id: str

    source_type: str      # pdf, whatsapp, docx

    source_name: str

    author: str

    created_at: datetime

    pages: list[SourcePage]