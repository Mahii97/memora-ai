from dataclasses import dataclass

@dataclass
class Page:
    page_number: int
    text: str

@dataclass
class Document:
    id: str
    filename: str
    pages: list[Page]