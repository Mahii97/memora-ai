from dataclasses import dataclass

@dataclass
class Citation:

    document: str
    page: int
    chunk: str = ""

    def to_text(self):
        return f"{self.document} — Page {self.page}"