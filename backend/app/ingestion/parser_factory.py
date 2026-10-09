from pathlib import Path

from app.ingestion.parser import PDFParser
from app.ingestion.text_parser import TxtParser
from app.ingestion.docx_parser import DocxParser
from app.ingestion.whatsapp_parser import WhatsAppParser


class ParserFactory:

    def get_parser(self, filename: str):

        extension = Path(filename).suffix.lower()

        if extension == ".pdf":
            return PDFParser()

        elif extension == ".txt":
            return TxtParser()

        elif extension == ".docx":
            return DocxParser()

        elif extension in [".zip", ".txtwa"]:
            return WhatsAppParser()

        else:
            raise ValueError(
                f"Unsupported file type: {extension}"
            )