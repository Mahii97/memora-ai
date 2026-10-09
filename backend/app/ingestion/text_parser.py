import uuid
from datetime import datetime

from app.models.source_document import SourceDocument, SourcePage
from app.ingestion.cleaner import TextCleaner


class TxtParser:

    def parse(self, filepath):

        pages = []

        with open(filepath, "r", encoding="utf-8") as file:

            for line_number, line in enumerate(file, start=1):

                cleaned = TextCleaner.clean(line)

                if cleaned:
                    pages.append(
                        SourcePage(
                            location=f"Line {line_number}",
                            text=cleaned
                        )
                    )

        return SourceDocument(
            id=str(uuid.uuid4()),
            source_type="txt",
            source_name=filepath.split("\\")[-1],
            author="Unknown",
            created_at=datetime.now(),
            pages=pages
        )