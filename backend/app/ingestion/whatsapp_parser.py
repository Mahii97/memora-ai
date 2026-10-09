import re
import uuid
from datetime import datetime

from app.models.source_document import SourceDocument, SourcePage
from app.ingestion.cleaner import TextCleaner


class WhatsAppParser:

    # Example:
    # 21/09/26, 10:15 - Rahul: Meeting at 4 PM
    MESSAGE_PATTERN = re.compile(
        r"^(\d{1,2}/\d{1,2}/\d{2}),\s(\d{1,2}:\d{2})\s-\s([^:]+):\s(.*)$"
    )

    def parse(self, filepath):

        messages = []

        with open(filepath, "r", encoding="utf-8") as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                match = self.MESSAGE_PATTERN.match(line)

                if not match:
                    continue

                date, time, sender, message = match.groups()

                cleaned = TextCleaner.clean(message)

                messages.append(
                    SourcePage(
                        location=f"{date} {time} | {sender}",
                        text=cleaned
                    )
                )

        return SourceDocument(
            id=str(uuid.uuid4()),
            source_type="whatsapp",
            source_name=filepath.split("/")[-1],
            author="Multiple Participants",
            created_at=datetime.now(),
            pages=messages
        )