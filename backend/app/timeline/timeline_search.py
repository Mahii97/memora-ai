
import json
from datetime import date, datetime
from pathlib import Path

class TimelineSearch:

    def __init__(self, index_path="data/timeline_index.json"):

        with Path(index_path).open(encoding="utf-8") as f:
            self.timeline = json.load(f)

    def by_date(self, date):

        result = []

        for item in self.timeline:

            if item["created_at"].startswith(date):
                result.append(item)

        return result

    def by_author(self, author):

        result = []

        for item in self.timeline:

            if item["author"].lower() == author.lower():
                result.append(item)

        return result

    def by_source(self, source):

        result = []

        for item in self.timeline:

            if item["source_type"] == source:
                result.append(item)

        return result

    def by_range(self, date_from: str, date_to: str) -> list[dict]:
        """Return indexed memories in an inclusive ISO date range, oldest first."""
        start = date.fromisoformat(date_from)
        end = date.fromisoformat(date_to)
        result = []
        for item in self.timeline:
            try:
                created = date.fromisoformat(str(item.get("created_at", ""))[:10])
            except ValueError:
                continue
            if start <= created <= end:
                result.append(item)
        return sorted(
            result,
            key=lambda item: (str(item.get("created_at", "")), str(item.get("chunk_id", ""))),
        )
