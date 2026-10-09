
import json
from datetime import datetime

class TimelineIndexer:

    def parse_date(self, value):

        try:
            return datetime.fromisoformat(value)
        except:
            return None

    def build(self, dataset):

        timeline = []

        for chunk in dataset:

            dt = self.parse_date(chunk["created_at"])

            timeline.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "source_type": chunk["source_type"],
                "source_name": chunk["source_name"],
                "author": chunk["author"],
                "created_at": chunk["created_at"],
                "location": chunk["location"],
                "text": chunk["text"],
                "timestamp": dt.timestamp() if dt else 0
            })

        timeline.sort(key=lambda x: x["timestamp"])

        return timeline

    def save(self, timeline, path):

        with open(path, "w") as f:
            json.dump(timeline, f, indent=4)