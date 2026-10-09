
import json

from app.timeline.timeline_index import TimelineIndexer

with open("data/tokenized_chunks.json") as f:
    dataset = json.load(f)

indexer = TimelineIndexer()

timeline = indexer.build(dataset)

indexer.save(
    timeline,
    "data/timeline_index.json"
)

print(f"Indexed {len(timeline)} memories")