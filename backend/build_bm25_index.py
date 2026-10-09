import json

from app.retrieval.bm25 import BM25Index


with open("data/tokenized_chunks.json","r") as f:

    dataset = json.load(f)

index = BM25Index()

index.build(dataset)

index.save("data/bm25_index.json")

print("BM25 index created")

print("Documents:", len(dataset))