import json
import numpy as np
from pathlib import Path

from app.vector.vector_index import VectorIndex

DATA_DIR = Path(__file__).resolve().parent / "data"
embeddings = np.load(DATA_DIR / "embeddings.npy")

with (DATA_DIR / "tokenized_chunks.json").open(encoding="utf-8") as f:

    metadata = json.load(f)

index = VectorIndex()

index.build(
    embeddings,
    metadata
)

index.save(
    DATA_DIR / "vector_index.pkl"
)

print("Vector index created")