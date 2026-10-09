import json

from app.embeddings.pair_generator import PairGenerator

with open("data/tokenized_chunks.json") as f:
    dataset = json.load(f)

generator = PairGenerator()

pairs = generator.generate(dataset)

generator.save(
    pairs,
    "data/training_pairs.json"
)

print(f"Generated {len(pairs)} pairs")