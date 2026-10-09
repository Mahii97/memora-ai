from app.retrieval.bm25 import BM25Index
from app.tokenizer.tokenizer import Tokenizer

tokenizer = Tokenizer()

bm25 = BM25Index()
bm25.load("data/bm25_index.json")

query = input("Enter query: ")

tokens = tokenizer.tokenize(query)

print("Query tokens:",tokens)

results = bm25.search(tokens, k=5)

for i, r in enumerate(results, 1):

    print("\n", "="*50)

    print("Rank:", i)

    print("Score:", round(r["score"], 3))

    print("Source:", r["chunk"]["source_name"])

    print("Location:", r["chunk"]["location"])

    print("Text:", r["chunk"]["text"][:200])