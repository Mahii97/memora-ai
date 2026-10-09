import math
import json
from collections import Counter


class BM25Index:

    def __init__(self):
        # All tokenized chunks
        self.documents = []

        # Document frequency of each token
        self.doc_freq = {}

        # Average document (chunk) length
        self.avg_doc_length = 0

        # BM25 parameters
        self.k1 = 1.5
        self.b = 0.75

    # ---------- BUILD INDEX ----------

    def build(self, dataset):

        self.documents = dataset

        total_length = 0
        self.doc_freq = {}

        for doc in dataset:

            tokens = doc["tokens"]
            total_length += len(tokens)

            unique_tokens = set(tokens)

            for token in unique_tokens:
                self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

        if len(dataset) > 0:
            self.avg_doc_length = total_length / len(dataset)
        else:
            self.avg_doc_length = 0

    # ---------- BM25 SCORE ----------

    def score(self, query_tokens, document):

        tokens = document["tokens"]

        tf = Counter(tokens)

        score = 0

        N = len(self.documents)
        dl = len(tokens)

        for word in query_tokens:

            if word not in self.doc_freq:
                continue

            df = self.doc_freq[word]

            # Inverse Document Frequency
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

            freq = tf.get(word, 0)

            numerator = freq * (self.k1 + 1)

            denominator = freq + self.k1 * (
                1 - self.b + self.b * dl / self.avg_doc_length
            )

            score += idf * (numerator / denominator)

        return score

    # ---------- SEARCH ----------

    def search(self, query_tokens, k=5):

        results = []

        for doc in self.documents:

            s = self.score(query_tokens, doc)

            results.append({
                "score": s,
                "chunk": doc
            })

        results.sort(
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:k]

    # Same function used by Hybrid Engine
    def search_text(self, query_tokens, k=5):
        return self.search(query_tokens, k)

    # ---------- SAVE ----------

    def save(self, path):

        with open(path, "w", encoding="utf-8") as f:

            json.dump(
                {
                    "documents": self.documents,
                    "doc_freq": self.doc_freq,
                    "avg_doc_length": self.avg_doc_length
                },
                f,
                indent=2,
                ensure_ascii=False
            )

    # ---------- LOAD ----------

    def load(self, path):

        with open(path, "r", encoding="utf-8") as f:

            data = json.load(f)

        self.documents = data["documents"]
        self.doc_freq = data["doc_freq"]
        self.avg_doc_length = data["avg_doc_length"]