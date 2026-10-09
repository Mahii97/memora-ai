import pickle
import numpy as np

class VectorIndex:

    def __init__(self):

        self.embeddings = None
        self.metadata = []

    def build(self, embeddings, metadata):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2:
            raise ValueError("embeddings must be a two-dimensional array")
        if embeddings.shape[0] != len(metadata):
            raise ValueError("embedding count must match metadata count")

        self.embeddings = embeddings
        self.metadata = metadata

    def save(self, path):

        with open(path, "wb") as f:

            pickle.dump(
                {
                    "embeddings": self.embeddings,
                    "metadata": self.metadata
                },
                f
            )

    def load(self, path):

        with open(path, "rb") as f:

            data = pickle.load(f)

        self.embeddings = data["embeddings"]
        self.metadata = data["metadata"]