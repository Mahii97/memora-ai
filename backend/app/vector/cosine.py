import numpy as np

class CosineSimilarity:

    @staticmethod
    def score(v1, v2):

        numerator = np.dot(v1, v2)

        denominator = (
            np.linalg.norm(v1)
            * np.linalg.norm(v2)
        )

        if denominator == 0:
            return 0

        return float(numerator / denominator)