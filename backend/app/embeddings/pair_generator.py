import json

WINDOW = 2

class PairGenerator:

    def generate(self, tokenized_dataset):

        pairs = []

        for chunk in tokenized_dataset:

            ids = chunk["token_ids"]

            for i in range(len(ids)):

                target = ids[i]

                left = max(0, i - WINDOW)
                right = min(len(ids), i + WINDOW + 1)

                for j in range(left, right):

                    if i == j:
                        continue

                    context = ids[j]

                    pairs.append({
                        "target": target,
                        "context": context
                    })

        return pairs

    def save(self, pairs, path):

        with open(path, "w") as f:
            json.dump(pairs, f, indent=4)