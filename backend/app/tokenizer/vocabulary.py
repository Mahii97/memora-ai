from collections import Counter
import json

class Vocabulary:

    def __init__(self):
        self.word_to_id = {
            "<PAD>": 0,
            "<UNK>": 1
        }
        self.id_to_word = {
            0: "<PAD>",
            1: "<UNK>"
        }

    def build(self, tokenized_texts):

        next_id = 2

        for tokens in tokenized_texts:
            for token in tokens:

                if token not in self.word_to_id:

                    self.word_to_id[token] = next_id
                    self.id_to_word[next_id] = token

                    next_id += 1

    def get_id(self, word):
        return self.word_to_id.get(word, 1)

    def save(self, path):

        with open(path, "w") as f:
            json.dump(self.word_to_id, f, indent=4)