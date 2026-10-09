class Encoder:

    def __init__(self, vocabulary):

        self.vocab = vocabulary

    def encode(self, tokens):

        ids = []

        for token in tokens:

            ids.append(
                self.vocab.word_to_id.get(
                    token,
                    1
                )
            )

        return ids

    def decode(self, ids):

        words = []

        for i in ids:

            words.append(
                self.vocab.id_to_word.get(
                    i,
                    "<UNK>"
                )
            )

        return words