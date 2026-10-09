import json

from app.tokenizer.tokenizer import Tokenizer
from app.tokenizer.vocabulary import Vocabulary
from app.tokenizer.encoder import Encoder


class DatasetBuilder:

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.vocab = Vocabulary()

    def build(self, chunks):

        tokenized_documents = []

        # Tokenize every chunk
        for chunk in chunks:

            tokens = self.tokenizer.tokenize(chunk.text)

            tokenized_documents.append(tokens)

        # Build vocabulary
        self.vocab.build(tokenized_documents)

        encoder = Encoder(self.vocab)

        dataset = []

        # Preserve metadata
        for chunk, tokens in zip(chunks, tokenized_documents):

            ids = encoder.encode(tokens)

            dataset.append({

                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,

                "source_type": chunk.source_type,
                "source_name": chunk.source_name,

                "author": chunk.author,
                "created_at": chunk.created_at,

                "location": chunk.location,

                "text": chunk.text,

                "tokens": tokens,
                "token_ids": ids

            })

        return dataset

    def save_dataset(self, dataset, path):

        with open(path, "w") as f:

            json.dump(dataset, f, indent=4)

    def save_vocabulary(self, path):

        self.vocab.save(path)