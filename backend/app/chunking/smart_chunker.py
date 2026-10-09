import uuid
from app.models.chunk_model import Chunk


class SmartChunker:

    def __init__(self, chunk_size=500, overlap=100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def create_chunks(self, document):

        chunks = []

        for page in document.pages:

            text = page.text

            start = 0
            index = 1

            while start < len(text):

                end = min(start + self.chunk_size, len(text))

                chunk_text = text[start:end]

                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document.id,
                        source_type=document.source_type,
                        source_name=document.source_name,
                        author=document.author,
                        created_at=str(document.created_at),
                        location=page.location,
                        chunk_index=index,
                        text=chunk_text,
                        start_char=start,
                        end_char=end
                    )
                )

                index += 1

                start += self.chunk_size - self.overlap

        return chunks