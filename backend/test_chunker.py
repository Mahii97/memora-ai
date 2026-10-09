from app.ingestion.docx_parser import DocxParser
from app.chunking.smart_chunker import SmartChunker

parser = DocxParser()
doc = parser.parse("data/uploads/Amey_Kulkarni_Main.docx")

chunker = SmartChunker()
chunks = chunker.create_chunks(doc)

print("Total chunks:", len(chunks))

for c in chunks:
    if "language" in c.text.lower():
        print("\nFOUND IN CHUNK")
        print(c.location)
        print(c.text)