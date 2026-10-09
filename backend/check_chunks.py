from app.ingestion.docx_parser import DocxParser
from app.chunking.smart_chunker import SmartChunker

doc = DocxParser().parse("data/uploads/Amey_Kulkarni_Main.docx")
chunks = SmartChunker().create_chunks(doc)

for c in chunks:
    if "language" in c.text.lower():
        print(c.location)
        print(c.text)
        