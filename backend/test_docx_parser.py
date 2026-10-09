from app.ingestion.docx_parser import DocxParser

parser = DocxParser()

doc = parser.parse("data/uploads/Amey_Kulkarni_Main.docx")

print("Total records:", len(doc.pages))
print()

for page in doc.pages:
    if "language" in page.text.lower():
        print("FOUND")
        print("Location:", page.location)
        print("Text:", page.text)