import json

with open("data/tokenized_chunks.json", "r", encoding="utf-8") as f:
    data = json.load(f)

found = False

for chunk in data:
    text = chunk["text"].lower()

    if "Language" in text or "Languages" in text:
        print("=" * 60)
        print("Location :", chunk["location"])
        print("Source   :", chunk["source_name"])
        print("Tokens   :", chunk["tokens"])
        print("\nText:")
        print(chunk["text"])
        found = True

if not found:
    print("No language section found.")