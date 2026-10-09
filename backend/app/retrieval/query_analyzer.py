import re
from datetime import datetime

class QueryAnalyzer:

    def analyze(self, query):

        query = query.lower()

        metadata = {
            "date": None,
            "source_type": None,
            "author": None
        }

        # YYYY-MM-DD
        match = re.search(r"\d{4}-\d{2}-\d{2}", query)
        if match:
            metadata["date"] = match.group()

        # Source detection
        if "pdf" in query:
            metadata["source_type"] = "pdf"

        elif "docx" in query or "resume" in query:
            metadata["source_type"] = "docx"

        elif "whatsapp" in query:
            metadata["source_type"] = "whatsapp"

        # Author
        author = re.search(r"from ([a-zA-Z ]+)", query)

        if author:
            metadata["author"] = author.group(1).strip()

        return metadata