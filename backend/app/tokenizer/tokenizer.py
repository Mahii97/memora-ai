import re

class Tokenizer:

    def __init__(self):
        self.pattern = re.compile(r"[a-zA-Z0-9]+")

    def tokenize(self, text: str):

        text = text.lower()

        tokens = self.pattern.findall(text)

        return tokens