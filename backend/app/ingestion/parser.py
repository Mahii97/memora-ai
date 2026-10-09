import fitz
import uuid
import os
from datetime import datetime

try:
    import pytesseract
    from PIL import Image
except ImportError:  # OCR is an optional fallback; text extraction still works.
    pytesseract = None
    Image = None

from app.models.source_document import SourceDocument, SourcePage
from app.ingestion.cleaner import TextCleaner


# Tesseract OCR executable
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if pytesseract is not None:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


class PDFParser:

    def _extract_ocr_text(self, page):
        """
        Render a PDF page as an image and extract text using Tesseract OCR.
        """

        if pytesseract is None or Image is None:
            raise RuntimeError("OCR dependencies are not installed")

        # Render page at higher resolution for better OCR accuracy
        matrix = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=matrix, alpha=False)

        # Convert PyMuPDF pixmap to PIL image
        image = Image.frombytes(
            "RGB",
            [pix.width, pix.height],
            pix.samples
        )

        # Run OCR
        text = pytesseract.image_to_string(
            image,
            lang="eng"
        )

        return text

    def parse(self, filepath):

        pdf = fitz.open(filepath)
        pages = []

        for page_number, page in enumerate(pdf, start=1):

            # --------------------------------------------------
            # 1. Try normal PDF text extraction first
            # --------------------------------------------------

            raw_text = page.get_text("text")
            clean_text = TextCleaner.clean(raw_text)

            extraction_method = "text"

            # --------------------------------------------------
            # 2. If no text exists, use OCR
            # --------------------------------------------------

            if not clean_text:

                try:
                    raw_text = self._extract_ocr_text(page)
                    clean_text = TextCleaner.clean(raw_text)
                    extraction_method = "ocr"

                except Exception as exc:
                    print(
                        f"OCR failed on page {page_number}: {exc}"
                    )
                    clean_text = ""

            # --------------------------------------------------
            # 3. Store page
            # --------------------------------------------------

            pages.append(
                SourcePage(
                    location=f"Page {page_number} ({extraction_method})",
                    text=clean_text
                )
            )

        pdf.close()

        return SourceDocument(
            id=str(uuid.uuid4()),
            source_type="pdf",
            source_name=os.path.basename(filepath),
            author="Unknown",
            created_at=datetime.now(),
            pages=pages
        )
