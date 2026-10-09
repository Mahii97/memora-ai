import os
import uuid
import hashlib
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException

from app.ingestion.parser_factory import ParserFactory
from app.database.repository import DocumentRepository
from app.chunking.smart_chunker import SmartChunker
from app.pipeline.index_pipeline import IndexPipeline
from app.search_resources import search_resources


router = APIRouter()

UPLOAD_DIR = "data/uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

MAX_UPLOAD_BYTES = 100 * 1024 * 1024

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".txt",
    ".txtwa"
}


def validate_upload(filename):

    if not filename:
        raise HTTPException(
            status_code=400,
            detail="A filename is required."
        )

    display_name = Path(
        filename.replace("\\", "/")
    ).name

    extension = Path(
        display_name
    ).suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type."
        )

    return display_name, extension


def calculate_content_hash(chunks):

    content_for_hash = "\n".join(
        chunk.text or ""
        for chunk in chunks
    )

    return hashlib.sha256(
        content_for_hash.encode("utf-8")
    ).hexdigest()


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    display_name, extension = validate_upload(file.filename)

    filepath = os.path.join(
        UPLOAD_DIR,
        f"{uuid.uuid4().hex}{extension}"
    )

    document_persisted = False

    try:

        # --------------------------------------------------
        # 1. Save uploaded file temporarily
        # --------------------------------------------------

        size = 0

        with open(filepath, "xb") as buffer:

            while True:

                content = await file.read(
                    1024 * 1024
                )

                if not content:
                    break

                size += len(content)

                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail="File exceeds the 100 MB upload limit."
                    )

                buffer.write(content)

        # --------------------------------------------------
        # 2. Parse document
        # --------------------------------------------------

        factory = ParserFactory()

        parser = factory.get_parser(
            display_name
        )

        document = parser.parse(filepath)

        document.source_name = display_name

        # --------------------------------------------------
        # 3. Create chunks
        # --------------------------------------------------

        chunker = SmartChunker()

        chunks = chunker.create_chunks(
            document
        )

        # --------------------------------------------------
        # 4. Calculate content-based hash
        # --------------------------------------------------

        file_hash = calculate_content_hash(
            chunks
        )

        repo = DocumentRepository()

        # --------------------------------------------------
        # 5. Check for exact duplicate content
        # --------------------------------------------------

        existing_document = repo.find_document_by_hash(
            file_hash
        )

        if existing_document:

            old_document_id = existing_document[0]

            repo.delete_document(
                old_document_id
            )

        # --------------------------------------------------
        # 6. Save document
        # --------------------------------------------------

        repo.save_document(
            document,
            file_hash
        )

        document_persisted = True

        # --------------------------------------------------
        # 7. Save extracted pages
        # --------------------------------------------------

        repo.save_pages(
            document
        )

        # --------------------------------------------------
        # 8. Save chunks
        # --------------------------------------------------

        repo.save_chunks(
            chunks
        )

        # --------------------------------------------------
        # 9. Rebuild indexes
        # --------------------------------------------------

        all_chunks = repo.get_all_chunks()

        index_summary = IndexPipeline.rebuild(
            all_chunks
        )

        search_resources.invalidate()

        # --------------------------------------------------
        # 10. Response
        # --------------------------------------------------

        return {
            "status": "success",
            "document_id": document.id,
            "source_type": document.source_type,
            "source_name": document.source_name,
            "records": len(document.pages),
            "chunks_created": len(chunks),
            "tokenized_chunks": len(chunks),
            "indexed_chunks": index_summary["chunk_count"],
            "duplicate_replaced": bool(existing_document)
        }

    except HTTPException:

        if (
            not document_persisted
            and os.path.exists(filepath)
        ):
            os.remove(filepath)

        raise

    except ValueError as error:

        if (
            not document_persisted
            and os.path.exists(filepath)
        ):
            os.remove(filepath)

        raise HTTPException(
            status_code=422,
            detail=str(error)
        ) from error

    except Exception as error:

        if (
            not document_persisted
            and os.path.exists(filepath)
        ):
            os.remove(filepath)

        raise HTTPException(
            status_code=500,
            detail="File processing failed."
        ) from error