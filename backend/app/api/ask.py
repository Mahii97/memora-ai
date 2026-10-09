from fastapi import APIRouter
from pydantic import BaseModel

from app.search_resources import search_resources

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str


def load_resources():
    return search_resources.get_rag_service()


def encode_query(text: str):
    return search_resources.embedding_service.embed_query(text)


@router.post("/ask")
async def ask(data: QuestionRequest):

    load_resources()

    rag_service = load_resources()
    if rag_service is None:
        return {
            "status": "error",
            "message": "Vector index not built."
        }

    return rag_service.ask(data.question).model_dump()