from fastapi import APIRouter
from pydantic import BaseModel

from app.search_resources import search_resources

router = APIRouter(prefix="/timeline", tags=["Timeline"])

class DateRequest(BaseModel):
    date: str


class AuthorRequest(BaseModel):
    author: str


class SourceRequest(BaseModel):
    source: str


@router.post("/date")
async def search_by_date(data: DateRequest):
    search = search_resources.get_timeline_search()
    results = search.by_date(data.date)
    return {
        "query_date": data.date,
        "count": len(results),
        "results": results
    }


@router.post("/author")
async def search_by_author(data: AuthorRequest):
    search = search_resources.get_timeline_search()
    results = search.by_author(data.author)
    return {
        "author": data.author,
        "count": len(results),
        "results": results
    }


@router.post("/source")
async def search_by_source(data: SourceRequest):
    search = search_resources.get_timeline_search()
    results = search.by_source(data.source)
    return {
        "source": data.source,
        "count": len(results),
        "results": results
    }