import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.database.database import initialize_database

from app.api.upload import router
from app.api.ask import router as ask_router
from app.api.timeline import router as timeline_router
from app.api.frontend import router as frontend_router

initialize_database()

app = FastAPI(title="MemoraAI API")

origins = [
    item.strip()
    for item in os.getenv(
        "FRONTEND_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
    ).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(router)
app.include_router(ask_router)
app.include_router(timeline_router)
app.include_router(frontend_router)
