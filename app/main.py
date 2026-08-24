import os
import time
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from app.rag import RAGService
from app.logging_config import configure_logging, get_logger

load_dotenv()
configure_logging()
logger = get_logger("onboardai.api")

rag = RAGService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup_begin")
    started = time.perf_counter()
    await rag.initialize()
    logger.info(
        "startup_complete",
        extra={"duration_ms": round((time.perf_counter() - started) * 1000, 2)}
    )
    yield

app = FastAPI(
    title="OnboardAI RAG Assistant",
    description="FastAPI wrapper around the OnboardAI RAG assignment over CPython documentation.",
    version="1.0.0",
    lifespan=lifespan,
)

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=8)

@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())
    started = time.perf_counter()
    try:
        response = await call_next(request)
        return response
    finally:
        logger.info(
            "http_request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": getattr(locals().get("response"), "status_code", 500),
                "duration_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )

@app.get("/")
def root():
    return {
        "service": "OnboardAI RAG Assistant",
        "docs": "/docs",
        "health": "/health",
        "query_endpoint": "POST /query",
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "ready": rag.ready,
        "documents": rag.document_count,
        "chunks": rag.chunk_count,
        "chunking_strategy": rag.chunking_strategy,
    }

@app.post("/query")
def query(payload: QueryRequest):
    started = time.perf_counter()
    result = rag.answer(payload.question, top_k=payload.top_k)
    logger.info(
        "rag_query",
        extra={
            "question": payload.question,
            "answer_status": result["status"],
            "retrieved_count": len(result["sources"]),
            "duration_ms": round((time.perf_counter() - started) * 1000, 2),
        },
    )
    return result
