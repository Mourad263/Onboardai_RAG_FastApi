import asyncio
import time
import uuid
from contextlib import asynccontextmanager, suppress

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.logging_config import configure_logging, get_logger
from app.rag import RAGService


load_dotenv()

configure_logging()

logger = get_logger("onboardai.api")

rag = RAGService()


async def initialize_rag_in_background():
    """
    Run the existing async RAG initialization in a background thread.

    The current RAGService.initialize() performs heavy synchronous work
    (downloads, embeddings, FAISS, etc.). Running it in a thread prevents
    FastAPI's event loop from being blocked during startup.
    """
    try:
        started = time.perf_counter()

        # RAGService.initialize() is currently async, but its body performs
        # synchronous/heavy operations. Run it in its own thread.
        await asyncio.to_thread(
            lambda: asyncio.run(rag.initialize())
        )

        logger.info(
            "startup_complete",
            extra={
                "duration_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                )
            },
        )

    except Exception:
        logger.exception("rag_initialization_failed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup_begin")

    # Start RAG initialization in the background.
    app.state.rag_init_task = asyncio.create_task(
        initialize_rag_in_background()
    )

    # IMPORTANT:
    # Do not wait for RAG initialization here.
    # FastAPI can start accepting requests immediately.
    logger.info("startup_ready")

    yield

    # Graceful shutdown.
    task = getattr(app.state, "rag_init_task", None)

    if task and not task.done():
        task.cancel()

        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="OnboardAI RAG Assistant",
    description=(
        "FastAPI wrapper around the OnboardAI RAG assignment "
        "over CPython documentation."
    ),
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
    response = None

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
                "status_code": getattr(response, "status_code", 500),
                "duration_ms": round(
                    (time.perf_counter() - started) * 1000,
                    2,
                ),
            },
        )


@app.get("/")
def root():
    return {
        "service": "OnboardAI RAG Assistant",
        "docs": "/docs",
        "health": "/health",
        "ready": "/ready",
        "status": "/status",
        "query_endpoint": "POST /query",
    }


@app.get("/health")
def health():
    """
    Liveness endpoint.

    This should stay fast and return 200 even while the RAG is
    still initializing. Use this endpoint for Railway health checks.
    """
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """
    Readiness endpoint.

    Returns 503 until the RAG initialization has completed.
    """
    if not rag.ready:
        return JSONResponse(
            status_code=503,
            content={
                "status": "starting",
                "ready": False,
            },
        )

    return {
        "status": "ready",
        "ready": True,
    }


@app.get("/status")
def status():
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

    result = rag.answer(
        payload.question,
        top_k=payload.top_k,
    )

    logger.info(
        "rag_query",
        extra={
            "question": payload.question,
            "answer_status": result["status"],
            "retrieved_count": len(result["sources"]),
            "duration_ms": round(
                (time.perf_counter() - started) * 1000,
                2,
            ),
        },
    )

    return result