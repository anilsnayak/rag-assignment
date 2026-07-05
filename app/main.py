# app/main.py
"""
Application entry point.

- Configures structured logging.
- Registers a lifespan handler that logs startup/shutdown events.
- Injects an X-Request-ID header into every response for traceability.
- Mounts the API router.
- Serves the static UI at /ui (root / redirects to /ui).
"""

import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.config import settings
from app.logging_config import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ANN001
    """Handle application startup and shutdown events."""
    logger.info(
        "Starting %s v%s | env=%s | llm=%s | model=%s",
        settings.app_name,
        "1.0.0",
        settings.app_env,
        settings.llm_provider,
        settings.ollama_model,
    )
    yield
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description=(
        "REST API for uploading PDF documents and performing grounded question answering. "
        "Answers are derived strictly from the uploaded content. "
        "Supports multi-turn conversation history and streaming responses."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS middleware
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request-ID middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Attach a unique X-Request-ID to every request/response for traceability."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal error occurred."},
    )


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
app.include_router(router)

# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root to the web UI."""
    return RedirectResponse(url="/ui/index.html")
