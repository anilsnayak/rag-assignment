# app/api/routes.py
"""
FastAPI router.

Endpoints:
  GET  /health                           – Liveness check
  POST /documents/upload                 – Upload a PDF
  GET  /documents                        – List all uploaded documents
  GET  /documents/{document_id}          – Get a single document's metadata
  POST /questions/ask                    – Ask a question (blocking)
  POST /questions/stream                 – Ask a question (streaming SSE)
  GET  /conversations                    – List active conversation sessions
  GET  /conversations/{conversation_id}  – Get conversation history
  DELETE /conversations/{conversation_id} – Delete a conversation session
"""

import logging
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import StreamingResponse

from app.config import settings
from app.core.exceptions import (
    DocumentNotFoundError,
    QuestionAnsweringError,
    UnsupportedFileError,
)
from app.models.schemas import (
    AskRequest,
    AskResponse,
    ConversationHistoryResponse,
    ConversationListResponse,
    DocumentListResponse,
    DocumentResponse,
    HealthResponse,
)
from app.services.conversation_service import get_conversation_service
from app.services.document_service import DocumentService
from app.services.qa_service import QAService

logger = logging.getLogger(__name__)

router = APIRouter()
document_service = DocumentService()
qa_service = QAService()


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness check",
    tags=["Health"],
)
def health() -> HealthResponse:
    """Returns the application status and environment."""
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.app_env,
    )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


@router.post(
    "/documents/upload",
    response_model=DocumentResponse,
    status_code=201,
    summary="Upload a PDF document",
    tags=["Documents"],
)
async def upload_document(file: UploadFile = File(...)) -> DocumentResponse:
    """
    Upload a PDF file. The system extracts text, chunks it, generates embeddings,
    and stores them in the vector database for later retrieval.

    Returns document metadata including the assigned `document_id`.
    """
    try:
        return await document_service.upload_document(file)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Document upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc


@router.get(
    "/documents",
    response_model=DocumentListResponse,
    summary="List all uploaded documents",
    tags=["Documents"],
)
def list_documents() -> DocumentListResponse:
    """Returns metadata for every uploaded document."""
    documents = document_service.list_documents()
    return DocumentListResponse(documents=documents)


@router.get(
    "/documents/{document_id}",
    response_model=DocumentResponse,
    summary="Get a single document's metadata",
    tags=["Documents"],
)
def get_document(document_id: str) -> DocumentResponse:
    """Returns metadata for the specified document."""
    try:
        metadata = document_service.get_document(document_id)
        return DocumentResponse(**metadata)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Question Answering
# ---------------------------------------------------------------------------


@router.post(
    "/questions/ask",
    response_model=AskResponse,
    summary="Ask a question (blocking)",
    tags=["Q&A"],
)
def ask_question(payload: AskRequest) -> AskResponse:
    """
    Ask a natural language question about an uploaded document.

    - Set `document_id` to restrict retrieval to a single document.
    - Set `conversation_id` to continue a multi-turn conversation.
    - Omit both to search across all uploaded documents and start a new session.

    The `grounded` field in the response indicates whether supporting evidence
    was found. If `grounded` is `false`, the answer reflects that the system
    could not locate relevant content in the uploaded document(s).
    """
    try:
        if payload.document_id:
            document_service.get_document(payload.document_id)
        return qa_service.ask(
            payload.question,
            payload.document_id,
            payload.conversation_id,
        )
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except QuestionAnsweringError as exc:
        raise HTTPException(
            status_code=500, detail=f"Question answering failed: {exc}"
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /questions/ask")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc


@router.post(
    "/questions/stream",
    summary="Ask a question (streaming SSE)",
    tags=["Q&A"],
    response_class=StreamingResponse,
)
def ask_question_stream(
    payload: AskRequest,
    document_id: Optional[str] = Query(None, description="Restrict to a specific document"),
    conversation_id: Optional[str] = Query(None, description="Continue an existing conversation"),
) -> StreamingResponse:
    """
    Streaming version of `/questions/ask`.

    Returns a `text/event-stream` (SSE) response. Each event delivers a token
    chunk from the LLM. The final event is `data: [DONE]`.

    Example event stream:
    ```
    data: The eligibility
    data:  criteria are
    data: ...
    data: [DONE]
    ```
    """
    try:
        if payload.document_id:
            document_service.get_document(payload.document_id)

        # Query params override body fields for flexibility.
        doc_id = document_id or payload.document_id
        conv_id = conversation_id or payload.conversation_id

        generator = qa_service.ask_stream(payload.question, doc_id, conv_id)
        return StreamingResponse(generator, media_type="text/event-stream")
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error in /questions/stream")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}") from exc


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


@router.get(
    "/conversations",
    response_model=ConversationListResponse,
    summary="List active conversation sessions",
    tags=["Conversations"],
)
def list_conversations() -> ConversationListResponse:
    """Returns a list of all active conversation session IDs."""
    svc = get_conversation_service()
    return ConversationListResponse(conversations=svc.list_sessions())


@router.get(
    "/conversations/{conversation_id}",
    response_model=ConversationHistoryResponse,
    summary="Get conversation history",
    tags=["Conversations"],
)
def get_conversation(conversation_id: str) -> ConversationHistoryResponse:
    """Returns the full message history for a conversation session."""
    svc = get_conversation_service()
    messages = svc.get_history_as_schemas(conversation_id)
    if not messages and conversation_id not in svc.list_sessions():
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found.",
        )
    return ConversationHistoryResponse(
        conversation_id=conversation_id,
        messages=messages,
    )


@router.delete(
    "/conversations/{conversation_id}",
    status_code=204,
    summary="Delete a conversation session",
    tags=["Conversations"],
)
def delete_conversation(conversation_id: str) -> None:
    """Deletes a conversation session and all its history."""
    svc = get_conversation_service()
    deleted = svc.delete_session(conversation_id)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Conversation '{conversation_id}' not found.",
        )
