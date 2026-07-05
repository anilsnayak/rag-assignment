# app/models/schemas.py
from typing import List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Source / Citation
# ---------------------------------------------------------------------------


class SourceItem(BaseModel):
    """A single source chunk used to derive the answer."""

    page: int = Field(..., description="Page number from the source PDF")
    document_id: Optional[str] = Field(None, description="ID of the source document")
    filename: Optional[str] = Field(None, description="Original filename of the source document")
    chunk_id: Optional[str] = None
    snippet: Optional[str] = Field(None, description="Short excerpt from the matched chunk")


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class DocumentResponse(BaseModel):
    """Returned after a successful document upload or in document list."""

    document_id: str
    filename: str
    uploaded_at: Optional[str] = None
    num_pages: Optional[int] = None
    num_chunks: Optional[int] = None


class DocumentItem(BaseModel):
    """Lightweight document listing item."""

    document_id: str
    filename: str
    uploaded_at: Optional[str] = None
    num_pages: Optional[int] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]


# ---------------------------------------------------------------------------
# Question Answering
# ---------------------------------------------------------------------------


class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question to ask")
    document_id: Optional[str] = Field(
        None,
        description="Restrict retrieval to a specific document. Omit to search all documents.",
    )
    conversation_id: Optional[str] = Field(
        None,
        description="Session ID for multi-turn conversation history. Omit to start a new session.",
    )


class AskResponse(BaseModel):
    answer: str
    grounded: bool = Field(
        ...,
        description="True if the answer is supported by retrieved document content; False if the system could not find supporting evidence.",
    )
    sources: List[SourceItem] = Field(default_factory=list)
    conversation_id: Optional[str] = Field(
        None,
        description="Session ID that can be passed in subsequent requests to continue the conversation.",
    )


# ---------------------------------------------------------------------------
# Conversation History
# ---------------------------------------------------------------------------


class ConversationMessage(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str


class ConversationHistoryResponse(BaseModel):
    conversation_id: str
    messages: List[ConversationMessage]


class ConversationListResponse(BaseModel):
    conversations: List[str] = Field(
        ..., description="List of active conversation / session IDs"
    )


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
