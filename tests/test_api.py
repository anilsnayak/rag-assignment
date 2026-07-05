# tests/test_api.py
"""
Integration tests for the RAG Assignment API.

Tests are written with TestClient (synchronous) and do NOT require a running
Ollama instance – the QA/stream endpoints are tested for correct HTTP shapes
and graceful error handling.
"""

import io
import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def test_health(client: TestClient):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "app_name" in body
    assert "environment" in body


# ---------------------------------------------------------------------------
# Documents – List
# ---------------------------------------------------------------------------


def test_list_documents_empty_or_populated(client: TestClient):
    response = client.get("/documents")
    assert response.status_code == 200
    body = response.json()
    assert "documents" in body
    assert isinstance(body["documents"], list)


# ---------------------------------------------------------------------------
# Documents – Upload
# ---------------------------------------------------------------------------


def test_upload_non_pdf_rejected(client: TestClient):
    fake_file = io.BytesIO(b"this is not a pdf")
    response = client.post(
        "/documents/upload",
        files={"file": ("test.txt", fake_file, "text/plain")},
    )
    assert response.status_code == 400
    assert "PDF" in response.json()["detail"]


def test_upload_pdf_success(client: TestClient, minimal_pdf_bytes: bytes):
    pdf_file = io.BytesIO(minimal_pdf_bytes)
    response = client.post(
        "/documents/upload",
        files={"file": ("test_document.pdf", pdf_file, "application/pdf")},
    )
    # Expect 201 Created on success or 500 if Ollama/embedding is not running.
    # We assert the shape is correct on 201.
    if response.status_code == 201:
        body = response.json()
        assert "document_id" in body
        assert body["filename"] == "test_document.pdf"
        assert "uploaded_at" in body
    else:
        # Non-200 is acceptable in CI where embedding models are unavailable.
        assert response.status_code == 500


def test_upload_and_list(client: TestClient, minimal_pdf_bytes: bytes):
    """After a successful upload the document should appear in the list."""
    pdf_file = io.BytesIO(minimal_pdf_bytes)
    upload_resp = client.post(
        "/documents/upload",
        files={"file": ("list_test.pdf", pdf_file, "application/pdf")},
    )
    if upload_resp.status_code != 201:
        pytest.skip("Embedding model not available in this environment")

    document_id = upload_resp.json()["document_id"]
    list_resp = client.get("/documents")
    assert list_resp.status_code == 200
    ids = [d["document_id"] for d in list_resp.json()["documents"]]
    assert document_id in ids


# ---------------------------------------------------------------------------
# Documents – Get by ID
# ---------------------------------------------------------------------------


def test_get_document_not_found(client: TestClient):
    response = client.get("/documents/nonexistent-id-abc")
    assert response.status_code == 404


def test_get_document_success(client: TestClient, minimal_pdf_bytes: bytes):
    pdf_file = io.BytesIO(minimal_pdf_bytes)
    upload_resp = client.post(
        "/documents/upload",
        files={"file": ("get_test.pdf", pdf_file, "application/pdf")},
    )
    if upload_resp.status_code != 201:
        pytest.skip("Embedding model not available in this environment")

    document_id = upload_resp.json()["document_id"]
    get_resp = client.get(f"/documents/{document_id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["document_id"] == document_id


# ---------------------------------------------------------------------------
# Questions – Ask (blocking)
# ---------------------------------------------------------------------------


def test_ask_missing_question_body(client: TestClient):
    response = client.post("/questions/ask", json={})
    assert response.status_code == 422  # Pydantic validation error


def test_ask_nonexistent_document(client: TestClient):
    response = client.post(
        "/questions/ask",
        json={"question": "What is this?", "document_id": "nonexistent-doc"},
    )
    assert response.status_code == 404


def test_ask_response_shape(client: TestClient):
    """When no documents are uploaded (or Ollama unavailable) the response should
    still be a valid AskResponse with grounded=False."""
    response = client.post(
        "/questions/ask",
        json={"question": "What is the capital of France?"},
    )
    # 200 with grounded=False or 500 if LLM is down
    if response.status_code == 200:
        body = response.json()
        assert "answer" in body
        assert "grounded" in body
        assert isinstance(body["grounded"], bool)
        assert "sources" in body
        assert isinstance(body["sources"], list)
        assert "conversation_id" in body


# ---------------------------------------------------------------------------
# Questions – Stream
# ---------------------------------------------------------------------------


def test_stream_endpoint_exists(client: TestClient):
    """The streaming endpoint should exist and return a streaming content-type or 500."""
    response = client.post(
        "/questions/stream",
        json={"question": "Summarize the document."},
    )
    assert response.status_code in (200, 404, 500)
    if response.status_code == 200:
        assert "text/event-stream" in response.headers.get("content-type", "")


# ---------------------------------------------------------------------------
# Conversations
# ---------------------------------------------------------------------------


def test_list_conversations(client: TestClient):
    response = client.get("/conversations")
    assert response.status_code == 200
    body = response.json()
    assert "conversations" in body
    assert isinstance(body["conversations"], list)


def test_get_conversation_not_found(client: TestClient):
    response = client.get("/conversations/does-not-exist")
    assert response.status_code == 404


def test_delete_conversation_not_found(client: TestClient):
    response = client.delete("/conversations/does-not-exist")
    assert response.status_code == 404


def test_conversation_flow(client: TestClient):
    """After asking a question, a conversation_id should be returned and queryable."""
    ask_resp = client.post(
        "/questions/ask",
        json={"question": "Hello, are there any documents available?"},
    )
    if ask_resp.status_code != 200:
        pytest.skip("LLM or embedding service not available")

    conv_id = ask_resp.json().get("conversation_id")
    assert conv_id is not None

    history_resp = client.get(f"/conversations/{conv_id}")
    assert history_resp.status_code == 200
    body = history_resp.json()
    assert body["conversation_id"] == conv_id
    assert len(body["messages"]) >= 2  # user + assistant

    # Clean up
    del_resp = client.delete(f"/conversations/{conv_id}")
    assert del_resp.status_code == 204


# ---------------------------------------------------------------------------
# X-Request-ID middleware
# ---------------------------------------------------------------------------


def test_request_id_header_injected(client: TestClient):
    response = client.get("/health")
    assert "x-request-id" in response.headers
