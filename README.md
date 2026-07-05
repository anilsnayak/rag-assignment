# RAG Assignment API

A production-oriented **FastAPI** application that lets users upload PDF documents and ask natural-language questions **strictly grounded in the uploaded content**.

[![CI](https://github.com/anilsnayak/rag-assignment/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/rag-assignment/actions/workflows/ci.yml)

---

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) pipeline as a REST API:

1. **Upload** a PDF → text is extracted, chunked, embedded, and stored in a local vector database.
2. **Ask** a natural-language question → the most relevant chunks are retrieved, passed to an LLM with strict instructions, and an answer is returned together with **source citations** (page numbers).
3. The system **refuses** to answer when the evidence is not present in the document, instead of hallucinating.

### Bonus features implemented
| Feature | Endpoint |
|---|---|
| Multi-turn conversation history | `conversation_id` field in ask/response |
| Streaming responses (SSE) | `POST /questions/stream` |
| Source citations with page numbers | `sources[].page` in every response |
| Multiple uploaded documents | Filter by `document_id` or search all |
| Unit tests | `tests/` |
| Structured logging | `app/logging_config.py` |
| Docker Compose | `docker-compose.yml` |
| Environment-based config | `.env` / `pydantic-settings` |
| CI/CD (GitHub Actions) | `.github/workflows/ci.yml` |

---

## Architecture

```
┌─────────────┐      HTTP       ┌───────────────────────────────┐
│   Client    │ ─────────────▶ │         FastAPI (uvicorn)      │
└─────────────┘                └───────────┬───────────────────┘
                                           │
            ┌──────────────────────────────┼────────────────────┐
            ▼                              ▼                     ▼
   DocumentService                    QAService         ConversationService
  (upload, list, get)          (retrieve + generate)    (history store)
            │                              │
            ▼                              ▼
   pypdf + LangChain          ChromaDB ◀── HuggingFace     JSON on disk
   text splitter               (vector     Embeddings      (persisted
   + ChromaDB ingest            store)    (local)          sessions)
                                           │
                                           ▼
                                      Ollama LLM
                                   (local inference)
```

### Design decisions

| Choice | Rationale |
|--------|-----------|
| **FastAPI** | First-class async support, automatic OpenAPI docs, Pydantic validation |
| **ChromaDB** | Local-first, persistent, easy to set up; no separate service required |
| **Sentence Transformers** | Fully local embeddings; no API key required |
| **Ollama** | Run any LLM locally; avoids vendor lock-in and external API costs |
| **Strict prompting** | System prompt explicitly forbids answers outside the context to prevent hallucination |
| **`grounded` flag** | Consumers can distinguish confident answers from "I don't know" responses programmatically |
| **JSON-backed conversation store** | Conversations persist across server restarts without a database dependency |

---

## Project Structure

```
rag-assignment/
├── app/
│   ├── api/
│   │   └── routes.py           # All HTTP endpoints
│   ├── core/
│   │   ├── exceptions.py       # Custom exception types
│   │   └── prompts.py          # LLM system prompt
│   ├── models/
│   │   └── schemas.py          # Pydantic request/response models
│   ├── services/
│   │   ├── conversation_service.py  # Multi-turn history
│   │   ├── document_service.py      # Upload + metadata management
│   │   ├── qa_service.py            # Retrieval + LLM call + streaming
│   │   └── vector_store.py          # ChromaDB singleton
│   ├── utils/
│   │   └── pdf_loader.py       # pypdf extraction + chunking
│   ├── config.py               # pydantic-settings configuration
│   ├── logging_config.py       # Structured logging setup
│   └── main.py                 # FastAPI app + middleware
├── data/
│   ├── chroma/                 # ChromaDB persistence
│   ├── documents/              # Uploaded PDF files
│   └── metadata/               # Document + conversation JSON metadata
├── tests/
│   ├── conftest.py             # Shared fixtures
│   └── test_api.py             # Integration tests
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions CI/CD
├── .env.example                # Environment variable template
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server |
| `pydantic-settings` | Config from environment variables |
| `langchain` + `langchain-chroma` | RAG pipeline orchestration |
| `chromadb` | Vector database |
| `sentence-transformers` | Local embedding model |
| `pypdf` | PDF text extraction |
| `ollama` | Local LLM client |
| `pytest` + `pytest-asyncio` | Testing |

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- The chosen model pulled locally (default: `qwen3:4b`)

```bash
# Pull the LLM (first time only)
ollama pull qwen3:4b
```

### Option A — Local (without Docker)

```bash
# 1. Clone the repository
git clone <repo-url>
cd rag-assignment

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip3 install -r requirements.txt

# 4. Configure environment variables
cp ".env.example " .env
# Edit .env as needed (Ollama URL, model, etc.)

# 5. Start the API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now at `http://localhost:8000` using a simple UI.  
Interactive docs: `http://localhost:8000/docs`

### Option B — Docker Compose (recommended)

```bash
# 1. Build and start
docker compose up --build

# 2. To stop
docker compose down
```

> **Note:** Make sure Ollama is reachable from inside the container.  
> Set `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env` when running Docker on macOS/Windows.

---

## API Documentation

All endpoints and schemas are available interactively at `/docs` (Swagger UI) or `/redoc`.

### Endpoints Summary

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/documents/upload` | Upload a PDF document |
| `GET` | `/documents` | List all uploaded documents |
| `GET` | `/documents/{document_id}` | Get a single document's metadata |
| `POST` | `/questions/ask` | Ask a question (blocking) |
| `POST` | `/questions/stream` | Ask a question (streaming SSE) |
| `GET` | `/conversations` | List active conversation sessions |
| `GET` | `/conversations/{id}` | Get conversation history |
| `DELETE` | `/conversations/{id}` | Delete a conversation session |

### Upload a PDF

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@/path/to/document.pdf"
```

**Response** (`201 Created`):
```json
{
  "document_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "filename": "document.pdf",
  "uploaded_at": "2026-07-05T10:00:00Z",
  "num_pages": 12,
  "num_chunks": 48
}
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/questions/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the eligibility criteria?", "document_id": "<id>"}'
```

**Response** (`200 OK`):
```json
{
  "answer": "The eligibility criteria are...",
  "grounded": true,
  "sources": [
    { "page": 4, "document_id": "...", "filename": "document.pdf", "snippet": "..." }
  ],
  "conversation_id": "abc123-..."
}
```

**When the answer is not in the document:**
```json
{
  "answer": "I cannot answer this from the provided document(s).",
  "grounded": false,
  "sources": [],
  "conversation_id": "..."
}
```

### Multi-turn Conversation

```bash
# First question — no conversation_id needed
curl -X POST http://localhost:8000/questions/ask \
  -d '{"question": "What is the purpose of this document?"}'
# → returns conversation_id: "abc123"

# Follow-up question — pass the returned conversation_id
curl -X POST http://localhost:8000/questions/ask \
  -d '{"question": "Can you elaborate on section 2?", "conversation_id": "abc123"}'
```

### Streaming (SSE)

```bash
curl -N -X POST http://localhost:8000/questions/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the document."}'
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `APP_NAME` | `RAG Assignment API` | Application name |
| `APP_ENV` | `development` | Environment label |
| `APP_PORT` | `8000` | Server port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `CHROMA_PERSIST_DIRECTORY` | `./data/chroma` | ChromaDB storage path |
| `DOCUMENTS_DIRECTORY` | `./data/documents` | Uploaded PDF storage |
| `METADATA_DIRECTORY` | `./data/metadata` | Document + conversation JSON |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `LLM_PROVIDER` | `ollama` | LLM backend |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `qwen3:4b` | LLM model name |
| `CHUNK_SIZE` | `1200` | Token chunk size |
| `CHUNK_OVERLAP` | `200` | Chunk overlap |
| `TOP_K` | `4` | Number of retrieved chunks |
| `MAX_CONTEXT_CHUNKS` | `6` | Max chunks sent to LLM |

---

## Assumptions Made

1. **Text-based PDFs only** – scanned PDFs (image-only) are accepted but will yield empty extraction. Adding OCR (e.g., `pytesseract`) would be a straightforward extension.
2. **Ollama must be running** before the API starts. The application will start successfully but answer requests will fail until Ollama is available.
3. **Conversation history is server-side** – the `conversation_id` is a server-generated UUID. Clients only need to echo it back in subsequent requests.
4. **No authentication** – the API is open. See future improvements below.
5. **Single-node deployment** – conversation history and document files are stored locally. A distributed deployment would require shared storage and a shared vector database.

---

## Future Improvements

- [ ] **Authentication / API keys** – protect endpoints with Bearer token or API key middleware.
- [ ] **OCR support** – use `pytesseract` or `pdf2image` to handle scanned PDFs.
- [ ] **Async LLM calls** – use `ollama`'s async client to avoid blocking the event loop.
- [ ] **Hybrid search** – combine BM25 keyword search with vector similarity for better retrieval.
- [ ] **Re-ranking** – add a cross-encoder re-ranker step after retrieval to improve answer quality.
- [ ] **Distributed conversation store** – replace JSON files with Redis or PostgreSQL.
- [ ] **Cloud deployment** – Dockerize with a managed vector DB (e.g., Pinecone) and deploy on GCP/AWS/Azure.
- [ ] **Document deletion** – add `DELETE /documents/{id}` to remove documents and their embeddings.
- [ ] **Evaluation metrics** – integrate RAGAS or DeepEval for automated retrieval/generation quality scoring.
