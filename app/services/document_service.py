import json
import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from fastapi import UploadFile

from app.config import settings
from app.core.exceptions import DocumentNotFoundError, UnsupportedFileError
from app.models.schemas import DocumentResponse
from app.services.vector_store import get_vector_store
from app.utils.pdf_loader import extract_pdf_text, split_documents

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self) -> None:
        self.documents_dir = Path(settings.documents_directory)
        self.metadata_dir = Path(settings.metadata_directory)
        self.documents_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _metadata_path(self, document_id: str) -> Path:
        return self.metadata_dir / f"{document_id}.json"

    def _save_metadata(self, metadata: Dict) -> None:
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        with open(self._metadata_path(metadata["document_id"]), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    def _load_metadata(self, document_id: str) -> Dict:
        path = self._metadata_path(document_id)
        if not path.exists():
            raise DocumentNotFoundError(f"Document '{document_id}' not found.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    async def upload_document(self, file: UploadFile) -> DocumentResponse:
        if not file.filename or not file.filename.lower().endswith(".pdf"):
            raise UnsupportedFileError("Only PDF files are supported.")

        document_id = str(uuid.uuid4())
        stored_filename = f"{document_id}__{file.filename}"
        stored_path = self.documents_dir / stored_filename

        self.documents_dir.mkdir(parents=True, exist_ok=True)
        with open(stored_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)

        pages, num_pages = extract_pdf_text(str(stored_path))
        chunks = split_documents(pages)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["document_id"] = document_id
            chunk.metadata["filename"] = file.filename
            chunk.metadata["chunk_id"] = idx

        vector_store = get_vector_store()
        vector_store.add_documents(chunks)

        metadata = {
            "document_id": document_id,
            "filename": file.filename,
            "stored_filename": stored_filename,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
            "num_pages": num_pages,
            "num_chunks": len(chunks),
        }
        self._save_metadata(metadata)

        logger.info("Uploaded document_id=%s filename=%s chunks=%s", document_id, file.filename, len(chunks))
        return DocumentResponse(**metadata)

    def list_documents(self) -> List[DocumentResponse]:
        documents: List[DocumentResponse] = []
        for path in sorted(self.metadata_dir.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                documents.append(DocumentResponse(**json.load(f)))
        return documents

    def get_document(self, document_id: str) -> Dict:
        return self._load_metadata(document_id)
