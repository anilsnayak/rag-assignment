from pathlib import Path
from typing import List, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from pypdf import PdfReader

from app.config import settings


def extract_pdf_text(file_path: str) -> Tuple[List[Document], int]:
    reader = PdfReader(file_path)
    filename = Path(file_path).name
    docs: List[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "page": page_number,
                        "filename": filename,
                    },
                )
            )

    return docs, len(reader.pages)


def split_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)
