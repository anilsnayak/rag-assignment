import os

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from app.config import settings

_embeddings = HuggingFaceEmbeddings(model_name=settings.embedding_model)
_vector_store = Chroma(
    collection_name="rag_assignment_collection",
    persist_directory=settings.chroma_persist_directory,
    embedding_function=_embeddings,
)


def get_vector_store() -> Chroma:
    return _vector_store
