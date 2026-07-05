from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "RAG Assignment API"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "INFO"

    chroma_persist_directory: str = "./data/chroma"
    documents_directory: str = "./data/documents"
    metadata_directory: str = "./data/metadata"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3.5:4b"
    ollama_think: bool = False

    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 4
    max_context_chunks: int = 6

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
