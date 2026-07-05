# app/services/qa_service.py
"""
Question-answering service.

- Retrieves relevant chunks from ChromaDB.
- Builds a grounded context with source citations.
- Calls the configured LLM (Ollama) with strict, document-bound instructions.
- Optionally maintains multi-turn conversation history.
- Exposes both a blocking `ask()` and a generator `ask_stream()` for SSE streaming.
"""

import logging
from typing import Generator, List, Optional, Tuple

import ollama

from app.config import settings
from app.core.exceptions import QuestionAnsweringError
from app.core.prompts import QA_SYSTEM_PROMPT
from app.models.schemas import AskResponse, SourceItem
from app.services.conversation_service import get_conversation_service
from app.services.vector_store import get_vector_store

logger = logging.getLogger(__name__)

# Canonical "no answer" message – returned when retrieval finds nothing relevant.
_NO_ANSWER = "I cannot answer this from the provided document(s)."


class QAService:
    def __init__(self) -> None:
        self.vector_store = get_vector_store()
        self.ollama_client = ollama.Client(host=settings.ollama_base_url)
        self.conversation_service = get_conversation_service()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_context(self, results) -> Tuple[str, List[SourceItem]]:
        """Convert retrieval results into a formatted context string and source list."""
        context_parts: List[str] = []
        sources: List[SourceItem] = []

        for doc in results[: settings.max_context_chunks]:
            page = doc.metadata.get("page")
            filename = doc.metadata.get("filename", "unknown")
            document_id = doc.metadata.get("document_id", "unknown")
            snippet = doc.page_content[:400].strip()

            context_parts.append(
                f"[Document: {filename} | Page: {page}]\n{doc.page_content}"
            )
            sources.append(
                SourceItem(
                    page=page,
                    document_id=document_id,
                    filename=filename,
                    snippet=snippet,
                )
            )

        return "\n\n".join(context_parts), sources

    def _retrieve(self, question: str, document_id: Optional[str]):
        """Run similarity search, optionally filtered to a single document."""
        search_kwargs = {"k": settings.top_k}
        if document_id:
            retriever = self.vector_store.as_retriever(
                search_kwargs={
                    **search_kwargs,
                    "filter": {"document_id": document_id},
                }
            )
        else:
            retriever = self.vector_store.as_retriever(search_kwargs=search_kwargs)
        return retriever.invoke(question)

    def _build_messages(
        self,
        question: str,
        context: str,
        conversation_id: Optional[str],
    ) -> List[dict]:
        """Assemble the Ollama message list including optional conversation history."""
        messages: List[dict] = [{"role": "system", "content": QA_SYSTEM_PROMPT}]

        # Inject prior conversation turns (without the system prompt repetition).
        if conversation_id:
            history = self.conversation_service.get_history(conversation_id)
            messages.extend(history)

        user_content = (
            f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )
        messages.append({"role": "user", "content": user_content})
        return messages

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ask(
        self,
        question: str,
        document_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> AskResponse:
        """
        Blocking question-answer call.

        Returns an AskResponse with `grounded=True` when the answer is derived
        from retrieved document content, and `grounded=False` when no relevant
        chunks were found.
        """
        results = self._retrieve(question, document_id)

        if not results:
            logger.info("No relevant chunks found for question=%r", question)
            # Persist the exchange even when ungrounded so history is coherent.
            session_id = self.conversation_service.get_or_create(conversation_id)
            self.conversation_service.add_message(session_id, "user", question)
            self.conversation_service.add_message(session_id, "assistant", _NO_ANSWER)
            return AskResponse(
                answer=_NO_ANSWER,
                grounded=False,
                sources=[],
                conversation_id=session_id,
            )

        context, sources = self._build_context(results)

        # Resolve / create the conversation session.
        session_id = self.conversation_service.get_or_create(conversation_id)
        messages = self._build_messages(question, context, session_id)

        try:
            response = self.ollama_client.chat(
                model=settings.ollama_model,
                messages=messages,
                think=settings.ollama_think,
                options={"temperature": 0},
            )
            answer = response["message"]["content"].strip()
        except Exception as exc:
            logger.exception("LLM call failed during question answering")
            raise QuestionAnsweringError(str(exc)) from exc

        # Detect if the model effectively said "I don't know".
        grounded = bool(answer) and _NO_ANSWER.lower() not in answer.lower()
        if not answer:
            answer = _NO_ANSWER

        # Persist to conversation history.
        self.conversation_service.add_message(session_id, "user", question)
        self.conversation_service.add_message(session_id, "assistant", answer)

        logger.info(
            "QA: grounded=%s session=%s sources=%d", grounded, session_id, len(sources)
        )
        return AskResponse(
            answer=answer,
            grounded=grounded,
            sources=sources,
            conversation_id=session_id,
        )

    def ask_stream(
        self,
        question: str,
        document_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Streaming generator for Server-Sent Events.

        Yields SSE-formatted strings: ``data: <chunk>\\n\\n``
        Sends ``data: [DONE]\\n\\n`` as the final event.
        """
        results = self._retrieve(question, document_id)

        session_id = self.conversation_service.get_or_create(conversation_id)

        if not results:
            self.conversation_service.add_message(session_id, "user", question)
            self.conversation_service.add_message(session_id, "assistant", _NO_ANSWER)
            yield f"data: {_NO_ANSWER}\n\n"
            yield "data: [DONE]\n\n"
            return

        context, _ = self._build_context(results)
        messages = self._build_messages(question, context, session_id)

        full_answer_parts: List[str] = []
        try:
            stream = self.ollama_client.chat(
                model=settings.ollama_model,
                messages=messages,
                stream=True,
                think=settings.ollama_think,
                options={"temperature": 0},
            )
            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    full_answer_parts.append(token)
                    # Escape newlines so the SSE frame stays on one line.
                    safe_token = token.replace("\n", "\\n")
                    yield f"data: {safe_token}\n\n"
        except Exception as exc:
            logger.exception("LLM streaming call failed")
            yield f"data: ERROR: {exc}\n\n"
            yield "data: [DONE]\n\n"
            return

        full_answer = "".join(full_answer_parts).strip()
        self.conversation_service.add_message(session_id, "user", question)
        self.conversation_service.add_message(session_id, "assistant", full_answer)
        yield "data: [DONE]\n\n"
