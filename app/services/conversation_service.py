# app/services/conversation_service.py
"""
In-memory conversation history store backed by JSON files for persistence across
server restarts.

Each conversation is identified by a UUID session ID. History is stored as a
list of {"role": "user"|"assistant", "content": "..."} dicts compatible with
the Ollama / OpenAI chat format.
"""

import json
import logging
import uuid
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

from app.config import settings
from app.models.schemas import ConversationMessage

logger = logging.getLogger(__name__)

_HISTORY_SUBDIR = "conversations"


class ConversationService:
    """Thread-safe conversation history manager."""

    def __init__(self) -> None:
        self._store: Dict[str, List[Dict[str, str]]] = {}
        self._lock = Lock()
        self._history_dir = Path(settings.metadata_directory) / _HISTORY_SUBDIR
        self._history_dir.mkdir(parents=True, exist_ok=True)
        self._load_all_from_disk()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _path(self, conversation_id: str) -> Path:
        return self._history_dir / f"{conversation_id}.json"

    def _load_all_from_disk(self) -> None:
        """Reload persisted conversations on startup."""
        for p in self._history_dir.glob("*.json"):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                conv_id = p.stem
                self._store[conv_id] = data
            except Exception:
                logger.warning("Failed to load conversation history from %s", p)

    def _persist(self, conversation_id: str) -> None:
        try:
            self._history_dir.mkdir(parents=True, exist_ok=True)
            with open(self._path(conversation_id), "w", encoding="utf-8") as f:
                json.dump(self._store[conversation_id], f, indent=2)
        except Exception:
            logger.warning("Failed to persist conversation %s", conversation_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_session(self) -> str:
        """Create a new conversation session and return its ID."""
        session_id = str(uuid.uuid4())
        with self._lock:
            self._store[session_id] = []
            self._persist(session_id)
        logger.debug("Created conversation session %s", session_id)
        return session_id

    def add_message(self, conversation_id: str, role: str, content: str) -> None:
        """Append a message to an existing conversation. Creates the session if missing."""
        with self._lock:
            if conversation_id not in self._store:
                self._store[conversation_id] = []
            self._store[conversation_id].append({"role": role, "content": content})
            self._persist(conversation_id)

    def get_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Return the full message list for a session (empty list if unknown)."""
        with self._lock:
            return list(self._store.get(conversation_id, []))

    def get_history_as_schemas(self, conversation_id: str) -> List[ConversationMessage]:
        return [ConversationMessage(**m) for m in self.get_history(conversation_id)]

    def list_sessions(self) -> List[str]:
        with self._lock:
            return sorted(self._store.keys())

    def delete_session(self, conversation_id: str) -> bool:
        """Delete a conversation session. Returns True if it existed."""
        with self._lock:
            existed = conversation_id in self._store
            if existed:
                self._store.pop(conversation_id)
                p = self._path(conversation_id)
                if p.exists():
                    p.unlink()
        return existed

    def get_or_create(self, conversation_id: Optional[str]) -> str:
        """Return the given session ID if valid, or create a new one."""
        if conversation_id and conversation_id in self._store:
            return conversation_id
        return self.create_session()


# Singleton – imported by other services
_conversation_service: Optional[ConversationService] = None


def get_conversation_service() -> ConversationService:
    global _conversation_service
    if _conversation_service is None:
        _conversation_service = ConversationService()
    return _conversation_service
