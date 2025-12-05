"""SQL Server Repository Interface."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class ISQLServerRepository(ABC):
    """Interface for SQL Server data access."""

    @classmethod
    @abstractmethod
    def from_connection_string(
        cls,
        connection_string: str,
        users_table: str = "AspNetUsers",
        user_id_column: str = "Id",
        voice_path_column: str = "VoiceSamplePath",
    ) -> "ISQLServerRepository":
        """Factory: build repository from connection string."""
        raise NotImplementedError

    @abstractmethod
    def update_voice_sample_path(self, user_id: str, blob_url: str) -> bool:
        """Update voice sample blob URL for a user."""
        raise NotImplementedError

    @abstractmethod
    def get_voice_sample_path(self, user_id: str) -> Optional[str]:
        """Return stored blob URL for user's voice sample."""
        raise NotImplementedError

    @abstractmethod
    def user_exists(self, user_id: str) -> bool:
        """Check if user exists in users table."""
        raise NotImplementedError

    @abstractmethod
    def save_transcript(
        self,
        session_id: str,
        start_time: str,
        end_time: str,
        duration_seconds: float,
        initial_speaker: str,
        lines: List[Dict[str, Any]],
        user_id: Optional[str] = None,
        transcripts_table: str = "SpeechTranscripts",
    ) -> bool:
        """Persist a transcript record."""
        raise NotImplementedError
