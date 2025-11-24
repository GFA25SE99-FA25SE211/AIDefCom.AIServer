"""Speech Service Interface - Abstract base for speech-to-text."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List
from fastapi import WebSocket


class ISpeechService(ABC):
    """Interface for speech-to-text business logic."""
    
    @abstractmethod
    async def handle_stt_session(self, ws: WebSocket) -> None:
        """
        Handle complete WebSocket STT session.
        
        Args:
            ws: FastAPI WebSocket connection
            
        Manages full lifecycle: query params, audio streaming, recognition,
        speaker identification, transcript persistence, cleanup.
        """
        pass
    
    @abstractmethod
    async def get_defense_session_users(self, session_id: str) -> Optional[List[str]]:
        """
        Fetch whitelist of user IDs from defense session.
        
        Args:
            session_id: Defense session identifier
            
        Returns:
            List of user_ids or None if API fails after retries
        """
        pass
