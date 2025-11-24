"""Voice Service Interface - Abstract base for voice authentication."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class IVoiceService(ABC):
    """Interface for voice authentication business logic."""
    
    @abstractmethod
    def enroll_voice(self, user_id: str, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Enroll voice sample for user.
        
        Args:
            user_id: User identifier
            audio_bytes: Audio data in bytes (WAV/PCM format)
            
        Returns:
            Dict with success, enrollment_count, is_complete, message
            
        Raises:
            VoiceAuthenticationError: If user not found
            AudioValidationError: If audio quality insufficient
        """
        pass
    
    @abstractmethod
    def identify_speaker(
        self, 
        audio_bytes: bytes, 
        whitelist_user_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Identify speaker from audio (1:N matching).
        
        Args:
            audio_bytes: Audio data in bytes
            whitelist_user_ids: Optional list of user IDs to search (None = all enrolled)
            
        Returns:
            Dict with identified, speaker_id, speaker_name, score, confidence
        """
        pass
    
    @abstractmethod
    def verify_voice(self, user_id: str, audio_bytes: bytes) -> Dict[str, Any]:
        """
        Verify if audio matches claimed user (1:1 matching).
        
        Args:
            user_id: User identifier to verify against
            audio_bytes: Audio data in bytes
            
        Returns:
            Dict with verified, match, score, confidence
            
        Raises:
            VoiceProfileNotFoundError: If user not enrolled
        """
        pass
    
    @abstractmethod
    def get_defense_session_users(self, session_id: str) -> Optional[List[str]]:
        """
        Fetch whitelist of user IDs from defense session.
        
        Args:
            session_id: Defense session identifier
            
        Returns:
            List of user_ids or None if API fails
        """
        pass
