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
        whitelist_user_ids: Optional[List[str]] = None,
        pre_filtered: bool = False,
    ) -> Dict[str, Any]:
        """
        Identify speaker from audio (1:N matching).
        
        Args:
            audio_bytes: Audio data in bytes
            whitelist_user_ids: Optional list of user IDs to search (None = all enrolled)
            pre_filtered: If True, skip noise filtering (already done upstream)
            
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
    
    @abstractmethod
    async def get_defense_session_users_with_info(self, session_id: str) -> Optional[Dict[str, Dict[str, str]]]:
        """
        Fetch user info (id, name, role) from defense session.
        
        Args:
            session_id: Defense session identifier
            
        Returns:
            Dict mapping user_id -> {"name": "...", "role": "...", "display_name": "Tên (Vai trò)"}
            or None if API fails
        """
        pass
    
    @abstractmethod
    def preload_session_profiles(
        self, 
        defense_session_id: str, 
        user_ids: List[str],
        user_info_map: Optional[Dict[str, Dict[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Preload and cache voice profiles for a defense session.
        
        This loads embeddings from DB/Blob ONCE at session start,
        eliminating I/O per identification call.
        
        Args:
            defense_session_id: Defense session identifier (for caching)
            user_ids: List of user IDs to preload
            user_info_map: Optional dict mapping user_id -> {name, role, display_name}
                          If provided, profiles will use display_name for speaker identification
            
        Returns:
            List of profile dicts with user_id, user_name (display_name), embeddings
        """
        pass
    
    @abstractmethod
    def get_session_profiles(self, defense_session_id: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached profiles for a session (if preloaded).
        
        Args:
            defense_session_id: Defense session identifier
            
        Returns:
            Cached profiles or None if not preloaded
        """
        pass
    
    @abstractmethod
    def clear_session_cache(self, defense_session_id: str) -> None:
        """
        Clear cached voice profiles for a session.
        
        Call this when a defense session ends to free memory.
        
        Args:
            defense_session_id: Defense session identifier
        """
        pass
    
    @abstractmethod
    def identify_speaker_with_cache(
        self,
        audio_bytes: bytes,
        preloaded_profiles: List[Dict[str, Any]],
        pre_filtered: bool = False,
    ) -> Dict[str, Any]:
        """
        Identify speaker using pre-loaded profiles (no DB/Blob I/O).
        
        This is the FAST PATH for streaming identification when profiles
        have been preloaded via preload_session_profiles().
        
        Args:
            audio_bytes: Audio data in bytes
            preloaded_profiles: Profiles from preload_session_profiles()
            pre_filtered: If True, skip noise filtering
            
        Returns:
            Dict with identified, speaker_id, speaker_name, score, confidence
        """
        pass
