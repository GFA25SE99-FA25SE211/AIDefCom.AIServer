"""Session State Manager - Redis-backed session state for horizontal scaling.

Enables stateless WebSocket handling by storing session state in Redis
instead of local memory. This allows clients to reconnect to any pod
without losing session state.

State stored per session:
- Question mode (active/inactive)
- Question buffer (accumulated text during question mode)
- Last final text (fallback for empty buffer)
- Speaker label (current speaker)
- Custom metadata
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import socket
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from repositories.interfaces import IRedisService

logger = logging.getLogger(__name__)

# Stable container ID for multi-pod deployments
# Priority: CONTAINER_ID env > HOSTNAME env > socket.gethostname()
CONTAINER_ID = os.getenv("CONTAINER_ID") or os.getenv("HOSTNAME") or socket.gethostname()


def get_container_id() -> str:
    """Get stable container ID for this instance."""
    return CONTAINER_ID


@dataclass
class SessionState:
    """Session state data structure."""
    session_id: str
    defense_session_id: Optional[str] = None
    
    # Question mode state
    question_mode: bool = False
    question_buffer: List[str] = field(default_factory=list)
    question_last_final: Optional[str] = None
    
    # Speaker state
    speaker_label: str = "Đang xác định"
    current_user_id: Optional[str] = None
    
    # Metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    container_id: str = field(default_factory=get_container_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Redis storage."""
        return {
            "session_id": self.session_id,
            "defense_session_id": self.defense_session_id,
            "question_mode": self.question_mode,
            "question_buffer": self.question_buffer,
            "question_last_final": self.question_last_final,
            "speaker_label": self.speaker_label,
            "current_user_id": self.current_user_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "container_id": self.container_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """Create from dictionary (Redis data)."""
        return cls(
            session_id=data.get("session_id", ""),
            defense_session_id=data.get("defense_session_id"),
            question_mode=data.get("question_mode", False),
            question_buffer=data.get("question_buffer", []),
            question_last_final=data.get("question_last_final"),
            speaker_label=data.get("speaker_label", "Đang xác định"),
            current_user_id=data.get("current_user_id"),
            created_at=data.get("created_at"),
            updated_at=data.get("updated_at"),
            container_id=data.get("container_id", get_container_id()),
        )


class SessionStateManager:
    """Redis-backed session state manager for horizontal scaling.
    
    Stores session state in Redis so any pod can handle reconnections.
    Falls back to local memory if Redis unavailable.
    
    Usage:
        manager = SessionStateManager(redis_service)
        
        # Start question mode
        await manager.set_question_mode(session_id, True)
        
        # Append to buffer
        await manager.append_question_buffer(session_id, "some text")
        
        # Get full state
        state = await manager.get_state(session_id)
    """
    
    # Redis key prefix
    KEY_PREFIX = "session:state:"
    
    # TTL for session state (2 hours - matches transcript cache)
    STATE_TTL = 7200
    
    def __init__(self, redis_service: Optional["IRedisService"] = None) -> None:
        """Initialize session state manager.
        
        Args:
            redis_service: Optional Redis service for distributed state.
                          Falls back to local memory if None.
        """
        self._redis = redis_service
        self._local_cache: Dict[str, SessionState] = {}  # Fallback if Redis unavailable
        self._lock = asyncio.Lock()
        
        logger.info(
            "SessionStateManager initialized | redis=%s | container_id=%s",
            redis_service is not None, CONTAINER_ID
        )
    
    def _get_key(self, session_id: str) -> str:
        """Get Redis key for session state."""
        return f"{self.KEY_PREFIX}{session_id}"
    
    async def get_state(self, session_id: str) -> SessionState:
        """Get session state, creating if not exists.
        
        Args:
            session_id: WebSocket session ID
            
        Returns:
            SessionState object
        """
        if self._redis:
            try:
                data = await asyncio.wait_for(
                    self._redis.get(self._get_key(session_id)),
                    timeout=1.0
                )
                if data and isinstance(data, dict):
                    return SessionState.from_dict(data)
            except asyncio.TimeoutError:
                logger.debug(f"Redis timeout getting state for {session_id}")
            except Exception as e:
                logger.debug(f"Redis error getting state: {e}")
        
        # Fallback to local cache
        async with self._lock:
            if session_id not in self._local_cache:
                self._local_cache[session_id] = SessionState(session_id=session_id)
            return self._local_cache[session_id]
    
    async def save_state(self, state: SessionState) -> bool:
        """Save session state to Redis.
        
        Args:
            state: SessionState to save
            
        Returns:
            True if saved successfully
        """
        from datetime import datetime
        state.updated_at = datetime.utcnow().isoformat()
        
        if self._redis:
            try:
                await asyncio.wait_for(
                    self._redis.set(
                        self._get_key(state.session_id),
                        state.to_dict(),
                        ttl=self.STATE_TTL
                    ),
                    timeout=1.0
                )
                return True
            except asyncio.TimeoutError:
                logger.debug(f"Redis timeout saving state for {state.session_id}")
            except Exception as e:
                logger.debug(f"Redis error saving state: {e}")
        
        # Always update local cache as fallback
        async with self._lock:
            self._local_cache[state.session_id] = state
        return True
    
    async def delete_state(self, session_id: str) -> bool:
        """Delete session state.
        
        Args:
            session_id: Session to delete
            
        Returns:
            True if deleted
        """
        if self._redis:
            try:
                await asyncio.wait_for(
                    self._redis.delete(self._get_key(session_id)),
                    timeout=1.0
                )
            except Exception as e:
                logger.debug(f"Redis error deleting state: {e}")
        
        async with self._lock:
            self._local_cache.pop(session_id, None)
        return True
    
    # === Question Mode Operations ===
    
    async def set_question_mode(
        self, 
        session_id: str, 
        active: bool,
        defense_session_id: Optional[str] = None,
    ) -> SessionState:
        """Set question mode active/inactive.
        
        Args:
            session_id: WebSocket session ID
            active: Whether question mode is active
            defense_session_id: Optional defense session ID
            
        Returns:
            Updated SessionState
        """
        state = await self.get_state(session_id)
        state.question_mode = active
        
        if defense_session_id:
            state.defense_session_id = defense_session_id
        
        if active:
            # Starting question mode - clear buffer
            state.question_buffer = []
            state.question_last_final = None
            from datetime import datetime
            state.created_at = datetime.utcnow().isoformat()
        
        await self.save_state(state)
        logger.debug(f"Question mode {'started' if active else 'ended'} | session={session_id}")
        return state
    
    async def is_question_mode(self, session_id: str) -> bool:
        """Check if question mode is active."""
        state = await self.get_state(session_id)
        return state.question_mode
    
    async def append_question_buffer(self, session_id: str, text: str) -> SessionState:
        """Append text to question buffer.
        
        Args:
            session_id: WebSocket session ID
            text: Text to append
            
        Returns:
            Updated SessionState
        """
        if not text or not text.strip():
            state = await self.get_state(session_id)
            return state
        
        state = await self.get_state(session_id)
        state.question_buffer.append(text.strip())
        state.question_last_final = text.strip()
        await self.save_state(state)
        return state
    
    async def get_question_text(self, session_id: str) -> str:
        """Get accumulated question text.
        
        Returns joined buffer text, or last final as fallback.
        
        Args:
            session_id: WebSocket session ID
            
        Returns:
            Question text string
        """
        state = await self.get_state(session_id)
        
        if state.question_buffer:
            return " ".join(state.question_buffer).strip()
        
        if state.question_last_final:
            return state.question_last_final.strip()
        
        return ""
    
    async def clear_question_buffer(self, session_id: str) -> SessionState:
        """Clear question buffer and mode.
        
        Args:
            session_id: WebSocket session ID
            
        Returns:
            Updated SessionState
        """
        state = await self.get_state(session_id)
        state.question_mode = False
        state.question_buffer = []
        state.question_last_final = None
        await self.save_state(state)
        return state
    
    # === Speaker Operations ===
    
    async def set_speaker(
        self, 
        session_id: str, 
        speaker_label: str,
        user_id: Optional[str] = None,
    ) -> SessionState:
        """Update current speaker.
        
        Args:
            session_id: WebSocket session ID
            speaker_label: Speaker display name
            user_id: Optional user ID
            
        Returns:
            Updated SessionState
        """
        state = await self.get_state(session_id)
        state.speaker_label = speaker_label
        state.current_user_id = user_id
        await self.save_state(state)
        return state
    
    async def get_speaker(self, session_id: str) -> tuple[str, Optional[str]]:
        """Get current speaker info.
        
        Returns:
            Tuple of (speaker_label, user_id)
        """
        state = await self.get_state(session_id)
        return state.speaker_label, state.current_user_id


# Global singleton instance
_session_state_manager: Optional[SessionStateManager] = None


def get_session_state_manager(
    redis_service: Optional["IRedisService"] = None
) -> SessionStateManager:
    """Get or create global session state manager.
    
    Args:
        redis_service: Optional Redis service (used on first call)
        
    Returns:
        SessionStateManager singleton
    """
    global _session_state_manager
    if _session_state_manager is None:
        _session_state_manager = SessionStateManager(redis_service)
    return _session_state_manager


def set_session_state_manager(manager: SessionStateManager) -> None:
    """Set global session state manager (for testing/DI)."""
    global _session_state_manager
    _session_state_manager = manager
