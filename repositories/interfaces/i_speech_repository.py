"""Speech Repository Interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, List, Optional


class ISpeechRepository(ABC):
    """Interface for Azure Speech SDK operations."""
    
    @abstractmethod
    def recognize_continuous_async(
        self,
        audio_stream: Any,
        speaker: str,
        extra_phrases: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Stream continuous speech recognition events.
        
        Args:
            audio_stream: Azure PushAudioInputStream
            speaker: Speaker name for phrase hints
            extra_phrases: Additional phrase hints
            
        Yields:
            Recognition event dicts (types: recognized, recognizing, canceled)
        """
        pass
