"""Speech Repository Interface - Provider-agnostic speech recognition."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, AsyncIterable, Dict, List, Optional, Sequence


class ISpeechRepository(ABC):
    """Interface for speech recognition operations.
    
    Provider-agnostic interface supporting Azure, Google, Whisper, etc.
    """
    
    sample_rate: int  # Audio sample rate in Hz
    
    @abstractmethod
    async def recognize_stream(
        self,
        audio_source: AsyncIterable[bytes],
        extra_phrases: Sequence[str] | None = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream speech recognition from an async audio source.
        
        Args:
            audio_source: Async iterable providing raw PCM audio chunks
            extra_phrases: Additional phrase hints for this session
        
        Yields:
            Recognition events: {type: partial|result|nomatch|error, text?, error?}
        """
        pass
    
    @abstractmethod
    async def recognize_continuous_async(
        self,
        audio_stream: Any,
        speaker: str,
        extra_phrases: Optional[List[str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream continuous speech recognition events."""
        pass
    
    @abstractmethod
    def recognize_once(self, audio_bytes: bytes) -> str:
        """Perform one-shot speech recognition on audio bytes."""
        pass
    
    def get_provider_name(self) -> str:
        """Get name of this speech provider."""
        return "unknown"
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return ["vi-VN", "en-US", "en-GB", "ja-JP", "ko-KR", "zh-CN"]
