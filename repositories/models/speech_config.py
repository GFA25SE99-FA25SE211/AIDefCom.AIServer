"""Speech Recognition Configuration - Provider-agnostic settings."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


@dataclass
class SpeechRecognitionConfig:
    """Provider-agnostic speech recognition configuration."""
    
    sample_rate: int = 16000
    channels: int = 1
    bits_per_sample: int = 16
    language: str = "vi-VN"
    continuous_recognition: bool = True
    enable_punctuation: bool = True
    enable_capitalization: bool = True
    profanity_filter: bool = True
    segmentation_silence_ms: int = 600
    initial_silence_timeout_ms: int = 5000
    end_silence_timeout_ms: int = 400
    enable_partial_results: bool = True
    stable_partial_threshold: int = 4
    phrase_hints: List[str] = field(default_factory=list)
    enable_word_timestamps: bool = True
    provider_options: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_env(cls) -> "SpeechRecognitionConfig":
        """Create configuration from environment variables."""
        try:
            from app.config import Config
            
            phrase_hints = cls._load_phrase_hints(Config.PHRASE_HINTS_FILE)
            if not phrase_hints:
                default_path = os.path.join(Config.PROJECT_ROOT, "data", "phrase_hints.json")
                phrase_hints = cls._load_phrase_hints(default_path)
            
            return cls(
                sample_rate=Config.SAMPLE_RATE,
                language="vi-VN",
                segmentation_silence_ms=Config.AZURE_SPEECH_SEGMENTATION_SILENCE_MS,
                initial_silence_timeout_ms=Config.AZURE_SPEECH_INITIAL_SILENCE_MS,
                end_silence_timeout_ms=Config.AZURE_SPEECH_END_SILENCE_MS,
                stable_partial_threshold=Config.AZURE_SPEECH_STABLE_PARTIAL_THRESHOLD,
                phrase_hints=phrase_hints,
            )
        except Exception as e:
            logger.warning(f"Failed to load config from environment: {e}")
            return cls()
    
    @staticmethod
    def _load_phrase_hints(file_path: str) -> List[str]:
        """Load phrase hints from JSON file."""
        if not file_path or not os.path.exists(file_path):
            return []
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            phrases: List[str] = []
            for key, value in data.items():
                if key.startswith("_"):
                    continue
                if isinstance(value, list):
                    phrases.extend([p for p in value if isinstance(p, str)])
            
            return phrases
        except Exception:
            return []
    
    def add_phrase_hints(self, hints: Sequence[str]) -> None:
        """Add additional phrase hints."""
        for hint in hints:
            if hint and hint not in self.phrase_hints:
                self.phrase_hints.append(hint)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "language": self.language,
            "segmentation_silence_ms": self.segmentation_silence_ms,
            "phrase_hints_count": len(self.phrase_hints),
        }


@dataclass
class RecognitionEvent:
    """Provider-agnostic speech recognition event."""
    
    type: str  # "partial", "result", "nomatch", "error"
    text: str = ""
    confidence: Optional[float] = None
    is_final: bool = False
    speaker: Optional[str] = None
    user_id: Optional[str] = None
    timestamp_ms: Optional[int] = None
    duration_ms: Optional[int] = None
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        result = {"type": self.type, "text": self.text}
        if self.confidence is not None:
            result["confidence"] = self.confidence
        if self.is_final:
            result["is_final"] = True
        if self.speaker:
            result["speaker"] = self.speaker
        if self.user_id:
            result["user_id"] = self.user_id
        if self.error_message:
            result["error_message"] = self.error_message
        return result
