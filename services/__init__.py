"""Services module - Business logic layer.

This module provides the main service classes for the application:
- SpeechService: Speech-to-text streaming with speaker identification
- OptimizedSpeechService: Low-latency STT with long-lived connections (recommended)
- VoiceService: Voice authentication (enroll, verify, identify)
- QuestionService: Question duplicate detection
- RedisService: Redis cache operations

Submodules:
- audio: Audio processing utilities (buffer, noise filter, VAD)
- speech: Speech processing utilities (recognition, text utils)
- voice: Voice processing utilities (speaker tracking, identification)
- session: Session management (room, state)
- interfaces: Service interfaces (protocols)
"""

from services.speech_service import SpeechService
from services.voice_service import VoiceService
from services.question_service import QuestionService
from services.redis_service import RedisService, get_redis_service

__all__ = [
    "SpeechService",
    "VoiceService",
    "QuestionService",
    "RedisService",
    "get_redis_service",
]
