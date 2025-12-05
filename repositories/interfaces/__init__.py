"""Repository interfaces for dependency inversion."""

from repositories.interfaces.speech_repository import ISpeechRepository
from repositories.interfaces.sql_repository import ISQLServerRepository
from repositories.interfaces.voice_repository import IVoiceProfileRepository
from repositories.interfaces.redis_service import IRedisService

__all__ = [
    "ISpeechRepository",
    "ISQLServerRepository",
    "IVoiceProfileRepository",
    "IRedisService",
]
