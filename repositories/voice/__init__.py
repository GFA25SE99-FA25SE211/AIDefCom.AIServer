"""Voice profile repositories."""

from repositories.voice.cloud_repository import CloudVoiceProfileRepository
from repositories.voice.user_repository import UserRepository

__all__ = [
    "CloudVoiceProfileRepository",
    "UserRepository",
]
