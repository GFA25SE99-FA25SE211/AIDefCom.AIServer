"""Repositories - Data access layer."""

# Azure repositories
from repositories.azure.blob_repository import AzureBlobRepository, AzureBlobStorageError
from repositories.azure.speech_repository import AzureSpeechRepository

# SQL repositories
from repositories.sql.sql_repository import SQLServerRepository, SQLServerError

# Voice repositories
from repositories.voice.cloud_repository import CloudVoiceProfileRepository
from repositories.voice.user_repository import UserRepository

# Database pool
from repositories.database import DatabasePool, DatabaseError, get_database_pool

# Interfaces
from repositories.interfaces import (
    ISpeechRepository,
    ISQLServerRepository,
    IVoiceProfileRepository,
    IRedisService,
)

# Models
from repositories.models import (
    VoiceProfile,
    VoiceSample,
    SpeechRecognitionConfig,
    RecognitionEvent,
)

__all__ = [
    # Azure
    "AzureBlobRepository",
    "AzureBlobStorageError",
    "AzureSpeechRepository",
    # SQL
    "SQLServerRepository",
    "SQLServerError",
    # Voice
    "CloudVoiceProfileRepository",
    "UserRepository",
    # Database
    "DatabasePool",
    "DatabaseError",
    "get_database_pool",
    # Interfaces
    "ISpeechRepository",
    "ISQLServerRepository",
    "IVoiceProfileRepository",
    "IRedisService",
    # Models
    "VoiceProfile",
    "VoiceSample",
    "SpeechRecognitionConfig",
    "RecognitionEvent",
]
