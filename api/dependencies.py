"""API dependencies - Dependency injection for services and repositories."""

from __future__ import annotations

from functools import lru_cache

from app.config import Config
from repositories.azure_speech_repository import AzureSpeechRepository
from repositories.voice_profile_repository import VoiceProfileRepository
from repositories.models.speechbrain_model import SpeechBrainModelRepository
from services.speech_service import SpeechService
from services.voice_service import VoiceService


# Singleton instances
_azure_speech_repo: AzureSpeechRepository | None = None
_voice_profile_repo: VoiceProfileRepository | None = None
_speechbrain_model_repo: SpeechBrainModelRepository | None = None
_voice_service: VoiceService | None = None
_speech_service: SpeechService | None = None


@lru_cache()
def get_azure_speech_repository() -> AzureSpeechRepository:
    """Get Azure Speech repository instance."""
    global _azure_speech_repo
    if _azure_speech_repo is None:
        _azure_speech_repo = AzureSpeechRepository(
            subscription_key=Config.AZURE_SPEECH_KEY,
            region=Config.AZURE_SPEECH_REGION,
            sample_rate=Config.SAMPLE_RATE,
        )
    return _azure_speech_repo


@lru_cache()
def get_voice_profile_repository() -> VoiceProfileRepository:
    """Get voice profile repository instance."""
    global _voice_profile_repo
    if _voice_profile_repo is None:
        _voice_profile_repo = VoiceProfileRepository(
            profiles_dir=Config.VOICE_PROFILES_DIR
        )
    return _voice_profile_repo


@lru_cache()
def get_speechbrain_model_repository() -> SpeechBrainModelRepository:
    """Get SpeechBrain model repository instance."""
    global _speechbrain_model_repo
    if _speechbrain_model_repo is None:
        _speechbrain_model_repo = SpeechBrainModelRepository(
            models_dir=Config.MODELS_DIR,
            sample_rate=Config.SAMPLE_RATE,
        )
    return _speechbrain_model_repo


@lru_cache()
def get_voice_service() -> VoiceService:
    """Get voice service instance."""
    global _voice_service
    if _voice_service is None:
        _voice_service = VoiceService(
            voice_profile_repo=get_voice_profile_repository(),
            model_repo=get_speechbrain_model_repository(),
        )
    return _voice_service


@lru_cache()
def get_speech_service() -> SpeechService:
    """Get speech service instance."""
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechService(
            azure_speech_repo=get_azure_speech_repository(),
            voice_service=get_voice_service(),
        )
    return _speech_service
