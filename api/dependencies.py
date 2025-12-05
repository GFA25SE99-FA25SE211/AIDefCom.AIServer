"""API dependencies - Dependency injection for services and repositories."""

from __future__ import annotations

import logging
from functools import lru_cache

from app.config import Config

# New import paths
from repositories.azure import AzureSpeechRepository, AzureBlobRepository
from repositories.voice import CloudVoiceProfileRepository
from repositories.sql import SQLServerRepository
from repositories.interfaces import ISQLServerRepository, IVoiceProfileRepository, ISpeechRepository
from services.voice import EmbeddingModel

from services.speech_service import SpeechService
from services.voice_service import VoiceService, QualityThresholds
from services.question_service import QuestionService
from services.redis_service import get_redis_service

from services.interfaces.i_voice_service import IVoiceService
from services.interfaces.i_speech_service import ISpeechService
from services.interfaces.i_question_service import IQuestionService

logger = logging.getLogger(__name__)


# Singleton instances
_azure_speech_repo: ISpeechRepository | None = None
_cloud_voice_profile_repo: IVoiceProfileRepository | None = None
_embedding_model: EmbeddingModel | None = None
_azure_blob_repo: AzureBlobRepository | None = None
_sql_server_repo: ISQLServerRepository | None = None
_voice_service: IVoiceService | None = None
_speech_service: ISpeechService | None = None
_question_service: IQuestionService | None = None



@lru_cache()
def get_azure_speech_repository() -> ISpeechRepository:
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
def get_voice_profile_repository() -> IVoiceProfileRepository:
    """Get cloud voice profile repository instance (Azure Blob only) with optional Redis L2 cache."""
    global _cloud_voice_profile_repo
    if _cloud_voice_profile_repo is None:
        blob = get_azure_blob_repository()
        if blob is None:
            raise RuntimeError("Azure Blob Storage is required but not configured. Set AZURE_STORAGE_CONNECTION_STRING.")
        _cloud_voice_profile_repo = CloudVoiceProfileRepository(
            blob_repo=blob,
            redis_service=get_redis_service(),
        )
        logger.info("Using CloudVoiceProfileRepository (Azure Blob Storage + Redis L2 cache)")
    return _cloud_voice_profile_repo


@lru_cache()
def get_embedding_model() -> EmbeddingModel:
    """Get speaker embedding model instance (Pyannote/WeSpeaker)."""
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = EmbeddingModel(sample_rate=Config.SAMPLE_RATE)
    return _embedding_model


# Backward compatibility alias
get_embedding_model_repository = get_embedding_model


@lru_cache()
def get_azure_blob_repository() -> AzureBlobRepository | None:
    """Get Azure Blob Storage repository instance."""
    global _azure_blob_repo
    if _azure_blob_repo is None:
        if Config.AZURE_STORAGE_CONNECTION_STRING:
            try:
                _azure_blob_repo = AzureBlobRepository(
                    connection_string=Config.AZURE_STORAGE_CONNECTION_STRING,
                    container_name=Config.AZURE_BLOB_CONTAINER_NAME,
                )
                logger.info("Azure Blob Storage repository initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Azure Blob Storage: {e}. Running without blob storage.")
                _azure_blob_repo = None
        else:
            logger.warning("AZURE_STORAGE_CONNECTION_STRING not set. Running without blob storage.")
    return _azure_blob_repo


@lru_cache()
def get_sql_server_repository() -> ISQLServerRepository | None:
    """Get SQL Server repository instance.
    
    Supports two configuration methods:
    1. Connection string (SQL_SERVER_CONNECTION_STRING) - will be parsed automatically
    2. Individual parameters (SQL_SERVER_HOST, SQL_SERVER_DATABASE, etc.)
    """
    global _sql_server_repo
    if _sql_server_repo is None:
        try:
            # Try connection string first (preferred for complex scenarios)
            if Config.SQL_SERVER_CONNECTION_STRING:
                _sql_server_repo = SQLServerRepository.from_connection_string(
                    connection_string=Config.SQL_SERVER_CONNECTION_STRING,
                    users_table=Config.SQL_USERS_TABLE,
                    user_id_column=Config.SQL_USER_ID_COLUMN,
                    voice_path_column=Config.SQL_VOICE_PATH_COLUMN,
                )
                logger.info("SQL Server repository initialized (from connection string)")
            # Fall back to individual parameters
            elif Config.SQL_SERVER_HOST and Config.SQL_SERVER_DATABASE:
                # Parse port from config if provided
                port = int(Config.SQL_SERVER_PORT) if Config.SQL_SERVER_PORT else None
                
                _sql_server_repo = SQLServerRepository(
                    server=Config.SQL_SERVER_HOST,
                    database=Config.SQL_SERVER_DATABASE,
                    username=Config.SQL_SERVER_USERNAME,
                    password=Config.SQL_SERVER_PASSWORD,
                    port=port,
                    users_table=Config.SQL_USERS_TABLE,
                    user_id_column=Config.SQL_USER_ID_COLUMN,
                    voice_path_column=Config.SQL_VOICE_PATH_COLUMN,
                )
                logger.info("SQL Server repository initialized (from individual parameters)")
            else:
                logger.warning("SQL Server config not set. Running without database updates.")
        except Exception as e:
            logger.warning(f"Failed to initialize SQL Server: {e}. Running without database updates.")
            _sql_server_repo = None
    return _sql_server_repo


@lru_cache()
def get_voice_service() -> IVoiceService:
    """Get voice service instance."""
    global _voice_service
    if _voice_service is None:
        # Build thresholds from Config overrides
        thresholds = QualityThresholds(
            min_duration=Config.VOICE_MIN_DURATION,
            min_enroll_duration=Config.VOICE_MIN_ENROLL_DURATION,
            rms_floor=Config.VOICE_RMS_FLOOR,
            voiced_floor=Config.VOICE_VOICED_FLOOR,
            snr_floor_db=Config.VOICE_SNR_FLOOR_DB,
            clip_ceiling=Config.VOICE_CLIP_CEILING,
        )

        _voice_service = VoiceService(
            voice_profile_repo=get_voice_profile_repository(),
            model_repo=get_embedding_model(),
            thresholds=thresholds,
            azure_blob_repo=get_azure_blob_repository(),
            sql_server_repo=get_sql_server_repository(),
        )
    return _voice_service


@lru_cache()
def get_speech_service() -> ISpeechService:
    """Get speech service instance."""
    global _speech_service
    if _speech_service is None:
        _speech_service = SpeechService(
            azure_speech_repo=get_azure_speech_repository(),
            voice_service=get_voice_service(),
            redis_service=get_redis_service(),
            question_service=get_question_service(),
        )
    return _speech_service


@lru_cache()
def get_question_service() -> IQuestionService:
    """Get question service instance."""
    global _question_service
    if _question_service is None:
        _question_service = QuestionService(
            redis_service=get_redis_service(),
            session_ttl=7200,
        )
    return _question_service
