"""Application configuration and environment variables."""

from __future__ import annotations

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def _parse_int_env(name: str, default: int) -> int:
    """Parse int from env safely, tolerating values like 'NAME=123' or quoted strings.
    Returns default on any parsing issue.
    """
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    # Accept accidental 'KEY=VALUE' format
    if "=" in raw:
        raw = raw.split("=", 1)[1]
    raw = raw.strip().strip("'").strip('"')
    try:
        return int(raw)
    except Exception:
        return default


class Config:
    """Application configuration class."""
    
    # Azure Speech Service
    AZURE_SPEECH_KEY: str = os.getenv("AZURE_SPEECH_KEY", "")
    AZURE_SPEECH_REGION: str = os.getenv("AZURE_SPEECH_REGION", "")

    # Azure Cache for Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_PASSWORD: str = os.getenv("REDIS_PASSWORD", "")
    REDIS_SSL: bool = os.getenv("REDIS_SSL", "false").lower() == "true"
    REDIS_DB: int = int(os.getenv("REDIS_DB", "0"))
    REDIS_TTL_SECONDS: int = int(os.getenv("REDIS_TTL_SECONDS", "3600"))  # 1 hour default
    
    # Azure Blob Storage
    AZURE_STORAGE_CONNECTION_STRING: str = os.getenv("AZURE_STORAGE_CONNECTION_STRING", "")
    AZURE_BLOB_CONTAINER_NAME: str = os.getenv("AZURE_BLOB_CONTAINER_NAME", "voice-sample")
    
    # Auth/User (.NET) Service
    AUTH_SERVICE_BASE_URL: str = os.getenv("AUTH_SERVICE_BASE_URL", "")
    AUTH_SERVICE_VERIFY_SSL: bool = os.getenv("AUTH_SERVICE_VERIFY_SSL", "false").lower() == "true"
    AUTH_SERVICE_TIMEOUT: int = _parse_int_env("AUTH_SERVICE_TIMEOUT", 10)
    
    # SQL Server Database (supports both connection string and individual parameters)
    # Option 1: Full connection string (will be parsed automatically if provided)
    SQL_SERVER_CONNECTION_STRING: str = os.getenv("SQL_SERVER_CONNECTION_STRING", "")
    # Option 2: Individual parameters (used if connection string not provided)
    SQL_SERVER_HOST: str = os.getenv("SQL_SERVER_HOST", "")
    SQL_SERVER_PORT: str = os.getenv("SQL_SERVER_PORT", "")  # Optional, defaults to 1433
    SQL_SERVER_DATABASE: str = os.getenv("SQL_SERVER_DATABASE", "")
    SQL_SERVER_USERNAME: str = os.getenv("SQL_SERVER_USERNAME", "")
    SQL_SERVER_PASSWORD: str = os.getenv("SQL_SERVER_PASSWORD", "")
    # SQL schema customization (optional)
    SQL_USERS_TABLE: str = os.getenv("SQL_USERS_TABLE", "AspNetUsers")
    SQL_USER_ID_COLUMN: str = os.getenv("SQL_USER_ID_COLUMN", "Id")
    SQL_VOICE_PATH_COLUMN: str = os.getenv("SQL_VOICE_PATH_COLUMN", "VoiceSamplePath")
    
    # Voice Enrollment Settings
    MAX_ENROLLMENT_COUNT: int = 3  # Maximum enrollment samples per user
    
    # Voice Authentication Thresholds (tunable via env vars)
    VOICE_COSINE_THRESHOLD: float = float(os.getenv("VOICE_COSINE_THRESHOLD", "0.45"))
    VOICE_SPEAKER_LOCK_DECAY_SECONDS: float = float(os.getenv("VOICE_SPEAKER_LOCK_DECAY_SECONDS", "5.0"))
    VOICE_SPEAKER_SWITCH_MARGIN: float = float(os.getenv("VOICE_SPEAKER_SWITCH_MARGIN", "0.06"))
    VOICE_SPEAKER_SWITCH_HITS_REQUIRED: int = int(os.getenv("VOICE_SPEAKER_SWITCH_HITS_REQUIRED", "3"))
    
    # Speaker Identification Algorithm Thresholds (tunable via env vars)
    # These control how the system identifies and switches between speakers
    SPEAKER_IDENTIFY_MIN_SECONDS: float = float(os.getenv("SPEAKER_IDENTIFY_MIN_SECONDS", "2.0"))
    SPEAKER_IDENTIFY_WINDOW_SECONDS: float = float(os.getenv("SPEAKER_IDENTIFY_WINDOW_SECONDS", "3.0"))
    SPEAKER_HISTORY_SECONDS: float = float(os.getenv("SPEAKER_HISTORY_SECONDS", "5.0"))
    SPEAKER_IDENTIFY_INTERVAL_SECONDS: float = float(os.getenv("SPEAKER_IDENTIFY_INTERVAL_SECONDS", "0.3"))
    SPEAKER_REDIS_TIMEOUT_SECONDS: float = float(os.getenv("SPEAKER_REDIS_TIMEOUT_SECONDS", "0.5"))
    
    # Fallback cosine thresholds for speaker identification
    SPEAKER_FALLBACK_COSINE_THRESHOLD: float = float(os.getenv("SPEAKER_FALLBACK_COSINE_THRESHOLD", "0.30"))
    SPEAKER_FALLBACK_MARGIN_THRESHOLD: float = float(os.getenv("SPEAKER_FALLBACK_MARGIN_THRESHOLD", "0.06"))
    SPEAKER_WEAK_COSINE_THRESHOLD: float = float(os.getenv("SPEAKER_WEAK_COSINE_THRESHOLD", "0.22"))
    
    # Multi-speaker tracker settings
    SPEAKER_MAX_CONCURRENT: int = int(os.getenv("SPEAKER_MAX_CONCURRENT", "4"))
    SPEAKER_INACTIVITY_TIMEOUT_SECONDS: float = float(os.getenv("SPEAKER_INACTIVITY_TIMEOUT_SECONDS", "30.0"))
    
    # Azure Speech recognition timeouts (ms)
    AZURE_SPEECH_SEGMENTATION_SILENCE_MS: int = int(os.getenv("AZURE_SPEECH_SEGMENTATION_SILENCE_MS", "600"))
    AZURE_SPEECH_INITIAL_SILENCE_MS: int = int(os.getenv("AZURE_SPEECH_INITIAL_SILENCE_MS", "5000"))
    AZURE_SPEECH_END_SILENCE_MS: int = int(os.getenv("AZURE_SPEECH_END_SILENCE_MS", "400"))
    AZURE_SPEECH_STABLE_PARTIAL_THRESHOLD: int = int(os.getenv("AZURE_SPEECH_STABLE_PARTIAL_THRESHOLD", "4"))
    
    # Path to phrase hints file (JSON array of strings)
    PHRASE_HINTS_FILE: str = os.getenv("PHRASE_HINTS_FILE", "")
    
    # Application
    APP_TITLE: str = "AIDefCom AI Service"
    APP_DESCRIPTION: str = "Speech-to-Text + Voice Authentication"
    APP_VERSION: str = "3.0.0"
    
    # Paths
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Local voice profiles directory (unused when VOICE_PROFILE_STORAGE_MODE=blob)
    VOICE_PROFILES_DIR: str = os.path.join(PROJECT_ROOT, "data", "voice_profiles")
    MODELS_DIR: str = os.path.join(PROJECT_ROOT, ".models")
    
    # Audio Settings
    SAMPLE_RATE: int = 16000
    MAX_AUDIO_SIZE_MB: int = 6
    MAX_AUDIO_BYTES: int = MAX_AUDIO_SIZE_MB * 1024 * 1024

    # Voice quality thresholds (override defaults if needed)
    VOICE_MIN_DURATION: float = float(os.getenv("VOICE_MIN_DURATION", "1.5"))
    VOICE_MIN_ENROLL_DURATION: float = float(os.getenv("VOICE_MIN_ENROLL_DURATION", "10.0"))
    VOICE_RMS_FLOOR: float = float(os.getenv("VOICE_RMS_FLOOR", "0.005"))
    VOICE_VOICED_FLOOR: float = float(os.getenv("VOICE_VOICED_FLOOR", "0.10"))
    VOICE_SNR_FLOOR_DB: float = float(os.getenv("VOICE_SNR_FLOOR_DB", "8.0"))
    VOICE_CLIP_CEILING: float = float(os.getenv("VOICE_CLIP_CEILING", "0.03"))

    # Adaptive gain/relax controls
    VOICE_GAIN_TARGET_PCTL: int = int(os.getenv("VOICE_GAIN_TARGET_PCTL", "98"))
    VOICE_GAIN_TARGET_PEAK: float = float(os.getenv("VOICE_GAIN_TARGET_PEAK", "0.80"))
    VOICE_GAIN_MAX: float = float(os.getenv("VOICE_GAIN_MAX", "10.0"))
    VOICE_DYNAMIC_RMS_RELAX: bool = os.getenv("VOICE_DYNAMIC_RMS_RELAX", "true").lower() == "true"
    # Voice profile storage mode: 'local' (JSON files), 'blob' (Azure Blob JSON), or 'sql' (store JSON in SQL Server)
    VOICE_PROFILE_STORAGE_MODE: str = os.getenv("VOICE_PROFILE_STORAGE_MODE", "local").lower()
    
    # CORS
    CORS_ORIGINS: list[str] = ["*"]  # Tighten in production
    CORS_ALLOW_CREDENTIALS: bool = True
    CORS_ALLOW_METHODS: list[str] = ["*"]
    CORS_ALLOW_HEADERS: list[str] = ["*"]
    
    @classmethod
    def validate(cls) -> None:
        """Validate required configuration."""
        if not cls.AZURE_SPEECH_KEY or not cls.AZURE_SPEECH_REGION:
            raise ValueError("Missing AZURE_SPEECH_KEY or AZURE_SPEECH_REGION in .env")
        
        # Ensure directories exist
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        # Only create local voice profiles dir when using local storage mode
        if cls.VOICE_PROFILE_STORAGE_MODE != "blob":
            os.makedirs(cls.VOICE_PROFILES_DIR, exist_ok=True)

        # Parse SQL connection string ONLY if individual components are missing
        # This prevents overwriting explicitly set SQL_SERVER_HOST with parsed value
        if cls.SQL_SERVER_CONNECTION_STRING:
            if not cls.SQL_SERVER_HOST:
                parts = {}
                for segment in cls.SQL_SERVER_CONNECTION_STRING.split(";"):
                    if not segment.strip():
                        continue
                    if "=" in segment:
                        k, v = segment.split("=", 1)
                        parts[k.strip().lower()] = v.strip()
                # Map common keys only if not already set
                if not cls.SQL_SERVER_HOST:
                    cls.SQL_SERVER_HOST = parts.get("server", "") or parts.get("data source", "")
                if not cls.SQL_SERVER_DATABASE:
                    cls.SQL_SERVER_DATABASE = parts.get("database", "") or parts.get("initial catalog", "")
                if not cls.SQL_SERVER_USERNAME:
                    cls.SQL_SERVER_USERNAME = parts.get("user id", "") or parts.get("uid", "")
                if not cls.SQL_SERVER_PASSWORD:
                    cls.SQL_SERVER_PASSWORD = parts.get("password", "") or parts.get("pwd", "")


# Validate configuration on import
Config.validate()
