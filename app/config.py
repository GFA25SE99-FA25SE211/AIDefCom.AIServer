"""Application configuration and environment variables."""

from __future__ import annotations

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration class."""
    
    # Azure Speech Service
    AZURE_SPEECH_KEY: str = os.getenv("AZURE_SPEECH_KEY", "")
    AZURE_SPEECH_REGION: str = os.getenv("AZURE_SPEECH_REGION", "")
    
    # Application
    APP_TITLE: str = "Speech Services API"
    APP_DESCRIPTION: str = "Speech-to-Text + Voice Authentication"
    APP_VERSION: str = "3.0.0"
    
    # Paths
    PROJECT_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    VOICE_PROFILES_DIR: str = os.path.join(PROJECT_ROOT, "data", "voice_profiles")
    MODELS_DIR: str = os.path.join(PROJECT_ROOT, ".models")
    
    # Audio Settings
    SAMPLE_RATE: int = 16000
    MAX_AUDIO_SIZE_MB: int = 6
    MAX_AUDIO_BYTES: int = MAX_AUDIO_SIZE_MB * 1024 * 1024
    
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
        os.makedirs(cls.VOICE_PROFILES_DIR, exist_ok=True)
        os.makedirs(cls.MODELS_DIR, exist_ok=True)


# Validate configuration on import
Config.validate()
