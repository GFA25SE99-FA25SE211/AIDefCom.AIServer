"""Custom exceptions for the application."""

from __future__ import annotations


class AIServerException(Exception):
    """Base exception for AIServer."""
    pass


class AudioValidationError(AIServerException):
    """Raised when audio validation fails."""
    pass


class VoiceProfileNotFoundError(AIServerException):
    """Raised when voice profile is not found."""
    pass


class VoiceAuthenticationError(AIServerException):
    """Raised when voice authentication fails."""
    pass


class AzureSpeechError(AIServerException):
    """Raised when Azure Speech service encounters an error."""
    pass


class ModelLoadError(AIServerException):
    """Raised when ML model fails to load."""
    pass
