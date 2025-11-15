"""Voice router - REST endpoints for voice authentication."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, UploadFile, Depends, Path
from fastapi.responses import JSONResponse

from api.dependencies import get_voice_service
from services.voice_service import VoiceService
from api.schemas.voice_schemas import (
    EnrollmentResponse,
    IdentificationResponse,
    VerificationResponse,
    ErrorResponse,
)

from app.config import Config


router = APIRouter(prefix="/voice", tags=["voice-auth"])
logger = logging.getLogger(__name__)


def _validate_audio_file(upload: UploadFile, raw: bytes) -> None:
    """Validate uploaded audio file."""
    if not raw:
        raise ValueError("Empty audio")
    if len(raw) > Config.MAX_AUDIO_BYTES:
        raise ValueError(f"Audio too large (>{Config.MAX_AUDIO_SIZE_MB}MB)")


@router.post(
    "/users/{user_id}/enroll",
    summary="Enroll Voice Sample",
    description="Register a voice sample for a specific user. Requires 3 samples for full enrollment.",
    response_model=EnrollmentResponse,
    responses={
        200: {
            "description": "Enrollment successful or failed (check success field)",
            "model": EnrollmentResponse,
        },
        400: {
            "description": "Invalid audio file or validation error",
            "model": ErrorResponse,
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse,
        },
    },
)
async def enroll_voice(
    user_id: str = Path(..., description="User ID to enroll voice for"),
    audio_file: UploadFile = File(..., description="Audio file for enrollment (WAV/MP3/FLAC, max 10MB)"),
    voice_service: VoiceService = Depends(get_voice_service),
):
    """Enroll voice sample for a specific user.
    
    **REST API:** `POST /voice/users/{user_id}/enroll`
    
    **Process:**
    1. Uploads audio sample for the specified user
    2. Extracts voice embedding using SpeechBrain ECAPA-TDNN model
    3. Stores embedding in Azure SQL Database and audio in Azure Blob Storage
    4. Requires minimum 3 samples for full enrollment before verification/identification
    
    **Requirements:**
    - Audio format: WAV, MP3, or FLAC
    - Max file size: 10MB
    - Recommended duration: 3-5 seconds of clean speech
    - Minimum 3 samples needed for enrollment completion
    """
    try:
        audio_data = await audio_file.read()
        if not audio_data:
            logger.error("Empty audio data received for enroll_voice")
            return JSONResponse(content={"error": "Empty audio data"}, status_code=400)
        try:
            _validate_audio_file(audio_file, audio_data)
        except ValueError as ve:
            logger.error(f"Audio validation error: {ve}")
            return JSONResponse(content={"error": str(ve)}, status_code=400)
        
        result = voice_service.enroll_voice(user_id, audio_data)
        status = 200 if result.get("success") else 400
        if not result.get("success"):
            logger.error(f"Enroll failed: {result}")
        return JSONResponse(content=result, status_code=status)
    except Exception as exc:
        logger.error(f"Unexpected error in enroll_voice: {exc}")
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@router.post(
    "/identify",
    summary="Identify Speaker",
    description="Identify which enrolled user is speaking from an audio sample.",
    response_model=IdentificationResponse,
    responses={
        200: {
            "description": "Identification completed (check identified field for success)",
            "model": IdentificationResponse,
        },
        400: {
            "description": "Invalid audio file or no speakers found",
            "model": ErrorResponse,
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse,
        },
    },
)
async def identify_speaker(
    audio_file: UploadFile = File(..., description="Audio file for speaker identification (WAV/MP3/FLAC, max 10MB)"),
    voice_service: VoiceService = Depends(get_voice_service),
):
    """Identify speaker from voice sample.
    
    **REST API:** `POST /voice/identify`
    
    **Process:**
    1. Extracts voice embedding from audio sample
    2. Compares against all enrolled users (≥3 samples) in database
    3. Returns best match if similarity score exceeds threshold (0.7)
    
    **Returns:**
    - `identified=true`: Speaker found with high confidence
    - `identified=false`: No matching speaker or score too low
    - `speaker_id`: User ID of identified speaker
    - `speaker_name`: Display name of identified speaker
    - `confidence`: Confidence score (0-1)
    - `score`: Similarity score (0-1)
    """
    try:
        audio_data = await audio_file.read()
        try:
            _validate_audio_file(audio_file, audio_data)
        except ValueError as ve:
            return JSONResponse(content={"error": str(ve)}, status_code=400)
        
        result = voice_service.identify_speaker(audio_data)
        status = 200 if result.get("success") else 400
        return JSONResponse(content=result, status_code=status)
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@router.post(
    "/users/{user_id}/verify",
    summary="Verify Voice",
    description="Verify if an audio sample matches a specific user's voice profile.",
    response_model=VerificationResponse,
    responses={
        200: {
            "description": "Verification completed (check verified field for success)",
            "model": VerificationResponse,
        },
        400: {
            "description": "Invalid audio file or user not enrolled",
            "model": ErrorResponse,
        },
        500: {
            "description": "Internal server error",
            "model": ErrorResponse,
        },
    },
)
async def verify_voice(
    user_id: str = Path(..., description="User ID to verify voice against"),
    audio_file: UploadFile = File(..., description="Audio file for verification (WAV/MP3/FLAC, max 10MB)"),
    voice_service: VoiceService = Depends(get_voice_service),
):
    """Verify if audio matches a specific user's voice profile.
    
    **REST API:** `POST /voice/users/{user_id}/verify`
    
    **Process:**
    1. Checks if user has completed enrollment (≥3 samples)
    2. Extracts voice embedding from audio sample
    3. Compares against user's stored embeddings
    4. Returns verification result based on similarity threshold (0.7)
    
    **Returns:**
    - `verified=true`: Voice matches claimed user ID
    - `verified=false`: Voice does not match or user not enrolled
    - `claimed_id`: User ID being verified
    - `speaker_id`: Actual identified speaker (if match)
    - `match`: Boolean indicating if claimed_id == speaker_id
    - `confidence`: Confidence score (0-1)
    - `score`: Similarity score (0-1)
    
    **Use Cases:**
    - Authentication: Verify user identity via voice
    - Access Control: Grant access if voice matches
    - Security: Detect voice spoofing attempts
    """
    try:
        audio_data = await audio_file.read()
        try:
            _validate_audio_file(audio_file, audio_data)
        except ValueError as ve:
            return JSONResponse(content={"error": str(ve)}, status_code=400)
        
        result = voice_service.verify_voice(user_id, audio_data)
        status = 200 if result.get("success") else 400
        return JSONResponse(content=result, status_code=status)
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)

        status = 200 if result.get("success") else 400
        return JSONResponse(content=result, status_code=status)
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)
