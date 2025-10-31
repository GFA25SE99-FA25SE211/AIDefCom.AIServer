"""Voice router - REST endpoints for voice authentication."""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, File, Form, UploadFile, Depends
from fastapi.responses import JSONResponse

from api.dependencies import get_voice_service
from services.voice_service import VoiceService

from app.config import Config


router = APIRouter(prefix="/voice", tags=["voice-auth"])
logger = logging.getLogger(__name__)


def _validate_audio_file(upload: UploadFile, raw: bytes) -> None:
    """Validate uploaded audio file."""
    if not raw:
        raise ValueError("Empty audio")
    if len(raw) > Config.MAX_AUDIO_BYTES:
        raise ValueError(f"Audio too large (>{Config.MAX_AUDIO_SIZE_MB}MB)")


@router.post("/profile")
async def get_or_create_profile(
    user_id: str = Form(...),
    name: Optional[str] = Form(None),
    voice_service: VoiceService = Depends(get_voice_service),
):
    """Get existing profile or create new one."""
    try:
        result = voice_service.get_or_create_profile(user_id, name)
        return JSONResponse(content=result, status_code=200)
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@router.post("/profile/create")
async def create_voice_profile(
    user_id: str = Form(...),
    name: Optional[str] = Form(None),
    voice_service: VoiceService = Depends(get_voice_service),
):
    """Create new voice profile."""
    try:
        result = voice_service.create_profile(user_id, name)
        return JSONResponse(content=result, status_code=201)
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@router.post("/enroll")
async def enroll_voice(
    user_id: str = Form(...),
    audio_file: UploadFile = File(...),
    voice_service: VoiceService = Depends(get_voice_service),
):
    """Enroll voice sample for user."""
    try:
        audio_data = await audio_file.read()
        
        try:
            _validate_audio_file(audio_file, audio_data)
        except ValueError as ve:
            return JSONResponse(content={"error": str(ve)}, status_code=400)
        
        result = voice_service.enroll_voice(user_id, audio_data)
        
        if "error" in result:
            return JSONResponse(content=result, status_code=400)
        
        return JSONResponse(content=result, status_code=200)
    
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@router.post("/identify")
async def identify_speaker(
    audio_file: UploadFile = File(...),
    voice_service: VoiceService = Depends(get_voice_service),
):
    """Identify speaker from voice sample."""
    try:
        audio_data = await audio_file.read()
        
        try:
            _validate_audio_file(audio_file, audio_data)
        except ValueError as ve:
            return JSONResponse(content={"error": str(ve)}, status_code=400)
        
        result = voice_service.identify_speaker(audio_data)
        return JSONResponse(content=result, status_code=200)
    
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@router.post("/verify")
async def verify_voice(
    user_id: str = Form(...),
    audio_file: UploadFile = File(...),
    voice_service: VoiceService = Depends(get_voice_service),
):
    """Verify if audio matches user's voice profile."""
    try:
        audio_data = await audio_file.read()
        
        try:
            _validate_audio_file(audio_file, audio_data)
        except ValueError as ve:
            return JSONResponse(content={"error": str(ve)}, status_code=400)
        
        result = voice_service.verify_voice(user_id, audio_data)
        
        if "error" in result:
            return JSONResponse(content=result, status_code=400)
        
        return JSONResponse(content=result, status_code=200)
    
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@router.get("/profiles")
async def list_profiles(
    voice_service: VoiceService = Depends(get_voice_service),
):
    """List all voice profiles."""
    try:
        profiles = voice_service.list_all_profiles()
        return JSONResponse(content={"profiles": profiles}, status_code=200)
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@router.get("/profile/{user_id}")
async def get_profile_info(
    user_id: str,
    voice_service: VoiceService = Depends(get_voice_service),
):
    """Get profile information."""
    try:
        profile = voice_service.get_profile_info(user_id)
        if profile is None:
            return JSONResponse(content={"error": "Profile not found"}, status_code=404)
        return JSONResponse(content=profile, status_code=200)
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@router.delete("/profile/{user_id}")
async def delete_profile(
    user_id: str,
    voice_service: VoiceService = Depends(get_voice_service),
):
    """Delete voice profile."""
    try:
        result = voice_service.delete_profile(user_id)
        if "error" in result:
            return JSONResponse(content=result, status_code=404)
        return JSONResponse(content=result, status_code=200)
    except Exception as exc:
        return JSONResponse(content={"error": str(exc)}, status_code=500)


