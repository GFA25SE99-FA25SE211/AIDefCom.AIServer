"""Voice router - REST endpoints for voice authentication."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, UploadFile, Depends, Path
from fastapi.responses import JSONResponse

from api.dependencies import get_voice_service
from services.interfaces.i_voice_service import IVoiceService
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
    summary="Enroll voice sample",
    description=(
        "Đăng ký mẫu giọng nói cho user. Cần tối thiểu 3 mẫu để hoàn tất enrollment. "
        "Mỗi mẫu ~3-5s giọng nói rõ, ít nhiễu."
    ),
    response_model=EnrollmentResponse,
    responses={
        200: {
            "description": "Enrollment result",
            "content": {
                "application/json": {
                    "example": {
                        "type": "enrollment",
                        "success": True,
                        "user_id": "USR001",
                        "enrollment_count": 2,
                        "min_required": 3,
                        "is_complete": False,
                        "message": "Enrollment sample 2/3 saved successfully"
                    }
                }
            },
        },
        400: {"description": "Invalid audio or validation error", "content": {"application/json": {"example": {"error": "Empty audio data"}}}},
        500: {"description": "Internal server error", "content": {"application/json": {"example": {"error": "Unexpected error"}}}},
    },
)
async def enroll_voice(
    user_id: str = Path(..., description="User ID to enroll voice for"),
    audio_file: UploadFile = File(..., description="Audio file for enrollment (WAV/MP3/FLAC, max 10MB)"),
    voice_service: IVoiceService = Depends(get_voice_service),
):
    """Enroll voice sample for a specific user.
    
    **REST API:** `POST /voice/users/{user_id}/enroll`
    
    **Process:**
    1. Nhận audio và trích xuất embedding (SpeechBrain ECAPA-TDNN)
    2. Lưu embedding + metadata (cloud storage)
    3. Cập nhật số mẫu; complete khi đạt `min_required` (3)

    **Yêu cầu:**
    - Định dạng: WAV / MP3 / FLAC
    - Kích thước tối đa: 10MB
    - Thời lượng khuyến nghị: 3–5 giây
    - Ít nhiễu nền, giọng nói rõ
    """
    import asyncio
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
    
    try:
        # Read audio data with timeout
        try:
            audio_data = await asyncio.wait_for(audio_file.read(), timeout=30.0)
        except asyncio.TimeoutError:
            logger.error(f"Timeout reading audio file for user {user_id}")
            return JSONResponse(
                content={"error": "Timeout reading audio file", "user_id": user_id},
                status_code=408
            )
        
        if not audio_data:
            logger.error("Empty audio data received for enroll_voice")
            return JSONResponse(content={"error": "Empty audio data"}, status_code=400)
        
        try:
            _validate_audio_file(audio_file, audio_data)
        except ValueError as ve:
            logger.error(f"Audio validation error: {ve}")
            return JSONResponse(content={"error": str(ve)}, status_code=400)
        
        # Run enrollment in thread pool with timeout to prevent blocking
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    executor,
                    voice_service.enroll_voice,
                    user_id,
                    audio_data
                ),
                timeout=60.0  # 60 second timeout for enrollment
            )
        except asyncio.TimeoutError:
            logger.error(f"Enrollment timeout for user {user_id}")
            return JSONResponse(
                content={
                    "error": "Enrollment processing timeout. Please try again.",
                    "user_id": user_id
                },
                status_code=504
            )
        finally:
            executor.shutdown(wait=False)
        
        # Check if there's an error in the result
        if "error" in result:
            logger.warning(f"Enrollment error for user {user_id}: {result.get('error')}")
            return JSONResponse(content=result, status_code=400)
        
        # Success case - result contains enrollment data
        logger.info(f"Enrollment successful for user {user_id}: {result.get('enrollment_count')}/{result.get('max_enrollment_count')}")
        return JSONResponse(content=result, status_code=200)
        
    except Exception as exc:
        logger.exception(f"Unexpected error in enroll_voice for user {user_id}: {exc}")
        return JSONResponse(
            content={
                "error": "Internal server error during enrollment",
                "details": str(exc),
                "user_id": user_id
            },
            status_code=500
        )


@router.post(
    "/identify",
    summary="Identify speaker",
    description=(
        "Nhận diện người nói từ audio so với tất cả users đã enroll (≥3 mẫu). "
        "Trả về match nếu vượt ngưỡng tương đồng hiện hành (mặc định ≈0.70)."
    ),
    response_model=IdentificationResponse,
    responses={
        200: {
            "description": "Identification result",
            "content": {
                "application/json": {
                    "example": {
                        "type": "identification",
                        "success": True,
                        "identified": True,
                        "speaker_id": "USR001",
                        "speaker_name": "Nguyễn Văn A",
                        "score": 0.88,
                        "confidence": 0.90,
                        "message": "Speaker identified successfully"
                    }
                }
            },
        },
        400: {"description": "Invalid audio or no enrolled users", "content": {"application/json": {"example": {"error": "No enrolled users found"}}}},
        500: {"description": "Internal server error", "content": {"application/json": {"example": {"error": "Unexpected error"}}}},
    },
)
async def identify_speaker(
    audio_file: UploadFile = File(..., description="Audio file for speaker identification (WAV/MP3/FLAC, max 10MB)"),
    voice_service: VoiceService = Depends(get_voice_service),
):
    """Identify speaker from voice sample.
    
    **REST API:** `POST /voice/identify`
    
    **Process:**
    1. Trích xuất embedding
    2. So sánh với embeddings của toàn bộ users hoàn tất enrollment
    3. Tính cosine similarity -> score; suy ra confidence
    4. `identified=true` nếu score vượt ngưỡng nội bộ (≈0.70)

    **Trả về:**
    - `identified=true/false`
    - `score`: Cosine similarity (0–1)
    - `confidence`: Độ tin cậy (0–1)
    - `speaker_id`, `speaker_name` (nếu match)
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
    summary="Verify voice",
    description=(
        "Xác thực audio có khớp user đã cho (1:1). Yêu cầu user đủ 3 mẫu enroll. "
        "Dùng điểm tương đồng và ngưỡng nội bộ (≈0.70)."
    ),
    response_model=VerificationResponse,
    responses={
        200: {
            "description": "Verification result",
            "content": {
                "application/json": {
                    "example": {
                        "type": "verification",
                        "success": True,
                        "verified": True,
                        "claimed_id": "USR001",
                        "speaker_id": "USR001",
                        "match": True,
                        "score": 0.91,
                        "confidence": 0.94,
                        "message": "Voice verified successfully"
                    }
                }
            },
        },
        400: {"description": "Invalid audio or user not enrolled", "content": {"application/json": {"example": {"error": "User not enrolled or insufficient samples"}}}},
        500: {"description": "Internal server error", "content": {"application/json": {"example": {"error": "Unexpected error"}}}},
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
    1. Kiểm tra user đủ mẫu (≥3)
    2. Trích xuất embedding từ audio verify
    3. So sánh với embeddings đã lưu
    4. `verified=true` nếu score vượt ngưỡng nội bộ (≈0.70)

    **Trả về:**
    - `match`: claimed_id == speaker_id
    - `verified`: trạng thái xác thực
    - `score`, `confidence`

    **Use Cases:**
    - Xác thực danh tính
    - Kiểm soát truy cập
    - Phát hiện giả mạo giọng nói
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
