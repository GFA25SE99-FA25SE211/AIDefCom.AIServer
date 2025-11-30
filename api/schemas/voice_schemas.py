"""Voice API schemas - Request/Response DTOs for Swagger documentation."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ==============================================================
# Request Models
# ==============================================================

class AudioFileUpload(BaseModel):
    """Audio file upload documentation (not used in FastAPI, just for Swagger)."""
    audio_file: bytes = Field(..., description="Audio file (WAV/MP3/FLAC, max 10MB)")


# ============================================================================
# Response Models - Enrollment
# ============================================================================

class EnrollmentResponse(BaseModel):
    """Response from voice enrollment endpoint."""
    type: str = Field(default="enrollment", description="Response type identifier")
    success: bool = Field(..., description="Whether enrollment was successful")
    user_id: str = Field(..., description="User identifier")
    enrollment_count: int = Field(..., description="Total enrollment samples for this user")
    min_required: int = Field(default=3, description="Minimum samples required for full enrollment")
    is_complete: bool = Field(..., description="Whether enrollment has reached minimum sample count")
    message: str = Field(..., description="Human-readable result message")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "enrollment",
                "success": True,
                "user_id": "USR001",
                "enrollment_count": 3,
                "min_required": 3,
                "is_complete": True,
                "message": "Enrollment sample 3/3 saved successfully"
            }
        }


# ============================================================================
# Response Models - Identification
# ============================================================================

class IdentificationResponse(BaseModel):
    """Response from speaker identification endpoint."""
    type: str = Field(default="identification", description="Response type identifier")
    success: bool = Field(..., description="Whether identification process executed successfully")
    identified: bool = Field(..., description="Whether a speaker passed similarity threshold")
    speaker_id: Optional[str] = Field(None, description="User ID of identified speaker if match")
    speaker_name: Optional[str] = Field(None, description="Display name of identified speaker")
    score: float = Field(..., ge=0.0, le=1.0, description="Raw similarity score (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Derived confidence (0-1)")
    message: str = Field(..., description="Human-readable result message")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "identification",
                "success": True,
                "identified": True,
                "speaker_id": "USR001",
                "speaker_name": "Nguyễn Văn A",
                "score": 0.89,
                "confidence": 0.92,
                "message": "Speaker identified successfully"
            }
        }


# ============================================================================
# Response Models - Verification
# ============================================================================

class VerificationResponse(BaseModel):
    """Response from voice verification endpoint."""
    type: str = Field(default="verification", description="Response type identifier")
    success: bool = Field(..., description="Whether verification process executed successfully")
    verified: bool = Field(..., description="Whether voice matched claimed user")
    claimed_id: str = Field(..., description="User ID being verified")
    speaker_id: Optional[str] = Field(None, description="Identified speaker's user ID")
    match: bool = Field(..., description="True if claimed_id == speaker_id")
    score: float = Field(..., ge=0.0, le=1.0, description="Raw similarity score (0-1)")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Derived confidence (0-1)")
    message: str = Field(..., description="Human-readable result message")

    class Config:
        json_schema_extra = {
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


# ============================================================================
# Error Response Model
# ============================================================================

class ErrorResponse(BaseModel):
    """Error response from any endpoint."""
    error: str = Field(..., description="Error message describing what went wrong")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Empty audio data"
            }
        }

