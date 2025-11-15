"""Voice API schemas - Request/Response DTOs for Swagger documentation."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ============================================================================
# Request Models
# ============================================================================

class AudioFileUpload(BaseModel):
    """Audio file upload documentation (not used in FastAPI, just for Swagger)."""
    audio_file: bytes = Field(..., description="Audio file (WAV/MP3/FLAC, max 10MB)")


# ============================================================================
# Response Models - Enrollment
# ============================================================================

class EnrollmentResponse(BaseModel):
    """Response from voice enrollment endpoint."""
    success: bool = Field(..., description="Whether enrollment was successful")
    message: str = Field(..., description="Human-readable result message")
    enrollment_count: int = Field(..., description="Total enrollment samples for this user")
    user_id: str = Field(..., alias="id", description="User identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "message": "Enrollment successful",
                "enrollment_count": 3,
                "id": "usr001"
            }
        }


# ============================================================================
# Response Models - Identification
# ============================================================================

class IdentificationResponse(BaseModel):
    """Response from speaker identification endpoint."""
    type: str = Field(default="identify", description="Response type")
    success: bool = Field(..., description="Whether identification was successful")
    identified: bool = Field(..., description="Whether a speaker was identified")
    speaker_id: Optional[str] = Field(None, description="Identified speaker's user ID")
    speaker_name: Optional[str] = Field(None, description="Identified speaker's display name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")
    message: str = Field(..., description="Human-readable result message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "identify",
                "success": True,
                "identified": True,
                "speaker_id": "usr001",
                "speaker_name": "Nguyễn Văn A",
                "confidence": 0.92,
                "score": 0.89,
                "message": "Speaker identified successfully"
            }
        }


# ============================================================================
# Response Models - Verification
# ============================================================================

class VerificationResponse(BaseModel):
    """Response from voice verification endpoint."""
    type: str = Field(default="verify", description="Response type")
    success: bool = Field(..., description="Whether verification was successful")
    verified: bool = Field(..., description="Whether voice was verified")
    claimed_id: str = Field(..., description="User ID claimed for verification")
    speaker_id: Optional[str] = Field(None, description="Actual identified speaker ID (if match)")
    match: bool = Field(..., description="Whether claimed_id matches speaker_id")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score (0-1)")
    score: float = Field(..., ge=0.0, le=1.0, description="Similarity score (0-1)")
    message: str = Field(..., description="Human-readable result message")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "verify",
                "success": True,
                "verified": True,
                "claimed_id": "usr001",
                "speaker_id": "usr001",
                "match": True,
                "confidence": 0.94,
                "score": 0.91,
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

