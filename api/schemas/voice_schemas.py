"""Voice API schemas - Request/Response DTOs."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ProfileCreateRequest(BaseModel):
    """Request to create voice profile."""
    user_id: str = Field(..., description="Unique user identifier")
    name: Optional[str] = Field(None, description="User's display name")


class ProfileResponse(BaseModel):
    """Voice profile response."""
    user_id: str
    name: str
    enrollment_status: str
    enrollment_count: int


class EnrollmentResponse(BaseModel):
    """Enrollment response."""
    user_id: str
    name: str
    enrollment_status: str
    enrollment_count: int
    remaining_samples: int
    quality: dict


class IdentificationResponse(BaseModel):
    """Speaker identification response."""
    identified: bool
    speaker: str
    score: float
    user_id: Optional[str] = None
    confidence: Optional[str] = None
    quality: dict


class VerificationResponse(BaseModel):
    """Voice verification response."""
    user_id: str
    verified: bool
    score: float
    confidence: Optional[str] = None
    quality: dict
