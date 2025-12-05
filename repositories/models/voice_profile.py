"""Voice Profile Data Model - Pydantic schema."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


class VoiceSample(BaseModel):
    sample_id: str
    session_id: str
    created_at: str
    metrics: Optional[Dict[str, Any]] = None
    embedding: List[float]


class VoiceProfile(BaseModel):
    user_id: str
    name: str
    enrollment_status: str
    voice_samples: List[VoiceSample] = Field(default_factory=list)
    voice_embeddings: List[List[float]] = Field(default_factory=list)
    embedding_dim: int
    created_at: str
    updated_at: str
    schema_version: int = 5
    mean_embedding: Optional[List[float]] = None
    within_var: Optional[float] = None
    sigma: Optional[float] = None
    enrollment_count: int = 0

    @validator("enrollment_status")
    def validate_status(cls, v: str) -> str:
        allowed = {"not_enrolled", "enrolled", "partial"}
        if v not in allowed:
            raise ValueError(f"Invalid enrollment_status: {v}")
        return v

    @validator("voice_embeddings", always=True)
    def validate_embeddings(cls, v: List[List[float]], values):
        """Validate embedding dimensions."""
        dim = values.get("embedding_dim")
        if not dim or not v:
            return v
        
        if v and len(v[0]) != dim:
            user_id = values.get("user_id", "unknown")
            actual_dim = len(v[0])
            logger.warning(
                f"Embedding dimension mismatch for {user_id}: "
                f"expected {dim}, got {actual_dim}."
            )
        return v

    class Config:
        extra = "ignore"
        validate_assignment = True
