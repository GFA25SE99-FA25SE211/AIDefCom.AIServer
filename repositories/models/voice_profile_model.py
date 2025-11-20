"""Type-safe voice profile schema using Pydantic."""

from __future__ import annotations

from typing import Any, List, Optional
from pydantic import BaseModel, Field, validator


class VoiceSample(BaseModel):
    sample_id: str
    session_id: str
    created_at: str
    metrics: Optional[dict[str, Any]] = None
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
        dim = values.get("embedding_dim")
        for emb in v:
            if len(emb) != dim:
                raise ValueError("Embedding dimension mismatch")
        return v

    class Config:
        extra = "ignore"
