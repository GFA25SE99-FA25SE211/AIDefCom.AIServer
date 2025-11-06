"""Schemas for question duplicate detection."""
from pydantic import BaseModel, Field
from typing import Optional


class QuestionCheckRequest(BaseModel):
    """Request to check if a question is duplicate."""
    session_id: str = Field(..., description="Session identifier")
    question_text: str = Field(..., min_length=3, description="Question text to check")
    threshold: float = Field(0.85, ge=0.0, le=1.0, description="Similarity threshold (0.0-1.0)")


class SimilarQuestion(BaseModel):
    """Similar question found."""
    text: str = Field(..., description="Similar question text")
    score: float = Field(0.0, description="Combined similarity score (0.0-1.0)")
    fuzzy_score: Optional[float] = Field(None, description="Fuzzy matching score")
    semantic_score: Optional[float] = Field(None, description="Semantic similarity score")


class QuestionCheckResponse(BaseModel):
    """Response for question duplicate check."""
    is_duplicate: bool = Field(..., description="Whether question is duplicate")
    question_text: str = Field(..., description="Original question text")
    similar_questions: list[SimilarQuestion] = Field(default_factory=list, description="List of similar questions")
    message: str = Field(..., description="Human-readable message")


class QuestionRegisterRequest(BaseModel):
    """Request to register a new question."""
    session_id: str = Field(..., description="Session identifier")
    question_text: str = Field(..., min_length=3, description="Question text to register")
    speaker: Optional[str] = Field(None, description="Speaker name")
    timestamp: Optional[str] = Field(None, description="Timestamp of question")


class QuestionRegisterResponse(BaseModel):
    """Response for question registration."""
    success: bool = Field(..., description="Whether registration was successful")
    question_id: int = Field(..., description="ID of registered question")
    total_questions: int = Field(..., description="Total questions in session")
    message: str = Field(..., description="Human-readable message")


class QuestionListResponse(BaseModel):
    """Response for listing questions."""
    session_id: str = Field(..., description="Session identifier")
    questions: list[str] = Field(..., description="List of question texts")
    total: int = Field(..., description="Total number of questions")
