"""Schemas for question duplicate detection."""
from pydantic import BaseModel, Field
from typing import Optional


class QuestionCheckRequest(BaseModel):
    """Request payload for duplicate question check."""
    session_id: str = Field(..., description="Session identifier")
    question_text: str = Field(..., min_length=3, description="Question text to check for duplication")

    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_123",
                "question_text": "AI là gì?"
            }
        }


class SimilarQuestion(BaseModel):
    """Similar question found."""
    text: str = Field(..., description="Similar question text")
    score: float = Field(0.0, description="Combined similarity score (0.0-1.0)")
    fuzzy_score: Optional[float] = Field(None, description="Fuzzy matching score")
    semantic_score: Optional[float] = Field(None, description="Semantic similarity score")


class QuestionCheckResponse(BaseModel):
    """Response payload for duplicate question check."""
    is_duplicate: bool = Field(..., description="True if similar question(s) found")
    question_text: str = Field(..., description="Original question text submitted")
    similar_questions: list[SimilarQuestion] = Field(default_factory=list, description="List of similar questions with scores")
    message: str = Field(..., description="Localized status message")

    class Config:
        schema_extra = {
            "example": {
                "is_duplicate": True,
                "question_text": "AI là gì?",
                "similar_questions": [
                    {"text": "Trí tuệ nhân tạo là gì?", "score": 0.92, "fuzzy_score": 0.85, "semantic_score": 0.92}
                ],
                "message": "⚠️ Câu hỏi trùng lặp! Tìm thấy 1 câu tương tự."
            }
        }


class QuestionRegisterRequest(BaseModel):
    """Request payload for question registration."""
    session_id: str = Field(..., description="Session identifier")
    question_text: str = Field(..., min_length=3, description="Question text to register")

    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_123",
                "question_text": "Machine Learning hoạt động thế nào?"
            }
        }


class QuestionRegisterResponse(BaseModel):
    """Response payload for successful question registration."""
    success: bool = Field(..., description="True if registration succeeded")
    question_id: int = Field(..., description="Sequential numeric ID of registered question in session")
    total_questions: int = Field(..., description="Total number of questions after registration")
    message: str = Field(..., description="Localized status message")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "question_id": 6,
                "total_questions": 6,
                "message": "✅ Câu hỏi đã được lưu. Tổng: 6"
            }
        }


class QuestionListResponse(BaseModel):
    """Response payload for listing all questions in a session."""
    session_id: str = Field(..., description="Session identifier")
    questions: list[str] = Field(..., description="List of question texts in insertion order")
    total: int = Field(..., description="Total number of questions")

    class Config:
        schema_extra = {
            "example": {
                "session_id": "session_123",
                "questions": ["AI là gì?", "Machine Learning hoạt động thế nào?"],
                "total": 2
            }
        }


class ClearQuestionsResponse(BaseModel):
    """Response payload for clearing all questions in a session."""
    success: bool = Field(..., description="True if clear operation succeeded")
    session_id: str = Field(..., description="Session identifier")
    deleted: int = Field(..., description="Number of questions deleted")
    message: str = Field(..., description="Localized status message")

    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "session_id": "session_123",
                "deleted": 5,
                "message": "✅ Đã xóa 5 câu hỏi."
            }
        }
