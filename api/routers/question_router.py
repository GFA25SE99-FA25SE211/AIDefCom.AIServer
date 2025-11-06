"""REST API endpoints for question duplicate detection."""
import traceback
from fastapi import APIRouter, HTTPException
from api.schemas.question_schemas import (
    QuestionCheckRequest,
    QuestionCheckResponse,
    QuestionRegisterRequest,
    QuestionRegisterResponse,
    QuestionListResponse,
    SimilarQuestion,
)
from services.question_service import QuestionService

router = APIRouter(prefix="/questions", tags=["Questions"])

# Initialize service
question_service = QuestionService(session_ttl=7200)


@router.post("/check-duplicate", response_model=QuestionCheckResponse)
async def check_duplicate(request: QuestionCheckRequest):
    """Check if a question is duplicate in the session."""
    try:
        is_duplicate, similar = await question_service.check_duplicate(
            session_id=request.session_id,
            question_text=request.question_text,
            threshold=request.threshold,
        )
        
        similar_questions = [
            SimilarQuestion(
                text=s['text'],
                score=max(s.get('fuzzy_score', 0), s.get('semantic_score', 0)),
                fuzzy_score=s.get('fuzzy_score'),
                semantic_score=s.get('semantic_score')
            )
            for s in similar
        ]
        
        if is_duplicate:
            message = f"⚠️ Câu hỏi trùng lặp! Tìm thấy {len(similar_questions)} câu tương tự."
        else:
            message = "✅ Câu hỏi hợp lệ, chưa bị trùng."
        
        return QuestionCheckResponse(
            is_duplicate=is_duplicate,
            question_text=request.question_text,
            similar_questions=similar_questions,
            message=message,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register", response_model=QuestionRegisterResponse)
async def register_question(request: QuestionRegisterRequest):
    """Register a new question in the session."""
    try:
        result = await question_service.register_question(
            session_id=request.session_id,
            question_text=request.question_text,
            speaker=request.speaker,
            timestamp=request.timestamp,
        )
        
        return QuestionRegisterResponse(
            success=result['success'],
            question_id=result['question_id'],
            total_questions=result['total_questions'],
            message=f"✅ Câu hỏi đã được lưu. Tổng: {result['total_questions']}",
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/check-and-register", response_model=QuestionCheckResponse)
async def check_and_register(request: QuestionRegisterRequest):
    """Check for duplicate and register if not duplicate (combo endpoint)."""
    try:
        print(f"🔍 API: Checking duplicate for: {request.question_text[:50]}...")
        
        # First check duplicate
        is_duplicate, similar = await question_service.check_duplicate(
            session_id=request.session_id,
            question_text=request.question_text,
            threshold=0.85,
        )
        
        print(f"✅ API: Check complete - is_duplicate={is_duplicate}, similar={len(similar)}")
        
        similar_questions = [
            SimilarQuestion(
                text=s['text'],
                score=max(s.get('fuzzy_score', 0), s.get('semantic_score', 0)),
                fuzzy_score=s.get('fuzzy_score'),
                semantic_score=s.get('semantic_score')
            )
            for s in similar
        ]
        
        if is_duplicate:
            message = f"⚠️ Câu hỏi trùng lặp! Không thể đăng ký."
        else:
            # Register if not duplicate
            print(f"💾 API: Registering question...")
            result = await question_service.register_question(
                session_id=request.session_id,
                question_text=request.question_text,
                speaker=request.speaker,
                timestamp=request.timestamp,
            )
            message = f"✅ Câu hỏi đã được lưu. Tổng: {result['total_questions']}"
        
        return QuestionCheckResponse(
            is_duplicate=is_duplicate,
            question_text=request.question_text,
            similar_questions=similar_questions,
            message=message,
        )
    
    except Exception as e:
        print(f"❌ API ERROR: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/session/{session_id}", response_model=QuestionListResponse)
async def get_session_questions(session_id: str):
    """Get all questions for a session."""
    try:
        questions = await question_service.get_questions(session_id)
        
        return QuestionListResponse(
            session_id=session_id,
            questions=questions,
            total=len(questions),
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/session/{session_id}")
async def clear_session_questions(session_id: str):
    """Clear all questions for a session."""
    try:
        deleted = await question_service.clear_questions(session_id)
        
        return {
            "success": True,
            "session_id": session_id,
            "deleted": deleted,
            "message": f"✅ Đã xóa {deleted} câu hỏi.",
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
