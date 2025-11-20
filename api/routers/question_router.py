"""REST API endpoints for question duplicate detection (Swagger-enhanced)."""
import traceback
from fastapi import APIRouter, HTTPException
from api.schemas.question_schemas import (
    QuestionCheckRequest,
    QuestionCheckResponse,
    QuestionRegisterRequest,
    QuestionRegisterResponse,
    QuestionListResponse,
    ClearQuestionsResponse,
    SimilarQuestion,
)
from services.question_service import QuestionService

router = APIRouter(prefix="/questions", tags=["Questions"])

# Initialize service (TTL 2h)
question_service = QuestionService(session_ttl=7200)


@router.post(
    "/check-duplicate",
    summary="Check duplicate question",
    description="Kiểm tra câu hỏi có bị trùng trong session (fuzzy + semantic, ngưỡng nội bộ 0.85).",
    response_model=QuestionCheckResponse,
    responses={
        200: {"description": "Kết quả kiểm tra câu hỏi"},
        500: {"description": "Lỗi hệ thống"},
    },
)
async def check_duplicate(request: QuestionCheckRequest):
    """Check if a question is duplicate in the session (threshold=0.85)."""
    try:
        is_duplicate, similar = await question_service.check_duplicate(
            session_id=request.session_id,
            question_text=request.question_text,
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


@router.post(
    "/register",
    summary="Register question",
    description="Đăng ký câu hỏi mới vào session (không kiểm tra trùng).",
    response_model=QuestionRegisterResponse,
    responses={
        200: {"description": "Đăng ký thành công"},
        500: {"description": "Lỗi hệ thống"},
    },
)
async def register_question(request: QuestionRegisterRequest):
    """Register a new question (speaker='Khách', timestamp tự sinh)."""
    try:
        from datetime import datetime

        result = await question_service.register_question(
            session_id=request.session_id,
            question_text=request.question_text,
            speaker="Khách",
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        return QuestionRegisterResponse(
            success=result['success'],
            question_id=result['question_id'],
            total_questions=result['total_questions'],
            message=f"✅ Câu hỏi đã được lưu. Tổng: {result['total_questions']}",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post(
    "/check-and-register",
    summary="Check & register question",
    description="Kiểm tra trùng lặp và tự động đăng ký nếu không trùng (ngưỡng nội bộ 0.85).",
    response_model=QuestionCheckResponse,
    responses={
        200: {"description": "Kết quả kiểm tra/đăng ký"},
        500: {"description": "Lỗi hệ thống"},
    },
)
async def check_and_register(request: QuestionRegisterRequest):
    """Check duplicate then register if unique (threshold=0.85)."""
    try:
        print(f"🔍 API: Checking duplicate for: {request.question_text[:50]}...")

        is_duplicate, similar = await question_service.check_duplicate(
            session_id=request.session_id,
            question_text=request.question_text,
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
            message = "⚠️ Câu hỏi trùng lặp! Không thể đăng ký."
        else:
            from datetime import datetime
            print("💾 API: Registering question...")
            result = await question_service.register_question(
                session_id=request.session_id,
                question_text=request.question_text,
                speaker="Khách",
                timestamp=datetime.utcnow().isoformat() + "Z",
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


@router.get(
    "/session/{session_id}",
    summary="List session questions",
    description="Lấy tất cả câu hỏi trong session theo thứ tự thêm vào.",
    response_model=QuestionListResponse,
    responses={
        200: {"description": "Danh sách câu hỏi"},
        500: {"description": "Lỗi hệ thống"},
    },
)
async def get_session_questions(session_id: str):
    """List all questions in a session."""
    try:
        questions = await question_service.get_questions(session_id)
        return QuestionListResponse(
            session_id=session_id,
            questions=questions,
            total=len(questions),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete(
    "/session/{session_id}",
    summary="Clear session questions",
    description="Xóa toàn bộ câu hỏi trong session và trả về số lượng đã xóa.",
    response_model=ClearQuestionsResponse,
    responses={
        200: {"description": "Kết quả xóa"},
        500: {"description": "Lỗi hệ thống"},
    },
)
async def clear_session_questions(session_id: str):
    """Clear all questions for a session and return count."""
    try:
        deleted = await question_service.clear_questions(session_id)
        return ClearQuestionsResponse(
            success=True,
            session_id=session_id,
            deleted=deleted,
            message=f"✅ Đã xóa {deleted} câu hỏi.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
