"""Question duplicate detection service using Redis and fuzzy matching."""
import asyncio
import json
import logging
from datetime import datetime
from rapidfuzz import fuzz
import string
from typing import Optional
from repositories.interfaces.i_redis_service import IRedisService
from services.interfaces.i_question_service import IQuestionService

logger = logging.getLogger(__name__)


class QuestionService(IQuestionService):
    """Service for detecting duplicate questions in a session."""
    
    def __init__(self, redis_service: Optional[IRedisService] = None, session_ttl: int = 7200):
        """Initialize with Redis service interface."""
        self.session_ttl = session_ttl
        self._redis_service = redis_service
        self._semantic_model = None  # Lazy load SBERT
        # Local locks per session (for single-container atomicity)
        self._session_locks: dict[str, asyncio.Lock] = {}
    
    @property
    def redis_service(self) -> IRedisService:
        """Lazy load Redis service."""
        if self._redis_service is None:
            from services.redis_service import get_redis_service
            self._redis_service = get_redis_service()
        return self._redis_service
    
    @property
    def semantic_model(self):
        """Lazy load SBERT model for semantic similarity (Vietnamese multilingual)."""
        if self._semantic_model is None:
            from sentence_transformers import SentenceTransformer
            # Use multilingual model for Vietnamese support
            self._semantic_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        return self._semantic_model
    
    def _get_session_key(self, session_id: str) -> str:
        """Get Redis key for session questions."""
        return f"questions:session:{session_id}"
    
    def _get_session_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create lock for a session (prevents race conditions)."""
        if session_id not in self._session_locks:
            self._session_locks[session_id] = asyncio.Lock()
        return self._session_locks[session_id]
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        text = text.lower().strip()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return ' '.join(text.split())
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (0.0 to 1.0)."""
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        
        # Weighted average of different algorithms
        simple_ratio = fuzz.ratio(norm1, norm2) / 100.0
        token_sort = fuzz.token_sort_ratio(norm1, norm2) / 100.0
        token_set = fuzz.token_set_ratio(norm1, norm2) / 100.0
        
        # Weight: simple 60%, token_sort 30%, token_set 10%
        return (simple_ratio * 0.6) + (token_sort * 0.3) + (token_set * 0.1)
    
    async def check_duplicate(
        self,
        session_id: str,
        question_text: str,
        threshold: float = 0.85,
        semantic_threshold: float = 0.85
    ) -> tuple[bool, list[dict]]:
        """Check if question is duplicate (fuzzy + semantic).
        
        Logic: Coi là trùng nếu:
        - Fuzzy >= 0.85 AND Semantic >= 0.70 (cả 2 khá cao)
        - HOẶC Fuzzy >= 0.95 (gần như giống hệt về từ)
        - HOẶC Semantic >= 0.85 (giống về ý nghĩa)
        """
        key = self._get_session_key(session_id)
        questions = await self.redis_service.get(key)
        if not questions:
            return False, []
        similar_questions = []
        # Semantic embedding for new question
        new_emb = self.semantic_model.encode(question_text, convert_to_tensor=True)
        for q in questions:
            text = q['text']
            fuzzy_score = self._calculate_similarity(question_text, text)
            # Semantic similarity
            q_emb = self.semantic_model.encode(text, convert_to_tensor=True)
            from torch import nn
            cos = nn.functional.cosine_similarity(new_emb, q_emb, dim=0).item()
            
            # Vietnamese-optimized logic: Prioritize semantic similarity for paraphrases
            is_duplicate = (
                (fuzzy_score >= 0.65 and cos >= 0.60) or  # Both moderately high (Vietnamese paraphrases)
                fuzzy_score >= 0.85 or  # Nearly identical wording
                cos >= 0.70  # Strong semantic similarity (multilingual model catches paraphrases)
            )
            
            if is_duplicate:
                similar_questions.append({
                    'text': text,
                    'fuzzy_score': round(fuzzy_score, 4),
                    'semantic_score': round(cos, 4)
                })
        # Sắp xếp theo max(fuzzy, semantic)
        similar_questions.sort(key=lambda x: max(x['fuzzy_score'], x['semantic_score']), reverse=True)
        is_duplicate = len(similar_questions) > 0
        return is_duplicate, similar_questions
    
    async def register_question(
        self,
        session_id: str,
        question_text: str,
        speaker: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> dict:
        """Register a new question in Redis."""
        key = self._get_session_key(session_id)
        
        # Get existing questions (already parsed by RedisService!)
        questions = await self.redis_service.get(key) or []
        
        # Add new question
        question = {
            'text': question_text,
            'speaker': speaker,
            'timestamp': timestamp or datetime.utcnow().isoformat(),
            'id': len(questions) + 1
        }
        questions.append(question)
        
        # Save back to Redis (RedisService will auto-stringify!)
        await self.redis_service.set(key, questions, ttl=self.session_ttl)
        
        return {
            'success': True,
            'question_id': question['id'],
            'total_questions': len(questions)
        }
    
    async def get_questions(self, session_id: str) -> list[str]:
        """Get all questions for a session."""
        key = self._get_session_key(session_id)
        questions = await self.redis_service.get(key)
        
        if not questions:
            return []
        
        return [q['text'] for q in questions]
    
    async def check_and_register(
        self,
        session_id: str,
        question_text: str,
        speaker: str = "Khách",
        timestamp: str = None,
        threshold: float = 0.85,
        semantic_threshold: float = 0.85
    ) -> dict:
        """Atomic: check duplicate and register if unique.
        
        Uses lock to prevent race conditions where two identical questions
        are submitted at the same time and both pass duplicate check.
        """
        # Get lock for this session to ensure atomicity
        lock = self._get_session_lock(session_id)
        
        async with lock:
            logger.info(f"🔒 Acquired lock for session {session_id} | checking: '{question_text[:50]}'")
            
            is_dup, similar = await self.check_duplicate(session_id, question_text, threshold, semantic_threshold)
            
            registered = False
            question_id = None
            
            if not is_dup:
                reg = await self.register_question(session_id, question_text, speaker=speaker, timestamp=timestamp)
                registered = reg.get("success", False)
                question_id = reg.get("question_id")
                logger.info(f"✅ Registered question #{question_id} | text='{question_text[:50]}'")
            else:
                logger.info(f"🔄 Duplicate detected | text='{question_text[:50]}' | similar_count={len(similar)}")
            
            # Get total questions count (after possible registration)
            existing = await self.get_questions(session_id)
            
            logger.info(f"🔓 Released lock for session {session_id}")
            
            return {
                "is_duplicate": is_dup,
                "similar": similar,
                "registered": registered,
                "question_id": question_id,
                "total_questions": len(existing),
                "question_text": question_text,
            }
    
    async def clear_questions(self, session_id: str) -> int:
        """Clear all questions for a session."""
        key = self._get_session_key(session_id)
        questions = await self.redis_service.get(key)
        
        count = 0
        if questions:
            count = len(questions)
        
        await self.redis_service.delete(key)
        return count

