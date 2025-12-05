"""Question duplicate detection service using Redis and fuzzy matching."""
import asyncio
import logging
import string
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
from rapidfuzz import fuzz

from repositories.interfaces import IRedisService
from services.interfaces.i_question_service import IQuestionService
from services.voice.vector_index import SessionVectorIndex, create_vector_index
from core.executors import run_cpu_bound

logger = logging.getLogger(__name__)


def _normalize_text_sync(text: str) -> str:
    """Normalize text for comparison (sync, CPU-bound)."""
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(text.split())


def _calculate_fuzzy_similarity_sync(text1: str, text2: str) -> float:
    """Calculate fuzzy similarity between two texts (sync, CPU-bound).
    
    This is CPU-intensive due to RapidFuzz algorithms.
    """
    norm1 = _normalize_text_sync(text1)
    norm2 = _normalize_text_sync(text2)
    
    # Weighted average of different algorithms
    simple_ratio = fuzz.ratio(norm1, norm2) / 100.0
    token_sort = fuzz.token_sort_ratio(norm1, norm2) / 100.0
    token_set = fuzz.token_set_ratio(norm1, norm2) / 100.0
    
    # Weight: simple 60%, token_sort 30%, token_set 10%
    return (simple_ratio * 0.6) + (token_sort * 0.3) + (token_set * 0.1)


def _encode_text_sync(model, text: str) -> np.ndarray:
    """Encode single text to embedding vector (sync, CPU-bound)."""
    embedding = model.encode(text, convert_to_tensor=False, show_progress_bar=False)
    return np.asarray(embedding, dtype=np.float32)


def _encode_texts_batch_sync(model, texts: List[str]) -> np.ndarray:
    """Encode multiple texts to embedding matrix (sync, CPU-bound).
    
    Returns:
        Numpy array of shape (N, dim) with L2-normalized embeddings
    """
    embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    
    # L2 normalize for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    return embeddings / norms


def _check_duplicates_with_index_sync(
    model,
    question_text: str,
    questions: List[Dict[str, Any]],
    existing_embeddings: Optional[np.ndarray],
    fuzzy_threshold: float = 0.85,
    semantic_threshold: float = 0.60,
) -> tuple[List[Dict[str, Any]], np.ndarray]:
    """Check for duplicate questions using vectorized operations (sync, CPU-bound).
    
    Uses matrix multiplication for O(1) similarity computation instead of O(N) loop.
    
    Returns:
        Tuple of (similar_questions, query_embedding)
    """
    if not questions:
        return [], _encode_text_sync(model, question_text)
    
    existing_texts = [q['text'] for q in questions]
    
    # Encode query text
    query_embedding = _encode_text_sync(model, question_text)
    query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
    
    # Get or compute existing embeddings
    if existing_embeddings is None or len(existing_embeddings) != len(questions):
        # Batch encode all existing texts
        existing_embeddings = _encode_texts_batch_sync(model, existing_texts)
    
    # VECTORIZED: Compute all cosine similarities at once via matrix multiplication
    # This is O(1) matrix op vs O(N) Python loop
    semantic_scores = existing_embeddings @ query_norm  # Shape: (N,)
    
    similar_questions = []
    
    for i, q in enumerate(questions):
        text = q['text']
        
        # Fuzzy similarity (still need to compute per-pair, but RapidFuzz is C-optimized)
        fuzzy_score = _calculate_fuzzy_similarity_sync(question_text, text)
        
        # Semantic similarity (already computed via matrix multiply)
        cos = float(semantic_scores[i])
        
        # Vietnamese-optimized duplicate detection logic
        is_duplicate = (
            (fuzzy_score >= 0.65 and cos >= 0.60) or  # Both moderately high
            fuzzy_score >= 0.85 or                     # Nearly identical wording
            cos >= 0.70                                 # Strong semantic similarity
        )
        
        if is_duplicate:
            similar_questions.append({
                'text': text,
                'fuzzy_score': round(fuzzy_score, 4),
                'semantic_score': round(cos, 4)
            })
    
    # Sort by max score
    similar_questions.sort(
        key=lambda x: max(x['fuzzy_score'], x['semantic_score']),
        reverse=True
    )
    
    return similar_questions, query_embedding


class QuestionService(IQuestionService):
    """Service for detecting duplicate questions in a session.
    
    Optimizations:
    1. Caches embeddings per session to avoid re-encoding
    2. Uses matrix multiplication for O(1) similarity computation
    3. Supports FAISS vector index for very large sessions (> 100 questions)
    """
    
    # Embedding dimension for paraphrase-multilingual-MiniLM-L12-v2
    EMBEDDING_DIM = 384
    
    def __init__(self, redis_service: Optional[IRedisService] = None, session_ttl: int = 7200):
        """Initialize with Redis service interface."""
        self.session_ttl = session_ttl
        self._redis_service = redis_service
        self._semantic_model = None  # Lazy load SBERT
        # Local locks per session (for single-container atomicity)
        self._session_locks: dict[str, asyncio.Lock] = {}
        # Embedding cache per session: session_id -> np.ndarray (N, dim)
        self._embedding_cache: Dict[str, np.ndarray] = {}
        # Vector index for fast similarity search (for large sessions)
        self._vector_index = SessionVectorIndex(dim=self.EMBEDDING_DIM, ttl_seconds=session_ttl)
    
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
        """Normalize text for comparison (delegates to sync function)."""
        return _normalize_text_sync(text)
    
    async def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts (0.0 to 1.0).
        
        Offloads CPU-intensive fuzzy matching to thread pool.
        """
        return await run_cpu_bound(_calculate_fuzzy_similarity_sync, text1, text2)
    
    async def check_duplicate(
        self,
        session_id: str,
        question_text: str,
        threshold: float = 0.85,
        semantic_threshold: float = 0.85
    ) -> tuple[bool, list[dict]]:
        """Check if question is duplicate (fuzzy + semantic).
        
        OPTIMIZED: Uses cached embeddings and matrix multiplication.
        - First call: Encodes all questions, caches embeddings
        - Subsequent calls: Reuses cached embeddings
        - Similarity computation: O(1) matrix-vector multiply vs O(N) loop
        
        Logic: Coi là trùng nếu:
        - Fuzzy >= 0.65 AND Semantic >= 0.60 (cả 2 moderately high)
        - HOẶC Fuzzy >= 0.85 (gần như giống hệt về từ)
        - HOẶC Semantic >= 0.70 (giống về ý nghĩa)
        """
        key = self._get_session_key(session_id)
        questions = await self.redis_service.get(key)
        
        if not questions:
            return False, []
        
        # Get cached embeddings for this session
        cached_embeddings = self._embedding_cache.get(session_id)
        
        # Offload CPU-intensive work to executor
        similar_questions, query_embedding = await run_cpu_bound(
            _check_duplicates_with_index_sync,
            self.semantic_model,
            question_text,
            questions,
            cached_embeddings,
            threshold,
            semantic_threshold,
        )
        
        # Update cache with any new embeddings computed
        # Note: cache will be fully rebuilt on register_question
        
        is_duplicate = len(similar_questions) > 0
        return is_duplicate, similar_questions
    
    async def register_question(
        self,
        session_id: str,
        question_text: str,
        speaker: Optional[str] = None,
        timestamp: Optional[str] = None
    ) -> dict:
        """Register a new question in Redis and update embedding cache."""
        key = self._get_session_key(session_id)
        
        # Get existing questions
        questions = await self.redis_service.get(key) or []
        
        # Add new question
        question = {
            'text': question_text,
            'speaker': speaker,
            'timestamp': timestamp or datetime.utcnow().isoformat(),
            'id': len(questions) + 1
        }
        questions.append(question)
        
        # Save back to Redis
        save_result = await self.redis_service.set(key, questions, ttl=self.session_ttl)
        
        # Update embedding cache
        try:
            new_embedding = await run_cpu_bound(
                _encode_text_sync,
                self.semantic_model,
                question_text,
            )
            norm = np.linalg.norm(new_embedding)
            if norm > 1e-8:
                new_embedding = new_embedding / norm
            
            if session_id in self._embedding_cache:
                self._embedding_cache[session_id] = np.vstack([
                    self._embedding_cache[session_id],
                    new_embedding.reshape(1, -1)
                ])
            else:
                self._embedding_cache[session_id] = new_embedding.reshape(1, -1)
        except Exception as e:
            logger.warning(f"Failed to update embedding cache: {e}")
        
        return {
            'success': save_result,
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
        lock = self._get_session_lock(session_id)
        
        async with lock:
            is_dup, similar = await self.check_duplicate(session_id, question_text, threshold, semantic_threshold)
            
            registered = False
            question_id = None
            
            if not is_dup:
                reg = await self.register_question(session_id, question_text, speaker=speaker, timestamp=timestamp)
                registered = reg.get("success", False)
                question_id = reg.get("question_id")
            
            existing = await self.get_questions(session_id)
            
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
        
        # Clear embedding cache for this session
        if session_id in self._embedding_cache:
            del self._embedding_cache[session_id]
        
        # Clear vector index
        self._vector_index.clear_session(session_id)
        
        return count

