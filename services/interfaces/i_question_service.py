"""Question Service Interface - Abstract base for question duplicate detection."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any


class IQuestionService(ABC):
    """Interface for question duplicate detection business logic."""
    
    @abstractmethod
    async def check_duplicate(
        self,
        session_id: str,
        question_text: str,
        threshold: float = 0.85,
        semantic_threshold: float = 0.85
    ) -> Tuple[bool, List[Dict]]:
        """
        Check if question is duplicate using hybrid fuzzy + semantic matching.
        
        Args:
            session_id: Session identifier
            question_text: Question text to check
            threshold: Fuzzy similarity threshold (default 0.85)
            semantic_threshold: Semantic similarity threshold (default 0.85)
            
        Returns:
            Tuple of (is_duplicate: bool, similar_questions: List[Dict])
        """
        pass
    
    @abstractmethod
    async def register_question(
        self,
        session_id: str,
        question_text: str,
        speaker: str,
        timestamp: str
    ) -> Dict[str, Any]:
        """
        Register new question without duplicate check.
        
        Args:
            session_id: Session identifier
            question_text: Question text
            speaker: Speaker name (default "Khách")
            timestamp: UTC ISO 8601 timestamp
            
        Returns:
            Dict with success, question_id, total_questions
        """
        pass
    
    @abstractmethod
    async def get_questions(self, session_id: str) -> List[str]:
        """
        Retrieve all questions for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            List of question text strings
        """
        pass
    
    @abstractmethod
    async def check_and_register(
        self,
        session_id: str,
        question_text: str,
        speaker: str = "Khách",
        timestamp: str = None,
        threshold: float = 0.85,
        semantic_threshold: float = 0.85
    ) -> Dict[str, Any]:
        """
        Atomic operation: check duplicate and register if not duplicate.
        
        Args:
            session_id: Session identifier
            question_text: Question text to check and register
            speaker: Speaker name (default "Khách")
            timestamp: UTC ISO 8601 timestamp (auto-generated if None)
            threshold: Fuzzy similarity threshold (default 0.85)
            semantic_threshold: Semantic similarity threshold (default 0.85)
            
        Returns:
            Dict with is_duplicate, similar, registered, question_id, total_questions, question_text
        """
        pass
    
    @abstractmethod
    async def clear_questions(self, session_id: str) -> int:
        """
        Clear all questions for session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Count of deleted questions
        """
        pass
