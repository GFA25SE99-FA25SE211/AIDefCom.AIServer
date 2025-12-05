"""Voice Profile Repository Interface."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Dict, List
import numpy as np


class IVoiceProfileRepository(ABC):
    """Interface for voice profile persistence operations."""
    
    @abstractmethod
    def profile_exists(self, user_id: str) -> bool:
        """Check if voice profile exists for user."""
        pass
    
    @abstractmethod
    def load_profile(self, user_id: str) -> Dict[str, Any]:
        """Load voice profile with caching.
        
        Raises:
            VoiceProfileNotFoundError: If profile not found
        """
        pass
    
    @abstractmethod
    def save_profile(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        """Save voice profile and invalidate caches."""
        pass
    
    @abstractmethod
    def create_profile(self, user_id: str, name: str, embedding_dim: int) -> Dict[str, Any]:
        """Create new voice profile."""
        pass
    
    @abstractmethod
    def add_voice_sample(self, user_id: str, embedding: np.ndarray, metrics: Dict[str, float]) -> None:
        """Add voice sample to profile."""
        pass
    
    @abstractmethod
    def update_enrollment_status(self, user_id: str, status: str) -> None:
        """Update enrollment status ('not_enrolled', 'partial', 'enrolled')."""
        pass
    
    @abstractmethod
    def get_embeddings(self, user_id: str) -> List[np.ndarray]:
        """Retrieve all voice embeddings for user."""
        pass
    
    @abstractmethod
    def update_mean_embedding(
        self, 
        user_id: str, 
        mean_unit: np.ndarray, 
        within_var: float, 
        sigma: float
    ) -> None:
        """Update profile statistics."""
        pass
    
    @abstractmethod
    def list_profiles(self) -> List[str]:
        """List all user IDs with voice profiles."""
        pass
    
    @abstractmethod
    def delete_profile(self, user_id: str) -> bool:
        """Delete voice profile for user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if deleted, False if not found
        """
        pass
