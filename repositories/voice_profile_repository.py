"""Voice Profile Repository - Manages voice profile data persistence."""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from core.exceptions import VoiceProfileNotFoundError

logger = logging.getLogger(__name__)


class VoiceProfileRepository:
    """Repository for voice profile file I/O operations."""
    
    def __init__(self, profiles_dir: str | Path) -> None:
        """
        Initialize voice profile repository.
        
        Args:
            profiles_dir: Directory where voice profiles are stored
        """
        self.profiles_dir = Path(profiles_dir)
        self.profiles_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Voice profile repository initialized | dir={self.profiles_dir}")
    
    def _profile_path(self, user_id: str) -> Path:
        """Get path to user's profile file."""
        return self.profiles_dir / f"{user_id}.json"
    
    def _atomic_write_json(self, path: Path, data: Dict[str, Any]) -> None:
        """Atomically write JSON to file using temporary file."""
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp.replace(path)
    
    def _iso_now(self) -> str:
        """Get current UTC timestamp in ISO format."""
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"
    
    def profile_exists(self, user_id: str) -> bool:
        """Check if profile exists for user."""
        return self._profile_path(user_id).exists()
    
    def load_profile(self, user_id: str) -> Dict[str, Any]:
        """
        Load profile from disk.
        
        Args:
            user_id: User identifier
        
        Returns:
            Profile data dictionary
        
        Raises:
            VoiceProfileNotFoundError: If profile doesn't exist
        """
        profile_file = self._profile_path(user_id)
        if not profile_file.exists():
            raise VoiceProfileNotFoundError(f"Profile not found for user: {user_id}")
        
        try:
            with open(profile_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise VoiceProfileNotFoundError(f"Invalid profile data for user {user_id}: {e}") from e
    
    def save_profile(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        """
        Save profile to disk.
        
        Args:
            user_id: User identifier
            profile_data: Profile data to save
        """
        profile_file = self._profile_path(user_id)
        profile_data["updated_at"] = self._iso_now()
        self._atomic_write_json(profile_file, profile_data)
        logger.debug(f"Profile saved | user_id={user_id}")
    
    def create_profile(
        self,
        user_id: str,
        name: str,
        embedding_dim: int,
    ) -> Dict[str, Any]:
        """
        Create new profile.
        
        Args:
            user_id: User identifier
            name: User's name
            embedding_dim: Embedding vector dimension
        
        Returns:
            New profile data
        """
        profile_data = {
            "user_id": user_id,
            "name": name,
            "enrollment_status": "not_enrolled",
            "voice_samples": [],
            "voice_embeddings": [],
            "embedding_dim": embedding_dim,
            "created_at": self._iso_now(),
            "updated_at": self._iso_now(),
            "schema_version": 5,
            "mean_embedding": None,
            "within_var": None,
            "sigma": None,
            "enrollment_count": 0,
        }
        
        self.save_profile(user_id, profile_data)
        logger.info(f"Profile created | user_id={user_id}")
        return profile_data
    
    def delete_profile(self, user_id: str) -> bool:
        """
        Delete profile from disk.
        
        Args:
            user_id: User identifier
        
        Returns:
            True if deleted, False if not found
        """
        profile_file = self._profile_path(user_id)
        if profile_file.exists():
            profile_file.unlink()
            logger.info(f"Profile deleted | user_id={user_id}")
            return True
        return False
    
    def list_profiles(self) -> List[str]:
        """
        List all user IDs with profiles.
        
        Returns:
            List of user IDs
        """
        return [p.stem for p in self.profiles_dir.glob("*.json")]
    
    def add_voice_sample(
        self,
        user_id: str,
        embedding: np.ndarray,
        session_id: str | None = None,
        metrics: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Add voice sample to profile.
        
        Args:
            user_id: User identifier
            embedding: Voice embedding vector
            session_id: Optional session identifier
            metrics: Optional quality metrics
        
        Returns:
            The created sample data
        """
        profile = self.load_profile(user_id)
        
        sample_data = {
            "sample_id": str(uuid.uuid4()),
            "session_id": session_id or "default",
            "created_at": self._iso_now(),
            "metrics": metrics,
            "embedding": embedding.tolist(),
        }
        
        if "voice_samples" not in profile:
            profile["voice_samples"] = []
        
        profile["voice_samples"].append(sample_data)
        
        # Update voice_embeddings list
        if "voice_embeddings" not in profile:
            profile["voice_embeddings"] = []
        profile["voice_embeddings"].append(embedding.tolist())
        
        # Update enrollment count
        profile["enrollment_count"] = len(profile["voice_embeddings"])
        
        self.save_profile(user_id, profile)
        return sample_data
    
    def get_embeddings(self, user_id: str) -> List[np.ndarray]:
        """
        Get all voice embeddings for user.
        
        Args:
            user_id: User identifier
        
        Returns:
            List of embedding vectors
        """
        profile = self.load_profile(user_id)
        embeddings = []
        
        for emb_data in profile.get("voice_embeddings", []):
            emb = np.asarray(emb_data, dtype=np.float32).reshape(-1)
            embeddings.append(emb)
        
        return embeddings
    
    def update_mean_embedding(
        self,
        user_id: str,
        mean_embedding: np.ndarray,
        within_var: float,
        sigma: float,
    ) -> None:
        """
        Update profile statistics.
        
        Args:
            user_id: User identifier
            mean_embedding: Mean embedding vector
            within_var: Within-class variance
            sigma: Standard deviation
        """
        profile = self.load_profile(user_id)
        profile["mean_embedding"] = mean_embedding.tolist()
        profile["within_var"] = within_var
        profile["sigma"] = sigma
        self.save_profile(user_id, profile)
    
    def update_enrollment_status(self, user_id: str, status: str) -> None:
        """
        Update enrollment status.
        
        Args:
            user_id: User identifier
            status: New enrollment status
        """
        profile = self.load_profile(user_id)
        profile["enrollment_status"] = status
        self.save_profile(user_id, profile)
