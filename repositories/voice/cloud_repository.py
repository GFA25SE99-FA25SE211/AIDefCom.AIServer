"""Cloud Voice Profile Repository - Stores profiles in Azure Blob with caching."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
from cachetools import LRUCache

from core.exceptions import VoiceProfileNotFoundError
from repositories.azure.blob_repository import AzureBlobRepository
from repositories.interfaces.voice_repository import IVoiceProfileRepository
from repositories.interfaces.redis_service import IRedisService
from repositories.models.voice_profile import VoiceProfile as VoiceProfileModel

logger = logging.getLogger(__name__)


class CloudVoiceProfileRepository(IVoiceProfileRepository):
    """Repository that reads/writes profile JSON with L1/L2/L3 cache."""

    def __init__(self, blob_repo: AzureBlobRepository, redis_service: Optional[IRedisService] = None) -> None:
        self.blob_repo = blob_repo
        self.redis_service = redis_service
        
        # L1 Cache: In-memory LRU (max 100 profiles ~5MB)
        self._memory_cache: LRUCache = LRUCache(maxsize=100)
        self._redis_ttl = 300  # L2 Cache TTL: 5 min
        
        logger.info("CloudVoiceProfileRepository initialized with L1/L2/L3 caching")

    def _iso_now(self) -> str:
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    def profile_exists(self, user_id: str) -> bool:
        return self.blob_repo.profile_exists_in_blob(user_id)

    def load_profile(self, user_id: str) -> Dict[str, Any]:
        """Load profile with L1/L3 cache hierarchy."""
        # L1: Memory cache
        if user_id in self._memory_cache:
            return self._memory_cache[user_id]
        
        # L3: Azure Blob
        try:
            data = self.blob_repo.download_voice_profile(user_id)
            if data is None:
                raise VoiceProfileNotFoundError(f"Profile not found for user: {user_id}")
            
            try:
                model = VoiceProfileModel(**data)
                data = model.dict()
            except Exception as ve:
                logger.warning(f"VoiceProfile validation failed for {user_id}: {ve}")
            
            self._memory_cache[user_id] = data
            return data
        except Exception as e:
            if user_id in self._memory_cache:
                logger.warning(f"Using stale L1 cache for {user_id}")
                return self._memory_cache[user_id]
            raise VoiceProfileNotFoundError(f"Profile not found for user: {user_id}") from e

    def save_profile(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        profile_data["updated_at"] = self._iso_now()
        self.blob_repo.upload_voice_profile(user_id, profile_data)
        self._memory_cache.pop(user_id, None)

    def create_profile(self, user_id: str, name: str, embedding_dim: int) -> Dict[str, Any]:
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
        logger.info(f"Cloud profile created | user_id={user_id}")
        return profile_data

    def delete_profile(self, user_id: str) -> bool:
        try:
            return self.blob_repo.delete_voice_profile(user_id)
        except Exception as e:
            logger.error(f"Failed to delete cloud profile | user_id={user_id} | error={e}")
            return False

    def list_profiles(self) -> List[str]:
        return self.blob_repo.list_voice_profile_ids()

    def add_voice_sample(
        self,
        user_id: str,
        embedding: np.ndarray,
        session_id: str | None = None,
        metrics: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        profile = self.load_profile(user_id)
        sample = {
            "sample_id": str(uuid.uuid4()),
            "session_id": session_id or "default",
            "created_at": self._iso_now(),
            "metrics": metrics,
            "embedding": embedding.tolist(),
        }
        profile.setdefault("voice_samples", []).append(sample)
        profile.setdefault("voice_embeddings", []).append(embedding.tolist())
        profile["enrollment_count"] = len(profile["voice_embeddings"])
        self.save_profile(user_id, profile)
        return sample

    def get_embeddings(self, user_id: str) -> List[np.ndarray]:
        profile = self.load_profile(user_id)
        embs = []
        for arr in profile.get("voice_embeddings", []):
            embs.append(np.asarray(arr, dtype=np.float32).reshape(-1))
        return embs

    def update_mean_embedding(self, user_id: str, mean_embedding: np.ndarray, within_var: float, sigma: float) -> None:
        profile = self.load_profile(user_id)
        profile["mean_embedding"] = mean_embedding.tolist()
        profile["within_var"] = within_var
        profile["sigma"] = sigma
        self.save_profile(user_id, profile)

    def update_enrollment_status(self, user_id: str, status: str) -> None:
        profile = self.load_profile(user_id)
        profile["enrollment_status"] = status
        self.save_profile(user_id, profile)
