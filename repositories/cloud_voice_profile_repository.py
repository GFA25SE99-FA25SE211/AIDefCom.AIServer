"""Cloud Voice Profile Repository - Stores profiles exclusively in Azure Blob.

Replaces local file-based VoiceProfileRepository with Azure Blob Storage backend.
Now includes L1 (memory), L2 (Redis), L3 (Blob) caching hierarchy.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

import numpy as np
from cachetools import LRUCache

from core.exceptions import VoiceProfileNotFoundError
from repositories.azure_blob_repository import AzureBlobRepository
from services.redis_service import RedisService, get_redis_service
from repositories.models.voice_profile_model import VoiceProfile as VoiceProfileModel
import asyncio

logger = logging.getLogger(__name__)


class CloudVoiceProfileRepository:
    """Repository that reads/writes profile JSON with L1/L2/L3 cache."""

    def __init__(self, blob_repo: AzureBlobRepository, redis_service: Optional[RedisService] = None) -> None:
        self.blob_repo = blob_repo
        self.redis_service = redis_service or get_redis_service()
        
        # L1 Cache: In-memory LRU for hot profiles (max 100 profiles ~5MB)
        self._memory_cache: LRUCache = LRUCache(maxsize=100)
        
        # L2 Cache: Redis (5 min TTL)
        self._redis_ttl = 300
        
        logger.info("CloudVoiceProfileRepository initialized with L1/L2/L3 caching")

    # Helper utilities mirror local repo behavior
    def _iso_now(self) -> str:
        from datetime import datetime
        return datetime.utcnow().isoformat(timespec="seconds") + "Z"

    # Interface methods expected by VoiceService
    def profile_exists(self, user_id: str) -> bool:
        return self.blob_repo.profile_exists_in_blob(user_id)

    def load_profile(self, user_id: str) -> Dict[str, Any]:
        """Load profile with L1/L2/L3 cache hierarchy (sync).
        Uses L1 memory cache, then best-effort L2 Redis (if safe), then L3 Blob.
        If running inside an event loop, skips Redis to avoid deadlocks.
        """
        # L1: Memory cache (hot profiles)
        if user_id in self._memory_cache:
            logger.debug(f"L1 cache hit: {user_id}")
            return self._memory_cache[user_id]
        
        # L2: Redis cache (5 min TTL)
        cache_key = f"voice:profile:{user_id}"
        try:
            asyncio.get_running_loop()
            can_use_async = False
        except RuntimeError:
            can_use_async = True

        if can_use_async:
            try:
                cached_data = asyncio.run(self.redis_service.get(cache_key))
                if cached_data:
                    logger.debug(f"L2 cache hit: {user_id}")
                    profile = cached_data if isinstance(cached_data, dict) else json.loads(cached_data)
                    self._memory_cache[user_id] = profile
                    return profile
            except Exception as e:
                logger.debug(f"Redis get failed (continuing without L2): {e}")
        
        # L3: Azure Blob (source of truth)
        try:
            data = self.blob_repo.download_voice_profile(user_id)
            if data is None:
                raise VoiceProfileNotFoundError(f"Profile not found for user: {user_id}")
            
            # Validate and populate caches
            try:
                model = VoiceProfileModel(**data)
                data = model.dict()
            except Exception as ve:
                logger.warning(f"VoiceProfile validation failed for {user_id}: {ve}")
            self._memory_cache[user_id] = data
            if can_use_async:
                try:
                    asyncio.run(self.redis_service.setex(cache_key, self._redis_ttl, json.dumps(data)))
                except Exception as e:
                    logger.debug(f"Redis setex failed (continuing): {e}")
            logger.debug(f"L3 blob fetch + cache updated: {user_id}")
            return data
        except Exception as e:
            logger.error(f"Failed to load profile from blob: {user_id} | {e}")
            # Fallback: check if L1 has stale data (better than nothing)
            if user_id in self._memory_cache:
                logger.warning(f"Using stale L1 cache for {user_id}")
                return self._memory_cache[user_id]
            raise VoiceProfileNotFoundError(f"Profile not found for user: {user_id}") from e

    def save_profile(self, user_id: str, profile_data: Dict[str, Any]) -> None:
        profile_data["updated_at"] = self._iso_now()
        # Upload JSON via blob repo (overwrite)
        self.blob_repo.upload_voice_profile(user_id, profile_data)
        
        # Invalidate L1 cache
        self._memory_cache.pop(user_id, None)
        
        # Invalidate L2 cache
        cache_key = f"voice:profile:{user_id}"
        try:
            asyncio.get_running_loop()
            can_use_async = False
        except RuntimeError:
            can_use_async = True
        if can_use_async:
            try:
                asyncio.run(self.redis_service.delete(cache_key))
            except Exception as e:
                logger.debug(f"Redis delete failed (continuing): {e}")
        
        logger.debug(f"Profile saved + caches invalidated: {user_id}")

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
        logger.info("Cloud profile created | user_id=%s", user_id)
        return profile_data

    def delete_profile(self, user_id: str) -> bool:
        try:
            return self.blob_repo.delete_voice_profile(user_id)
        except Exception as e:
            logger.error("Failed to delete cloud profile | user_id=%s | error=%s", user_id, e)
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
        import uuid
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
